import { parse as parseHtml } from "node-html-parser";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { parseArgs } from "node:util";
import OpenAI from "openai";
import type {
  FunctionTool,
  Response,
  ResponseFunctionToolCallItem,
  ResponseInputItem,
  WebSearchTool,
} from "openai/resources/responses/responses";

interface CliOptions {
  keywords: string[];
  minPrimary: number;
  minSecondary: number;
  pdfLimit: number;
  maxToolPasses: number;
  outputPath?: string;
  userLocation?: WebSearchTool["user_location"];
}

interface SourceEntry {
  classification: "primary" | "secondary";
  title: string;
  url: string;
  summary: string;
  publisher?: string;
  publishedDate?: string;
  whyTrusted: string;
  retrievalMethod: "web_search" | "pdf_search";
}

interface KeywordReport {
  keyword: string;
  summary: string;
  stats: {
    primaryCount: number;
    secondaryCount: number;
  };
  sources: SourceEntry[];
  pendingGaps?: string;
  error?: string;
}

interface StructuredPayload {
  keyword: string;
  summary: string;
  stats: { primaryCount: number; secondaryCount: number };
  sources: SourceEntry[];
  pendingGaps: string;
}

interface PdfSearchArgs {
  query: string;
  max_results?: number;
  site_filters?: string[];
}

interface PdfSearchHit {
  title: string;
  url: string;
  snippet: string;
  domain: string;
  publishedDate?: string;
}

const DEFAULT_PRIMARY_MIN = 2;
const DEFAULT_SECONDARY_MIN = 2;
const DEFAULT_PDF_LIMIT = 5;
const DEFAULT_TOOL_PASSES = 6;
const PDF_SEARCH_TOOL_NAME = "pdf_search";

const structuredOutputSchema = {
  name: "source_harvest_payload",
  schema: {
    type: "object",
    additionalProperties: false,
    properties: {
      keyword: { type: "string" },
      summary: { type: "string" },
      stats: {
        type: "object",
        additionalProperties: false,
        properties: {
          primaryCount: { type: "integer" },
          secondaryCount: { type: "integer" },
        },
        required: ["primaryCount", "secondaryCount"],
      },
      pendingGaps: { type: "string" },
      sources: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            classification: { type: "string", enum: ["primary", "secondary"] },
            title: { type: "string" },
            url: { type: "string" },
            publisher: { type: "string" },
            publishedDate: { type: "string" },
            summary: { type: "string" },
            whyTrusted: { type: "string" },
            retrievalMethod: {
              type: "string",
              enum: ["web_search", "pdf_search"],
            },
          },
          required: [
            "classification",
            "title",
            "url",
            "publisher",
            "publishedDate",
            "summary",
            "whyTrusted",
            "retrievalMethod",
          ],
        },
      },
    },
    required: ["keyword", "summary", "stats", "pendingGaps", "sources"],
  },
};

const pdfSearchTool: FunctionTool = {
  type: "function",
  name: PDF_SEARCH_TOOL_NAME,
  description:
    "Bing検索(filetype:pdf)を用いて公式ドキュメントなどのPDFを探します。行政・企業・研究機関などのサイトを優先してください。",
  strict: false,
  parameters: {
    type: "object",
    additionalProperties: false,
    properties: {
      query: { type: "string", description: "検索キーワード" },
      max_results: {
        type: "integer",
        minimum: 1,
        maximum: 10,
        description: "取得したい最大ヒット数 (最大10)",
      },
      site_filters: {
        type: "array",
        description: "site:example.com のようなフィルタで優先したいドメイン",
        items: { type: "string" },
      },
    },
    required: ["query"],
  },
};

const systemPrompt = `\
あなたは調査専門アシスタントです。一次情報(政府・企業・公式レポート等の元データ)と、信頼できる二次情報(報道・シンクタンク等の独自分析)を区別して収集してください。
- web_search ツールで幅広く探索し、公式ソースかどうか必ず確認する。
- 記事が孫引きの場合は必ずオリジナルの一次・二次情報を追加検索して辿る。
- PDFが必要/予想される場合は pdf_search を呼び出して原典を取得する。
- 指定件数を満たすまで必要なだけ検索⇄検証ループを繰り返す。
- 出力は指定の JSON スキーマに従い、各ソースが一次/二次のどちらかを明示する。
- 信頼根拠(公式発表/オリジナル資料/著名報道など)を whyTrusted に書く。
- publisher や publishedDate が不明な場合は "不明" 等のテキストを入れて必ず埋める。
- pendingGaps には残課題/取得できなかった情報を必ず文章で記す。ギャップが無ければ "なし" と記載する。
`;

async function main() {
  const options = parseCliOptions();
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error("OPENAI_API_KEY が設定されていません。");
    process.exit(1);
  }

  const client = new OpenAI({ apiKey });
  const reports: KeywordReport[] = [];

  for (const keyword of options.keywords) {
    console.log(`🔎 キーワード「${keyword}」の調査を開始します`);
    try {
      const report = await harvestKeyword(client, keyword, options);
      console.log(
        `✅ ${keyword}: 一次 ${report.stats.primaryCount} 件 / 二次 ${report.stats.secondaryCount} 件`
      );
      reports.push(report);
    } catch (error) {
      const message = error instanceof Error ? error.message : "不明なエラー";
      console.error(`❌ ${keyword}: ${message}`);
      reports.push({
        keyword,
        summary: "収集に失敗しました",
        stats: { primaryCount: 0, secondaryCount: 0 },
        sources: [],
        error: message,
      });
    }
  }

  const payload = { generatedAt: new Date().toISOString(), reports };

  if (options.outputPath) {
    ensureParentDir(options.outputPath);
    writeFileSync(
      options.outputPath,
      JSON.stringify(payload, null, 2),
      "utf-8"
    );
    console.log(`📁 結果を ${options.outputPath} に保存しました`);
  } else {
    console.log(JSON.stringify(payload, null, 2));
  }
}

async function harvestKeyword(
  client: OpenAI,
  keyword: string,
  options: CliOptions
): Promise<KeywordReport> {
  const userPrompt = buildUserPrompt(keyword, options);
  const initialInput: ResponseInputItem[] = [
    {
      role: "user",
      type: "message",
      content: [
        {
          type: "input_text",
          text: userPrompt,
        },
      ],
    },
  ];

  const tools: Response["tools"] = [
    {
      type: "web_search",
      user_location: options.userLocation,
      search_context_size: "medium",
    },
    pdfSearchTool,
  ];

  const baseParams = {
    model: "gpt-5-mini",
    instructions: systemPrompt,
    tools,
    parallel_tool_calls: true,
    text: {
      format: {
        type: "json_schema",
        name: structuredOutputSchema.name,
        schema: structuredOutputSchema.schema,
        strict: true,
      },
      verbosity: "medium",
    },
  } as const;

  const response = await runResponseLoop(client, {
    baseParams,
    initialInput,
    maxPasses: options.maxToolPasses,
  });

  const parsed = parseStructuredPayload(response.output_text);
  return {
    keyword: parsed.keyword,
    summary: parsed.summary,
    stats: parsed.stats,
    sources: parsed.sources,
    pendingGaps: parsed.pendingGaps,
  };
}

function buildUserPrompt(keyword: string, options: CliOptions): string {
  return (
    `対象キーワード: ${keyword}\n` +
    `一次情報目標: ${options.minPrimary}件以上\n` +
    `二次情報目標: ${options.minSecondary}件以上\n` +
    `PDFが有用な場合は pdf_search を必ず呼び出すこと。`
  );
}

function parseStructuredPayload(rawText: string): StructuredPayload {
  try {
    return JSON.parse(rawText) as StructuredPayload;
  } catch (error) {
    throw new Error(
      `JSON出力の解析に失敗しました: ${(error as Error).message}`
    );
  }
}

async function runResponseLoop(
  client: OpenAI,
  params: {
    baseParams: {
      model: string;
      instructions: string;
      tools: Response["tools"];
      parallel_tool_calls: boolean;
      text: Response["text"];
    };
    initialInput: ResponseInputItem[];
    maxPasses: number;
  }
): Promise<Response> {
  let passCount = 0;
  let response = await client.responses.create({
    ...params.baseParams,
    input: params.initialInput,
  });

  while (true) {
    const pendingCalls = extractFunctionCalls(response);
    if (pendingCalls.length === 0) {
      if (response.status === "in_progress") {
        await delay(1000);
        response = await client.responses.retrieve(response.id);
        continue;
      }
      return response;
    }

    if (passCount >= params.maxPasses) {
      throw new Error("ツール呼び出し回数が上限に達しました");
    }

    passCount += 1;
    const toolOutputs = await Promise.all(
      pendingCalls.map(
        async (call): Promise<ResponseInputItem.FunctionCallOutput> => {
          const payload = await handleFunctionCall(call);
          return {
            type: "function_call_output",
            call_id: call.id,
            output: JSON.stringify(payload),
          };
        }
      )
    );

    response = await client.responses.create({
      ...params.baseParams,
      previous_response_id: response.id,
      input: toolOutputs,
    });
  }
}

function extractFunctionCalls(
  response: Response
): ResponseFunctionToolCallItem[] {
  const calls: ResponseFunctionToolCallItem[] = [];
  for (const item of response.output ?? []) {
    if (item.type === "function_call" && typeof item.id === "string") {
      if (item.status && item.status !== "in_progress") {
        continue;
      }
      calls.push(item as ResponseFunctionToolCallItem);
    }
  }
  return calls;
}

async function handleFunctionCall(call: ResponseFunctionToolCallItem) {
  if (call.name !== PDF_SEARCH_TOOL_NAME) {
    return { error: `未対応の関数です: ${call.name}` };
  }

  let args: PdfSearchArgs;
  try {
    args = JSON.parse(call.arguments ?? "{}") as PdfSearchArgs;
  } catch (error) {
    return { error: `引数の解析に失敗: ${(error as Error).message}` };
  }

  if (!args.query || typeof args.query !== "string") {
    return { error: "query が指定されていません" };
  }

  const limit = clampNumber(args.max_results ?? DEFAULT_PDF_LIMIT, 1, 10);
  try {
    const hits = await pdfSearch(args.query, limit, args.site_filters ?? []);
    return { query: args.query, hits };
  } catch (error) {
    return { query: args.query, hits: [], error: (error as Error).message };
  }
}

async function pdfSearch(
  query: string,
  limit: number,
  siteFilters: string[]
): Promise<PdfSearchHit[]> {
  const filterSuffix = siteFilters
    .filter(Boolean)
    .map((domain) => `site:${domain}`)
    .join(" ");
  const q = ["filetype:pdf", query.trim(), filterSuffix]
    .filter(Boolean)
    .join(" ");
  const searchParams = new URLSearchParams({ q, setlang: "ja" });
  const url = `https://www.bing.com/search?${searchParams.toString()}`;

  const response = await fetch(url, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
      "Accept-Language": "ja,en-US;q=0.8",
    },
    redirect: "follow",
  });

  if (!response.ok) {
    throw new Error(`Bing検索が失敗しました (status ${response.status})`);
  }

  const html = await response.text();
  const root = parseHtml(html);
  const items = root.querySelectorAll("li.b_algo");
  const hits: PdfSearchHit[] = [];

  for (const item of items) {
    if (hits.length >= limit) {
      break;
    }
    const anchor = item.querySelector("h2 > a");
    if (!anchor) continue;
    const rawHref = anchor.getAttribute("href");
    if (!rawHref) continue;
    const resolved = normalizeBingUrl(rawHref);
    if (!resolved || !looksLikePdf(resolved)) {
      continue;
    }

    const title = sanitizeText(anchor.text);
    const snippet = sanitizeText(item.querySelector(".b_caption")?.text ?? "");
    const date = sanitizeText(item.querySelector(".news_dt")?.text ?? "");
    const domain = safeHostname(resolved);

    hits.push({
      title,
      url: resolved,
      snippet,
      domain,
      publishedDate: date || undefined,
    });
  }

  if (!hits.length) {
    throw new Error("PDFの検索結果が得られませんでした");
  }

  return hits.slice(0, limit);
}

function normalizeBingUrl(url: string): string | null {
  try {
    const parsed = new URL(url, "https://www.bing.com");
    const target = parsed.searchParams.get("u");
    if (target) {
      return decodeURIComponent(target);
    }
    return parsed.toString();
  } catch {
    return url.startsWith("http") ? url : null;
  }
}

function looksLikePdf(url: string): boolean {
  try {
    const parsed = new URL(url);
    return /\.pdf($|[?#])/i.test(parsed.pathname);
  } catch {
    return url.toLowerCase().includes(".pdf");
  }
}

function sanitizeText(input: string): string {
  return input.replace(/\s+/g, " ").trim();
}

function safeHostname(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return "";
  }
}

function clampNumber(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(Math.max(value, min), max);
}

function parseCliOptions(): CliOptions {
  const { values, positionals } = parseArgs({
    options: {
      keyword: { type: "string", multiple: true, short: "k" },
      "primary-min": { type: "string" },
      "secondary-min": { type: "string" },
      "pdf-limit": { type: "string" },
      "max-passes": { type: "string" },
      output: { type: "string", short: "o" },
      country: { type: "string" },
      region: { type: "string" },
      city: { type: "string" },
      timezone: { type: "string" },
    },
    allowPositionals: true,
  });

  const keywords = [...(values.keyword ?? []), ...positionals]
    .map((text) => text.trim())
    .filter(Boolean);

  if (!keywords.length) {
    console.error("キーワードを --keyword または位置引数で指定してください。");
    process.exit(1);
  }

  const minPrimary = parsePositiveInt(
    values["primary-min"],
    DEFAULT_PRIMARY_MIN
  );
  const minSecondary = parsePositiveInt(
    values["secondary-min"],
    DEFAULT_SECONDARY_MIN
  );
  const pdfLimit = clampNumber(
    parsePositiveInt(values["pdf-limit"], DEFAULT_PDF_LIMIT),
    1,
    10
  );
  const maxToolPasses = clampNumber(
    parsePositiveInt(values["max-passes"], DEFAULT_TOOL_PASSES),
    1,
    12
  );

  const hasLocation =
    values.country || values.region || values.city || values.timezone;
  const userLocation = hasLocation
    ? {
        type: "approximate" as const,
        country: values.country ?? null,
        region: values.region ?? null,
        city: values.city ?? null,
        timezone: values.timezone ?? null,
      }
    : undefined;

  return {
    keywords,
    minPrimary,
    minSecondary,
    pdfLimit,
    maxToolPasses,
    outputPath: values.output,
    userLocation,
  };
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ensureParentDir(pathStr: string) {
  try {
    const dir = dirname(pathStr);
    if (dir && dir !== ".") {
      mkdirSync(dir, { recursive: true });
    }
  } catch (error) {
    console.warn(
      `出力ディレクトリ作成に失敗しました: ${(error as Error).message}`
    );
  }
}

main().catch((error) => {
  console.error(
    "致命的なエラーが発生しました:",
    error instanceof Error ? error.message : error
  );
  process.exit(1);
});

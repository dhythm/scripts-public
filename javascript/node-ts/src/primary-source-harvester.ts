import { parse as parseHtml } from "node-html-parser";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { parseArgs } from "node:util";
import OpenAI from "openai";
import type {
  FunctionTool,
  Response,
  ResponseFunctionToolCallItem,
  ResponseFunctionWebSearch,
  ResponseInputItem,
  WebSearchTool,
} from "openai/resources/responses/responses";

type PdfStrategy = "auto" | "always" | "never";

interface CliOptions {
  keywords: string[];
  minPrimary: number;
  minSecondary: number;
  pdfLimit: number;
  maxToolPasses: number;
  outputPath?: string;
  userLocation?: WebSearchTool["user_location"];
  debug: boolean;
  responseTimeoutMs: number;
  pdfStrategy: PdfStrategy;
}

interface SourceEntry {
  classification: "primary" | "secondary";
  title: string;
  url: string;
  summary: string;
  excerpt: string;
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
const DEFAULT_PDF_STRATEGY: PdfStrategy = "auto";
const DEFAULT_RESPONSE_TIMEOUT_MS = 120_000;
const PDF_SEARCH_TOOL_NAME = "pdf_search";
const DEBUG_DIR = "reports/debug";
const POLL_INTERVAL_MS = 1500;

type ConsoleMethodName = "log" | "info" | "warn" | "error";

patchConsoleWithTimestamps();
const RUN_STARTED_AT = Date.now();
process.once("exit", () => {
  const elapsedMs = Date.now() - RUN_STARTED_AT;
  console.log(`⏱️ トータル実行時間: ${(elapsedMs / 1000).toFixed(1)} 秒`);
});

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
            excerpt: { type: "string" },
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
            "excerpt",
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
- source.excerpt には必ずページ本体から引用可能な本文(4〜8文、最大400語程度)を記載し、単なるリンク集やランディングページのみの記述は採用しない。
- ページ本文に具体的なデータ・記述が無い場合、そのソースは採用せず別のソースを探す。
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
    ...(options.pdfStrategy === "never" ? [] : [pdfSearchTool]),
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

  const response = await executeResponseWorkflow(client, {
    baseParams,
    initialInput,
    maxPasses: options.maxToolPasses,
    debug: options.debug,
    keyword,
    responseTimeoutMs: options.responseTimeoutMs,
  });

  if (options.debug) {
    try {
      const responseDumpPath = dumpRawResponse(
        keyword,
        "response-object",
        JSON.stringify(response, null, 2),
        { enabled: true }
      );
      if (responseDumpPath) {
        debugLog(
          options.debug,
          `[${keyword}] APIレスポンス全体を ${responseDumpPath} に保存しました`
        );
      }
    } catch (error) {
      debugLog(
        options.debug,
        `[${keyword}] レスポンス保存失敗: ${(error as Error).message}`
      );
    }
  }

  const rawOutput = extractStructuredOutputText(response);
  const dumpPath = dumpRawResponse(keyword, "response", rawOutput, {
    enabled: options.debug,
  });
  if (dumpPath) {
    debugLog(
      options.debug,
      `[${keyword}] モデル出力を ${dumpPath} に保存しました`
    );
  }
  if (options.debug) {
    debugLog(
      options.debug,
      `[${keyword}] 出力プレビュー: ${truncate(rawOutput, 200)}`
    );
  }

  const parsed = parseStructuredPayload(rawOutput, {
    keyword,
    debug: options.debug,
  });
  const {
    validSources,
    warnings,
    primaryCount,
    secondaryCount,
  } = sanitizeSourcesForQuotes(keyword, parsed.sources);
  const pendingGaps = [parsed.pendingGaps, ...warnings]
    .map((text) => text.trim())
    .filter(Boolean)
    .join(" / ") || "なし";
  return {
    keyword: parsed.keyword,
    summary: parsed.summary,
    stats: {
      primaryCount,
      secondaryCount,
    },
    sources: validSources,
    pendingGaps,
  };
}

function buildUserPrompt(keyword: string, options: CliOptions): string {
  const pdfPolicyDescription = (() => {
    switch (options.pdfStrategy) {
      case "never":
        return "PDF取得は不要。必要な情報はウェブ本文から引用すること。";
      case "always":
        return "各テーマでPDFが存在する場合は必ず取得し、本文から引用すること。";
      default:
        return "PDFが有用な場合のみ取得し、不要ならウェブ本文のみに頼ること。";
    }
  })();

  return (
    `対象キーワード: ${keyword}\n` +
    `一次情報目標: ${options.minPrimary}件以上\n` +
    `二次情報目標: ${options.minSecondary}件以上\n` +
    `PDF方針: ${pdfPolicyDescription}`
  );
}

function parseStructuredPayload(
  rawText: string,
  context: { keyword: string; debug: boolean }
): StructuredPayload {
  try {
    const parsed = JSON.parse(rawText) as StructuredPayload;
    if (!isStructuredPayloadCandidate(parsed)) {
      throw new Error("期待する構造化スキーマと一致しません");
    }
    return parsed;
  } catch (error) {
    const dumpPath = dumpRawResponse(context.keyword, "parse-error", rawText, {
      always: true,
    });
    console.error(
      `⚠️ JSON出力の解析に失敗しました (${context.keyword}). rawを ${dumpPath} に保存しました`
    );
    if (context.debug) {
      console.error(`[debug] raw output: ${rawText}`);
    }
    const message = `JSON出力の解析に失敗しました: ${
      (error as Error).message
    } (raw: ${dumpPath})`;
    throw new Error(message);
  }
}

function extractStructuredOutputText(response: Response): string {
  const direct = (response.output_text ?? "").trim();
  if (direct) {
    return direct;
  }

  const payloadObject = findStructuredPayload(response.output);
  if (payloadObject) {
    return JSON.stringify(payloadObject);
  }

  const fallbackString = findJsonLikeString(response.output);
  if (fallbackString) {
    return fallbackString;
  }

  throw new Error(
    `構造化出力が見つかりませんでした (response_id=${response.id})`
  );
}

function findStructuredPayload(
  value: unknown,
  seen: WeakSet<object> = new WeakSet()
): StructuredPayload | null {
  if (isStructuredPayloadCandidate(value)) {
    return value;
  }

  if (Array.isArray(value)) {
    if (seen.has(value)) {
      return null;
    }
    seen.add(value);
    for (const entry of value) {
      const found = findStructuredPayload(entry, seen);
      if (found) {
        return found;
      }
    }
    return null;
  }

  if (value && typeof value === "object") {
    if (seen.has(value as object)) {
      return null;
    }
    seen.add(value as object);
    for (const nested of Object.values(value as Record<string, unknown>)) {
      const found = findStructuredPayload(nested, seen);
      if (found) {
        return found;
      }
    }
  }

  return null;
}

function isStructuredPayloadCandidate(
  value: unknown
): value is StructuredPayload {
  if (!value || typeof value !== "object") {
    return false;
  }

  const record = value as Record<string, unknown>;
  if (
    typeof record.keyword !== "string" ||
    typeof record.summary !== "string" ||
    typeof record.pendingGaps !== "string" ||
    !Array.isArray(record.sources)
  ) {
    return false;
  }

  const stats = record.stats as Record<string, unknown> | undefined;
  if (
    !stats ||
    typeof stats !== "object" ||
    typeof stats.primaryCount !== "number" ||
    typeof stats.secondaryCount !== "number"
  ) {
    return false;
  }

  const sources = record.sources as SourceEntry[];
  if (
    !Array.isArray(sources) ||
    sources.some(
      (source) =>
        !source ||
        typeof source !== "object" ||
        typeof source.excerpt !== "string" ||
        source.excerpt.trim().length === 0
    )
  ) {
    return false;
  }

  return true;
}

function findJsonLikeString(
  value: unknown,
  seen: WeakSet<object> = new WeakSet()
): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!looksLikeJson(trimmed)) {
      return null;
    }
    try {
      const parsed = JSON.parse(trimmed);
      if (isStructuredPayloadCandidate(parsed)) {
        return trimmed;
      }
    } catch {
      return null;
    }
    return null;
  }

  if (Array.isArray(value)) {
    if (seen.has(value)) {
      return null;
    }
    seen.add(value);
    for (const entry of value) {
      const found = findJsonLikeString(entry, seen);
      if (found) {
        return found;
      }
    }
    return null;
  }

  if (value && typeof value === "object") {
    if (seen.has(value as object)) {
      return null;
    }
    seen.add(value as object);
    for (const nested of Object.values(value as Record<string, unknown>)) {
      const found = findJsonLikeString(nested, seen);
      if (found) {
        return found;
      }
    }
  }

  return null;
}

function looksLikeJson(text: string): boolean {
  if (!text) {
    return false;
  }
  const trimmed = text.trim();
  if (!trimmed) {
    return false;
  }
  return (
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  );
}

function patchConsoleWithTimestamps() {
  const methods: ConsoleMethodName[] = ["log", "info", "warn", "error"];
  for (const method of methods) {
    const original = console[method].bind(console) as (
      ...args: unknown[]
    ) => void;
    console[method] = ((...args: unknown[]) => {
      original(`[${new Date().toISOString()}]`, ...args);
    }) as (typeof console)[typeof method];
  }
}

function formatUsage(usage?: Response["usage"] | null): string {
  if (!usage) return "";
  const parts: string[] = [];
  if (typeof usage.input_tokens === "number") {
    parts.push(`in=${usage.input_tokens}`);
  }
  if (typeof usage.output_tokens === "number") {
    parts.push(`out=${usage.output_tokens}`);
  }
  return parts.length ? `tokens(${parts.join("/")})` : "";
}

function sanitizeSourcesForQuotes(keyword: string, sources: SourceEntry[]) {
  const MIN_EXCERPT_CHARS = 100;
  const warnings: string[] = [];
  const validSources: SourceEntry[] = [];
  let primaryCount = 0;
  let secondaryCount = 0;

  for (const source of sources) {
    const excerpt = source.excerpt?.trim() ?? "";
    if (excerpt.length < MIN_EXCERPT_CHARS) {
      warnings.push(
        `"${source.title}" は引用できる本文が不足 (${excerpt.length}文字) のため除外`
      );
      continue;
    }
    validSources.push(source);
    if (source.classification === "primary") {
      primaryCount += 1;
    } else if (source.classification === "secondary") {
      secondaryCount += 1;
    }
  }

  if (!validSources.length) {
    warnings.push(
      `[${keyword}] 引用要件を満たすソースが無かったため追加調査が必要です`
    );
  }

  return { validSources, warnings, primaryCount, secondaryCount };
}

async function executeResponseWorkflow(
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
    debug: boolean;
    keyword: string;
    responseTimeoutMs: number;
  }
): Promise<Response> {
  let passCount = 0;
  let previousResponseId: string | undefined;
  const handledFunctionCallIds = new Set<string>();

  let response = await submitAndPollResponse(client, {
    baseParams: params.baseParams,
    input: params.initialInput,
    keyword: params.keyword,
    debug: params.debug,
    timeoutMs: params.responseTimeoutMs,
  });

  while (true) {
    logToolActivities(response, params.keyword);

    const pendingCalls = extractFunctionCalls(response, handledFunctionCallIds);
    if (pendingCalls.length === 0) {
      if (response.status === "completed") {
        return response;
      }
      if (response.status === "failed" || response.status === "cancelled") {
        const message =
          response.error?.message ?? "OpenAIレスポンスが失敗しました";
        throw new Error(`[${params.keyword}] モデル実行失敗: ${message}`);
      }
      throw new Error(
        `[${params.keyword}] モデルが終了しましたがツール呼び出しも完了していません (status=${response.status})`
      );
    }

    if (passCount >= params.maxPasses) {
      throw new Error("ツール呼び出し回数が上限に達しました");
    }

    passCount += 1;
    console.log(
      `🛠️ [${params.keyword}] ツール呼び出し #${passCount}: ${pendingCalls
        .map((call) => call.name)
        .join(", ")}`
    );
    const toolOutputs = await Promise.all(
      pendingCalls.map(
        async (call): Promise<ResponseInputItem.FunctionCallOutput> => {
          const payload = await handleFunctionCall(call, {
            debug: params.debug,
            keyword: params.keyword,
          });
          handledFunctionCallIds.add(call.id);
          return {
            type: "function_call_output",
            call_id: call.call_id ?? call.id,
            output: JSON.stringify(payload),
          };
        }
      )
    );

    previousResponseId = response.id;
    response = await submitAndPollResponse(client, {
      baseParams: params.baseParams,
      input: toolOutputs,
      previousResponseId,
      keyword: params.keyword,
      debug: params.debug,
      timeoutMs: params.responseTimeoutMs,
    });
  }
}

async function submitAndPollResponse(
  client: OpenAI,
  args: {
    baseParams: {
      model: string;
      instructions: string;
      tools: Response["tools"];
      parallel_tool_calls: boolean;
      text: Response["text"];
    };
    input: ResponseInputItem[];
    previousResponseId?: string;
    keyword: string;
    debug: boolean;
    timeoutMs: number;
  }
): Promise<Response> {
  const requestPayload = {
    ...args.baseParams,
    input: args.input,
    background: true,
    ...(args.previousResponseId
      ? { previous_response_id: args.previousResponseId }
      : {}),
  };
  const requestStartedAt = Date.now();
  const initialResponse = await client.responses.create(requestPayload);
  console.log(`🚀 [${args.keyword}] リクエスト送信: id=${initialResponse.id}`);
  const finalResponse = await pollResponseUntilTerminal(
    client,
    initialResponse.id,
    {
      keyword: args.keyword,
    },
    args.timeoutMs
  );
  const elapsedSec = ((Date.now() - requestStartedAt) / 1000).toFixed(1);
  const tokenInfo = formatUsage(finalResponse.usage);
  console.log(
    `✅ [${args.keyword}] リクエスト id=${
      finalResponse.id
    } 完了 (${elapsedSec}秒${tokenInfo ? ", " + tokenInfo : ""})`
  );
  return finalResponse;
}

async function pollResponseUntilTerminal(
  client: OpenAI,
  responseId: string,
  context: { keyword: string },
  timeoutMs: number
): Promise<Response> {
  let lastStatus: Response["status"] | undefined;
  const startedAt = Date.now();
  while (true) {
    const response = await client.responses.retrieve(responseId);
    if (response.status !== lastStatus) {
      console.log(`⌛ [${context.keyword}] ステータス: ${response.status}`);
      lastStatus = response.status;
    }

    if (response.status === "in_progress" || response.status === "queued") {
      if (Date.now() - startedAt >= timeoutMs) {
        console.warn(
          `[${context.keyword}] レスポンス待機が ${timeoutMs}ms を超過しました。キャンセルを試みます`
        );
        try {
          await client.responses.cancel(responseId);
        } catch (error) {
          console.warn(
            `[${context.keyword}] キャンセルに失敗: ${(error as Error).message}`
          );
        }
        throw new Error(
          `[${context.keyword}] モデル応答がタイムアウトしました (${timeoutMs}ms)`
        );
      }
      await delay(POLL_INTERVAL_MS);
      continue;
    }

    if (response.status === "failed") {
      const message =
        response.error?.message ?? "OpenAIレスポンスが失敗しました";
      throw new Error(`[${context.keyword}] モデル実行失敗: ${message}`);
    }

    if (response.status === "cancelled") {
      throw new Error(`[${context.keyword}] モデル実行がキャンセルされました`);
    }

    return response;
  }
}

function extractFunctionCalls(
  response: Response,
  handledIds: Set<string>
): ResponseFunctionToolCallItem[] {
  const calls: ResponseFunctionToolCallItem[] = [];
  for (const item of response.output ?? []) {
    if (item.type === "function_call" && typeof item.id === "string") {
      if (handledIds.has(item.id)) {
        continue;
      }
      calls.push(item as ResponseFunctionToolCallItem);
    }
  }
  return calls;
}

function logToolActivities(response: Response, keyword: string) {
  const logs: string[] = [];
  for (const item of response.output ?? []) {
    if (item.type === "function_call") {
      logs.push(
        `📎 [${keyword}] 関数ツール '${item.name}' を呼び出し (status=${
          item.status ?? "pending"
        })`
      );
    } else if (isWebSearchCall(item)) {
      logs.push(`🌐 [${keyword}] web_search ステータス: ${item.status}`);
    }
  }

  for (const line of logs) {
    console.log(line);
  }
}

function isWebSearchCall(
  item: Response["output"][number]
): item is ResponseFunctionWebSearch {
  return (item as ResponseFunctionWebSearch)?.type === "web_search_call";
}

async function handleFunctionCall(
  call: ResponseFunctionToolCallItem,
  context: { debug: boolean; keyword: string }
) {
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
  debugLog(
    context.debug,
    `[${context.keyword}] pdf_search start: query="${
      args.query
    }" limit=${limit} filters=${(args.site_filters ?? []).join(",")}`
  );
  try {
    const hits = await pdfSearch(args.query, limit, args.site_filters ?? []);
    debugLog(
      context.debug,
      `[${context.keyword}] pdf_search hits=${hits.length}`
    );
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
    return [];
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
      "pdf-strategy": { type: "string" },
      "response-timeout": { type: "string" },
      output: { type: "string", short: "o" },
      country: { type: "string" },
      region: { type: "string" },
      city: { type: "string" },
      timezone: { type: "string" },
      debug: { type: "boolean" },
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
  const responseTimeoutMs = clampNumber(
    parsePositiveInt(values["response-timeout"], DEFAULT_RESPONSE_TIMEOUT_MS),
    10_000,
    600_000
  );
  const pdfStrategy = parsePdfStrategy(values["pdf-strategy"]);

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
    debug: Boolean(values.debug),
    responseTimeoutMs,
    pdfStrategy,
  };
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parsePdfStrategy(value: string | undefined): PdfStrategy {
  if (!value) return DEFAULT_PDF_STRATEGY;
  const normalized = value.toLowerCase();
  if (
    normalized === "auto" ||
    normalized === "always" ||
    normalized === "never"
  ) {
    return normalized as PdfStrategy;
  }
  console.warn(
    `未知の pdf-strategy '${value}' が指定されたため '${DEFAULT_PDF_STRATEGY}' を使用します`
  );
  return DEFAULT_PDF_STRATEGY;
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

function debugLog(enabled: boolean, message: string) {
  if (enabled) {
    console.log(`[debug] ${message}`);
  }
}

function dumpRawResponse(
  keyword: string,
  reason: string,
  rawText: string,
  options: { enabled?: boolean; always?: boolean } = {}
): string | null {
  const shouldWrite = options.always || options.enabled;
  if (!shouldWrite) {
    return null;
  }
  const safeName = slugify(keyword) || "keyword";
  const filename = `${safeName}-${reason}-${Date.now()}.txt`;
  const filePath = join(DEBUG_DIR, filename);
  ensureParentDir(filePath);
  writeFileSync(filePath, rawText ?? "", "utf-8");
  return filePath;
}

function slugify(text: string): string {
  return text
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();
}

function truncate(text: string, length: number): string {
  if (text.length <= length) return text;
  return `${text.slice(0, length)}…`;
}

main().catch((error) => {
  console.error(
    "致命的なエラーが発生しました:",
    error instanceof Error ? error.message : error
  );
  process.exit(1);
});

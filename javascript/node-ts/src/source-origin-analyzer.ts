import { parse as parseHtml } from "node-html-parser";
import { writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import OpenAI from "openai";
import type {
  Response,
  WebSearchTool,
} from "openai/resources/responses/responses";

type ClassificationLabel = "primary" | "secondary" | "tertiary" | "other";

interface CliOptions {
  url: string;
  outputPath?: string;
  maxChars: number;
  debug: boolean;
  asJson: boolean;
  userLocation?: WebSearchTool["user_location"];
}

interface FetchResult {
  requestedUrl: string;
  finalUrl: string;
  status: number;
  contentType?: string;
  body: string;
  fetchedAt: string;
}

interface ExtractedText {
  title?: string;
  text: string;
  truncated: boolean;
  originalLength: number;
}

interface FollowUpSource {
  classification: Exclude<ClassificationLabel, "tertiary" | "other">;
  title: string;
  url: string;
  summary: string;
  whyTrusted: string;
  relationToOriginal: string;
}

interface ModelPayload {
  classification: ClassificationLabel;
  reasoning: string;
  summary: string;
  keyEntities: string[];
  followUpSources: FollowUpSource[];
  pendingNeeds: string;
}

interface FinalResult extends ModelPayload {
  url: string;
  resolvedUrl: string;
  fetchedAt: string;
  httpStatus: number;
  contentType?: string;
  extractedTitle?: string;
  truncated: boolean;
  extractedCharLength: number;
}

const DEFAULT_MAX_CHARS = 8000;

const structuredOutputSchema = {
  name: "source_origin_payload",
  schema: {
    type: "object",
    additionalProperties: false,
    properties: {
      classification: {
        type: "string",
        enum: ["primary", "secondary", "tertiary", "other"],
      },
      reasoning: { type: "string" },
      summary: { type: "string" },
      keyEntities: {
        type: "array",
        items: { type: "string" },
      },
      followUpSources: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            classification: {
              type: "string",
              enum: ["primary", "secondary"],
            },
            title: { type: "string" },
            url: { type: "string" },
            summary: { type: "string" },
            whyTrusted: { type: "string" },
            relationToOriginal: { type: "string" },
          },
          required: [
            "classification",
            "title",
            "url",
            "summary",
            "whyTrusted",
            "relationToOriginal",
          ],
        },
      },
      pendingNeeds: { type: "string" },
    },
    required: [
      "classification",
      "reasoning",
      "summary",
      "followUpSources",
      "keyEntities",
      "pendingNeeds",
    ],
  },
} as const;

const systemPrompt = `あなたは調査編集者です。渡されたURLの本文を読み、情報源の性質を以下の４分類のいずれかで判定してください。
- 一次情報: 事例の当事者(政府・公的機関・企業など)がおこなう公式発表やレポート。
- 二次情報: 大手報道機関、シンクタンク、コンサルティングファームなど信頼できる分析・報道。一次情報ではないが高い信頼性がある。
- 三次情報: 上記の一次/二次情報を引用・再構成した記事やブログ等。
- その他: 信頼できる引用に紐づかない個人の感想や根拠不明な内容。

必須事項:
1. 本文、書き手、引用のされ方を精査し、分類と理由を日本語で明示する。
2. 三次情報と判断した場合は、web_searchツールを使って、本文中で言及される事実の元となる一次情報か信頼できる二次情報を最低1件以上探し、followUpSourcesにまとめる。見つかったソースには、なぜ信頼できるか・元記事とどう関係するかを書き添える。
3. 一次・二次・その他と判定した場合は followUpSources を空配列[]のままにする。
4. 回答は必ずJSONスキーマに一致させ、日本語で簡潔に記述する。`;

async function main() {
  try {
    const options = parseCliOptions();
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      console.error("エラー: OPENAI_API_KEY が設定されていません");
      process.exit(1);
    }

    const client = new OpenAI({ apiKey });
    const fetchResult = await fetchPage(options.url);
    const extracted = extractReadableText(fetchResult.body, options.maxChars);

    if (!extracted.text) {
      console.error("取得テキストが空のため、解析できませんでした。");
      process.exit(1);
    }

    if (options.debug) {
      console.log("--- 取得メタデータ ---");
      console.log(`最終URL: ${fetchResult.finalUrl}`);
      console.log(`HTTPステータス: ${fetchResult.status}`);
      console.log(`Content-Type: ${fetchResult.contentType ?? "不明"}`);
      console.log(
        `抽出長: ${extracted.text.length} chars (元:${extracted.originalLength}, trunc:${extracted.truncated})`
      );
    }

    const userPrompt = buildUserPrompt(fetchResult, extracted, options.maxChars);
    const modelResult = await runModel(client, userPrompt, options.userLocation);
    const finalResult: FinalResult = {
      ...modelResult,
      url: fetchResult.requestedUrl,
      resolvedUrl: fetchResult.finalUrl,
      fetchedAt: fetchResult.fetchedAt,
      httpStatus: fetchResult.status,
      contentType: fetchResult.contentType,
      extractedTitle: extracted.title,
      truncated: extracted.truncated,
      extractedCharLength: extracted.text.length,
    };

    if (options.outputPath) {
      writeFileSync(options.outputPath, JSON.stringify(finalResult, null, 2), "utf-8");
      console.log(`📁 結果を ${options.outputPath} に保存しました`);
    }

    if (options.asJson) {
      console.log(JSON.stringify(finalResult, null, 2));
    } else {
      printHumanReadable(finalResult);
    }
  } catch (error) {
    console.error(
      `解析に失敗しました: ${error instanceof Error ? error.message : String(error)}`
    );
    process.exit(1);
  }
}

function parseCliOptions(): CliOptions {
  const { values } = parseArgs({
    options: {
      url: { type: "string" },
      output: { type: "string" },
      "max-chars": { type: "string" },
      debug: { type: "boolean" },
      json: { type: "boolean" },
      country: { type: "string" },
      region: { type: "string" },
      city: { type: "string" },
      timezone: { type: "string" },
      help: { type: "boolean", short: "h" },
    },
  });

  if (values.help) {
    printHelp();
    process.exit(0);
  }

  if (!values.url) {
    console.error("エラー: --url を指定してください");
    printHelp();
    process.exit(1);
  }

  const maxChars = values["max-chars"]
    ? Math.max(1000, Number(values["max-chars"]))
    : DEFAULT_MAX_CHARS;

  const userLocation = buildUserLocation({
    country: values.country,
    region: values.region,
    city: values.city,
    timezone: values.timezone,
  });

  return {
    url: values.url,
    outputPath: values.output,
    maxChars: Number.isFinite(maxChars) ? maxChars : DEFAULT_MAX_CHARS,
    debug: Boolean(values.debug),
    asJson: Boolean(values.json),
    userLocation,
  };
}

function printHelp(): void {
  console.log(`一次/二次/三次/その他の情報源を判定するツール

使い方:
  npm run source-origin -- --url <URL> [--output result.json] [--json] [--max-chars 8000]

オプション:
  --url <URL>          判定したいURL (必須)
  --output <path>      結果JSONを保存するパス
  --json               結果をJSONとして標準出力に表示
  --max-chars <n>      LLMに渡す本文の最大文字数 (デフォルト: ${DEFAULT_MAX_CHARS})
  --debug              フェッチメタ情報を表示
  --country <ISO>      web_search向けの推定国コード (例: JP)
  --region <text>      推定地域
  --city <text>        推定都市
  --timezone <tz>      IANAタイムゾーン
  -h, --help           このヘルプを表示
`);
}

function buildUserLocation(values: {
  country?: string;
  region?: string;
  city?: string;
  timezone?: string;
}): WebSearchTool["user_location"] | undefined {
  const hasValue = [values.country, values.region, values.city, values.timezone].some(
    (item) => item && item.trim().length > 0
  );
  if (!hasValue) {
    return undefined;
  }
  return {
    country: values.country?.toUpperCase(),
    region: values.region,
    city: values.city,
    timezone: values.timezone,
    type: "approximate",
  };
}

async function fetchPage(url: string): Promise<FetchResult> {
  const response = await fetch(url, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
      Accept:
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
      "Accept-Language": "ja,en-US;q=0.9",
    },
    redirect: "follow",
  });

  const body = await response.text();
  return {
    requestedUrl: url,
    finalUrl: response.url,
    status: response.status,
    contentType: response.headers.get("content-type") ?? undefined,
    body,
    fetchedAt: new Date().toISOString(),
  };
}

function extractReadableText(html: string, maxChars: number): ExtractedText {
  const trimmed = html.trim();
  if (!trimmed) {
    return { text: "", truncated: false, originalLength: 0 };
  }

  const root = parseHtml(trimmed, {
    blockTextElements: {
      script: true,
      noscript: true,
      style: true,
      pre: false,
    },
  });
  root.querySelectorAll("script,style,noscript,template,iframe").forEach((node) =>
    node.remove()
  );

  const title = root.querySelector("title")?.text.trim();
  const selectors = [
    "article",
    "main",
    "section",
    "div",
    "p",
    "li",
    "blockquote",
    "dd",
    "span",
  ];
  const chunks: string[] = [];
  for (const selector of selectors) {
    for (const node of root.querySelectorAll(selector)) {
      const text = normalizeWhitespace(node.text);
      if (text && text.length > 40) {
        chunks.push(text);
      }
    }
    if (chunks.length > 1200) {
      break;
    }
  }

  const combined = chunks.length
    ? dedupeSequential(chunks)
    : [normalizeWhitespace(root.text)];
  const merged = combined
    .map((item) => item.trim())
    .filter(Boolean)
    .join("\n");

  const originalLength = merged.length;
  const truncated = originalLength > maxChars;
  const text = truncated ? merged.slice(0, maxChars) : merged;

  return { title, text, truncated, originalLength };
}

function normalizeWhitespace(value: string): string {
  return value
    .replace(/\u00a0/g, " ")
    .split(/\s+/)
    .join(" ")
    .replace(/ (?=[,.:;!?])/g, "")
    .trim();
}

function dedupeSequential(values: string[]): string[] {
  const result: string[] = [];
  for (const value of values) {
    if (!result.length || result[result.length - 1] !== value) {
      result.push(value);
    }
  }
  return result;
}

function buildUserPrompt(
  fetchResult: FetchResult,
  extracted: ExtractedText,
  maxChars: number
): string {
  return `# 対象URL
${fetchResult.finalUrl}

# 取得メタデータ
- HTTPステータス: ${fetchResult.status}
- Content-Type: ${fetchResult.contentType ?? "不明"}
- 取得日時(UTC): ${fetchResult.fetchedAt}
- タイトル: ${extracted.title ?? "不明"}
- 抽出文字数: ${extracted.text.length} / 元:${extracted.originalLength}
- 切り詰め: ${extracted.truncated ? "はい" : "いいえ"} (上限 ${maxChars} chars)

# 抽出本文
"""
${extracted.text}
"""`;
}

async function runModel(
  client: OpenAI,
  prompt: string,
  userLocation?: WebSearchTool["user_location"]
): Promise<ModelPayload> {
  const response = await client.responses.create({
    model: "gpt-5-mini",
    instructions: systemPrompt,
    input: [
      {
        role: "user",
        type: "message",
        content: [
          {
            type: "input_text",
            text: prompt,
          },
        ],
      },
    ],
    tools: [
      {
        type: "web_search",
        search_context_size: "medium",
        user_location: userLocation,
      },
    ],
    text: {
      format: {
        type: "json_schema",
        name: structuredOutputSchema.name,
        schema: structuredOutputSchema.schema,
        strict: true,
      },
      verbosity: "medium",
    },
  });

  return parseModelPayload(response);
}

function parseModelPayload(response: Response): ModelPayload {
  const raw = response.output_text?.trim();
  if (!raw) {
    throw new Error("モデル出力が空でした");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`モデル出力のJSON解析に失敗しました: ${(error as Error).message}`);
  }

  return validateModelPayload(parsed);
}

function validateModelPayload(payload: unknown): ModelPayload {
  if (!payload || typeof payload !== "object") {
    throw new Error("モデル出力の形式が不正です");
  }
  const record = payload as Record<string, unknown>;
  const classification = record["classification"];
  if (
    classification !== "primary" &&
    classification !== "secondary" &&
    classification !== "tertiary" &&
    classification !== "other"
  ) {
    throw new Error("classification が不正です");
  }
  const reasoning = ensureString(record["reasoning"], "reasoning");
  const summary = ensureString(record["summary"], "summary");
  const pendingNeeds = ensureString(record["pendingNeeds"], "pendingNeeds");
  const keyEntitiesValue = record["keyEntities"];
  if (!Array.isArray(keyEntitiesValue)) {
    throw new Error("keyEntities が配列ではありません");
  }
  const keyEntities = keyEntitiesValue
    .map((value) => (typeof value === "string" ? value : null))
    .filter((value): value is string => Boolean(value && value.trim().length > 0));

  const followUpSourcesValue = record["followUpSources"];
  if (!Array.isArray(followUpSourcesValue)) {
    throw new Error("followUpSources が配列ではありません");
  }
  const followUpSources = followUpSourcesValue.map((value) => validateFollowUpSource(value));

  if (classification !== "tertiary" && followUpSources.length > 0) {
    throw new Error("三次情報以外では followUpSources は空である必要があります");
  }

  if (classification === "tertiary" && followUpSources.length === 0) {
    throw new Error("三次情報なのに followUpSources が空です");
  }

  return {
    classification,
    reasoning,
    summary,
    keyEntities,
    followUpSources,
    pendingNeeds,
  };
}

function validateFollowUpSource(entry: unknown): FollowUpSource {
  if (!entry || typeof entry !== "object") {
    throw new Error("followUpSources の要素が不正です");
  }
  const record = entry as Record<string, unknown>;
  const classification = record["classification"];
  if (classification !== "primary" && classification !== "secondary") {
    throw new Error("followUpSources 内の classification が不正です");
  }
  return {
    classification,
    title: ensureString(record["title"], "followUpSources.title"),
    url: ensureString(record["url"], "followUpSources.url"),
    summary: ensureString(record["summary"], "followUpSources.summary"),
    whyTrusted: ensureString(record["whyTrusted"], "followUpSources.whyTrusted"),
    relationToOriginal: ensureString(
      record["relationToOriginal"],
      "followUpSources.relationToOriginal"
    ),
  };
}

function ensureString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} が空文字です`);
  }
  return value;
}

function printHumanReadable(result: FinalResult): void {
  console.log("=== 判定結果 ===");
  console.log(`分類: ${result.classification}`);
  console.log(`理由: ${result.reasoning}`);
  console.log(`要約: ${result.summary}`);
  console.log(
    `主要な主体: ${result.keyEntities.length ? result.keyEntities.join(", ") : "該当なし"}`
  );
  console.log(`未解決事項: ${result.pendingNeeds || "なし"}`);
  if (result.followUpSources.length) {
    console.log("--- 三次情報を補完する一次/二次情報 ---");
    result.followUpSources.forEach((source, index) => {
      console.log(
        `[${index + 1}] (${source.classification}) ${source.title}\n    URL: ${source.url}\n    関係: ${source.relationToOriginal}\n    信頼理由: ${source.whyTrusted}\n    サマリー: ${source.summary}`
      );
    });
  }
}

main();

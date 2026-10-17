import { parseArgs } from "node:util";
import { tmpdir } from "node:os";
import { writeFileSync, existsSync, mkdirSync, readFileSync, lstatSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { mkdtemp, readdir } from "node:fs/promises";
import { execFile } from "node:child_process";
import { v2, protos } from "@google-cloud/speech";
import { Storage } from "@google-cloud/storage";
import { Status } from "google-gax";

interface CliOptions {
  gcsUri?: string;
  filePath?: string;
  bucket?: string;
  object?: string;
  outputGcsUri?: string;
  chunkSeconds: number;
  ffmpegPath: string;
  noChunk: boolean;
  reencode: "flac" | "wav";
  sampleRate: number;
  language: string;
  region: string;
  minSpeakers: number;
  maxSpeakers: number;
  mergedOutput?: string;
  mergedTimestamps: boolean;
}

/**
 * CLI オプションをパースする
 */
function parseCliOptions(): CliOptions {
  const { values } = parseArgs({
    options: {
      "gcs-uri": { type: "string", short: "u" },
      file: { type: "string", short: "f" },
      bucket: { type: "string", short: "b" },
      object: { type: "string", short: "o" },
      "output-gcs": { type: "string" },
      "chunk-seconds": { type: "string" },
      ffmpeg: { type: "string" },
      "no-chunk": { type: "boolean" },
      reencode: { type: "string" },
      "sample-rate": { type: "string" },
      language: { type: "string", short: "l" },
      region: { type: "string", short: "r" },
      "min-speakers": { type: "string" },
      "max-speakers": { type: "string" },
      "merged-output": { type: "string" },
      timestamps: { type: "boolean", short: "t" },
      help: { type: "boolean", short: "h" },
    },
  });

  if (values.help) {
    printHelp();
    process.exit(0);
  }

  const gcsUri = values["gcs-uri"] as string | undefined;
  const filePath = values.file as string | undefined;
  const bucket = values.bucket as string | undefined;
  const object = values.object as string | undefined;
  const outputGcsUri = values["output-gcs"] as string | undefined;
  const chunkSeconds = values["chunk-seconds"]
    ? Number(values["chunk-seconds"])
    : 3300;
  const ffmpegPath = (values.ffmpeg as string) || "ffmpeg";
  const noChunk = Boolean(values["no-chunk"]);
  const reencodeRaw = (values.reencode as string | undefined)?.toLowerCase();
  const reencode: "flac" | "wav" =
    reencodeRaw === "wav" ? "wav" : "flac";
  const sampleRate = values["sample-rate"]
    ? Number(values["sample-rate"])
    : 16000;

  if (!gcsUri && !filePath) {
    console.error("エラー: --gcs-uri または --file のいずれかを指定してください");
    printHelp();
    process.exit(1);
  }

  if (gcsUri && filePath) {
    console.error("エラー: --gcs-uri と --file は同時指定できません。どちらか一方を指定してください");
    process.exit(1);
  }

  if (filePath && !bucket) {
    console.error("エラー: ローカルファイルを指定する場合は --bucket も必須です");
    process.exit(1);
  }

  const minSpeakers = values["min-speakers"]
    ? Number(values["min-speakers"])
    : 2;
  const maxSpeakers = values["max-speakers"]
    ? Number(values["max-speakers"])
    : 6;
  const mergedOutputArg = values["merged-output"] as string | undefined;
  // デフォルトで元ファイル名.txt を出力
  const mergedOutput = mergedOutputArg ?? (filePath ? `${filePath.replace(/\.[^.]+$/, "")}.txt` : undefined);
  const mergedTimestamps = values.timestamps !== false;

  if (Number.isNaN(minSpeakers) || Number.isNaN(maxSpeakers)) {
    console.error("エラー: --min-speakers と --max-speakers は数値で指定してください");
    process.exit(1);
  }
  if (minSpeakers < 1 || maxSpeakers < minSpeakers) {
    console.error("エラー: 話者数の指定が不正です (1以上かつ min <= max)");
    process.exit(1);
  }

  return {
    gcsUri,
    filePath,
    bucket,
    object,
    outputGcsUri,
    chunkSeconds,
    ffmpegPath,
    noChunk,
    reencode,
    sampleRate,
    language: (values.language as string) || "ja-JP",
    region: (values.region as string) || "asia-northeast1",
    minSpeakers,
    maxSpeakers,
    mergedOutput,
    mergedTimestamps,
  };
}

function printHelp(): void {
  console.log(`
Google Speech-to-Text (chirp_3) BatchRecognize with Diarization

使い方:
  npm run stt:batch -- --gcs-uri gs://bucket/audio.flac [オプション]

必須:
  --gcs-uri, -u        Cloud Storage 上の音声ファイル URI
  --file, -f           ローカル音声ファイルのパス（指定時は --bucket 必須）
  --bucket, -b         アップロード先の GCS バケット名 (--file と併用)
  --object, -o         アップロード時のオブジェクト名 (省略時は自動生成)
  --output-gcs         結果を書き出す GCS URI (未指定かつ --bucket 使用時は自動生成)
  --chunk-seconds      ローカル音声を秒数で分割して処理 (例: 3300)。0 の場合は分割しない
  --ffmpeg             ffmpeg のパス (既定: ffmpeg)
  --no-chunk           分割を強制的に無効化
  --reencode           flac | wav で再エンコード (既定: flac) ※非対応コーデック対策
  --sample-rate        再エンコード時のサンプルレート (既定: 16000)

オプション:
  --language, -l       言語コード (既定: ja-JP)
  --region, -r         リージョン (既定: asia-northeast1)
  --min-speakers       話者数の下限 (既定: 2)
  --max-speakers       話者数の上限 (既定: 6)
  --chunk-seconds      分割秒数 (既定: 3300)。0 なら分割しない
  --merged-output      マージ済みテキストを書き出すローカルパス (既定: 元ファイル名.txt)
  --timestamps, -t     マージ結果に mm:ss 形式の区間を付与 (既定: true)
  --help, -h           このヘルプを表示

環境変数:
  GOOGLE_CLOUD_PROJECT           プロジェクト ID (必須)
  GOOGLE_APPLICATION_CREDENTIALS 認証 JSON へのパス (推奨)
`);
}

/**
 * Duration を秒(小数)に変換
 */
function durationToSeconds(
  duration?: protos.google.protobuf.IDuration | null
): number {
  const seconds = Number(duration?.seconds ?? 0);
  const nanos = Number(duration?.nanos ?? 0);
  return seconds + nanos / 1_000_000_000;
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m.toString().padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}

function formatElapsed(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const hours = Math.floor(totalSec / 3600);
  const minutes = Math.floor((totalSec % 3600) / 60);
  const seconds = totalSec % 60;
  const millis = ms % 1000;
  if (hours > 0) {
    return `${hours}時間${minutes}分${seconds}秒`;
  } else if (minutes > 0) {
    return `${minutes}分${seconds}秒`;
  } else {
    return `${seconds}.${millis.toString().padStart(3, "0")}秒`;
  }
}

export async function main(): Promise<void> {
  const mainStartTime = performance.now();
  const options = parseCliOptions();
  assertSafeMergedOutput(options.filePath, options.mergedOutput);
  if (options.outputGcsUri) validateOutputGcsUri(options.outputGcsUri);

  if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
    setupCredentialsFromBase64();
  }

  const projectId = resolveProjectId();

  if (!projectId) {
    console.error(
      "エラー: GOOGLE_CLOUD_PROJECT (または GCLOUD_PROJECT) が未設定です。環境変数をセットするか、認証 JSON の project_id を確認してください。"
    );
    process.exit(1);
  }

  const decodingConfig: protos.google.cloud.speech.v2.IExplicitDecodingConfig =
    options.reencode === "wav"
      ? {
          encoding: "LINEAR16",
          sampleRateHertz: options.sampleRate,
          audioChannelCount: 1,
        }
      : {
          encoding: "FLAC",
          sampleRateHertz: options.sampleRate,
          audioChannelCount: 1,
        };

  const config: protos.google.cloud.speech.v2.IRecognitionConfig = {
    explicitDecodingConfig: decodingConfig,
    languageCodes: [options.language],
    model: "chirp_3",
    features: {
      enableAutomaticPunctuation: true,
      diarizationConfig: {
        minSpeakerCount: options.minSpeakers,
        maxSpeakerCount: options.maxSpeakers,
      },
    },
  };

  // stt:merge 相当のマージ結果をここで蓄積する
  const mergedLines: string[] = [];
  type Segment = { speaker: string; start?: number; end?: number; text: string };
  let mergedFound = false;
  const joinWord = (base: string, word: string): string => {
    if (!base) return word;
    const prev = base[base.length - 1];
    const curr = word[0];
    const needsSpace =
      /[A-Za-z0-9]/.test(prev) && /[A-Za-z0-9]/.test(curr);
    return needsSpace ? `${base} ${word}` : `${base}${word}`;
  };
  const appendMerged = (alt?: protos.google.cloud.speech.v2.ISpeechRecognitionAlternative | null) => {
    if (!alt) return;
    const segments: Segment[] = [];

    for (const w of alt.words ?? []) {
      const speaker = w.speakerLabel ?? "S?";
      const start = w.startOffset ? durationToSeconds(w.startOffset) : undefined;
      const end = w.endOffset ? durationToSeconds(w.endOffset) : undefined;
      const text = w.word ?? "";

      const last = segments[segments.length - 1];
      if (last && last.speaker === speaker) {
        last.text = joinWord(last.text, text);
        if (start !== undefined && last.start === undefined) last.start = start;
        if (end !== undefined) last.end = end;
      } else {
        segments.push({ speaker, start, end, text });
      }
    }

    if (segments.length === 0 && alt.transcript) {
      mergedLines.push(alt.transcript);
      mergedFound = true;
      return;
    }

    for (const seg of segments) {
      if (
        options.mergedTimestamps &&
        seg.start !== undefined &&
        seg.end !== undefined
      ) {
        mergedLines.push(
          `[${seg.speaker} ${formatTime(seg.start)}-${formatTime(seg.end)}] ${seg.text}`
        );
      } else {
        mergedLines.push(`[${seg.speaker}] ${seg.text}`);
      }
      mergedFound = true;
    }
  };

  // 入力ソースを決定（GCS URI かローカルアップロード）
  const sourceUri = options.gcsUri
    ? options.gcsUri
    : await (async () => {
        const uris = await prepareLocalSources({
          filePath: options.filePath!,
          bucket: options.bucket!,
          object: options.object,
          chunkSeconds: options.noChunk ? 0 : options.chunkSeconds,
          ffmpegPath: options.ffmpegPath,
          reencode: options.reencode,
          sampleRate: options.sampleRate,
        });
        return uris;
      })();

  const client = createSpeechClient(options.region);

  // リージョンごとに Recognizer を用意（存在しなければ自動作成）
  const recognizerName = await ensureRecognizer({
    client,
    projectId,
    region: options.region,
    model: config.model ?? "chirp_3",
    languageCodes: config.languageCodes ?? ["ja-JP"],
    defaultConfig: config,
  });

  const sourceUris = Array.isArray(sourceUri) ? sourceUri : [sourceUri];
  const primarySourceUri = sourceUris[0];
  const { outputConfig, outputPrefix } = buildOutputConfig(
    options,
    primarySourceUri
  );

  for (const [idx, uri] of sourceUris.entries()) {
    const chunkStartTime = performance.now();
    const request: protos.google.cloud.speech.v2.IBatchRecognizeRequest = {
      recognizer: recognizerName,
      files: [{ uri }],
      config,
      recognitionOutputConfig: outputConfig,
    };

    console.log("\n🎧 BatchRecognize を開始します...");
    console.log(`チャンク    : ${idx + 1}/${sourceUris.length}`);
    console.log(`モデル      : ${config.model}`);
    console.log(`言語        : ${options.language}`);
    console.log(`リージョン  : ${options.region}`);
    console.log(
      `話者数範囲  : ${options.minSpeakers} - ${options.maxSpeakers}`
    );
    console.log(`入力        : ${uri}`);
    if (outputConfig.gcsOutputConfig?.uri) {
      console.log(`出力        : ${outputConfig.gcsOutputConfig.uri}`);
    } else {
      console.log("出力        : inline (応答に含める)");
    }

    const [operation] = await client.batchRecognize(request);
    const [response, opMetaRaw] = await operation.promise();

    // ジョブメタデータを簡易表示
    const opMeta = opMetaRaw as
      | protos.google.cloud.speech.v2.IOperationMetadata
      | undefined;
    if (opMeta?.progressPercent !== undefined) {
      console.log(`進捗        : ${opMeta.progressPercent}%`);
    }

    const fileResult = response.results?.[uri];
    if (fileResult?.error?.code || fileResult?.error?.message) {
      throw new Error(`チャンク ${idx + 1} の認識に失敗しました: ${fileResult.error.message ?? fileResult.error.code}`);
    }
    const transcriptResults = await readTranscriptResults(fileResult);
    if (!transcriptResults.some((result) => hasTranscript(result.alternatives?.[0]))) {
      throw new Error(`チャンク ${idx + 1} に保存可能な文字起こし結果がありません。GCS データは保持します。`);
    }

    // チャンク区切りマーカーを挿入（2つ目以降のチャンク）
    if (idx > 0 && transcriptResults.length > 0) {
      mergedLines.push("");
      mergedLines.push(`--- チャンク ${idx + 1} 開始 ---`);
      mergedLines.push("");
    }

    for (const [index, result] of transcriptResults.entries()) {
      const alt = result.alternatives?.[0];
      if (!alt) continue;

      console.log(`\n=== チャンク${idx + 1} セグメント ${index + 1} ===`);
      console.log(`本文: ${alt.transcript}`);

      alt.words?.forEach((word) => {
        const start = durationToSeconds(word.startOffset);
        const end = durationToSeconds(word.endOffset);
        // v2 は speakerLabel を返す。例: "spk_0"
        const speaker = word.speakerLabel ?? "S?";
        console.log(
          `${speaker} [${start.toFixed(2)}s - ${end.toFixed(2)}s]: ${word.word}`
        );
      });

      appendMerged(alt);
    }

    const chunkElapsed = performance.now() - chunkStartTime;
    console.log(`\n⏱️  チャンク ${idx + 1} 処理時間: ${formatElapsed(chunkElapsed)}`);
  }

  if (mergedLines.length > 0) {
    const mergedText = mergedLines.join("\n");
    console.log("\n=== マージ済みテキスト (stt:batch) ===");
    console.log(mergedText);
    if (options.mergedOutput) {
      writeFileSync(options.mergedOutput, mergedText, { encoding: "utf8", flag: "wx" });
      console.log(`書き出し: ${options.mergedOutput}`);
    }
  } else if (options.mergedOutput) {
    console.warn("マージ可能な結果がありませんでした。");
  }

  if (!mergedFound) {
    throw new Error("保存可能な文字起こし結果がありません。GCS データは保持します。");
  }
  if (outputPrefix) {
    console.log(`GCS 入力・出力を保持しました。不要になったデータは手動で削除してください: gs://${outputPrefix.bucket}/${outputPrefix.prefix}`);
  }

  const totalElapsed = performance.now() - mainStartTime;
  console.log(`\n✅ 総処理時間: ${formatElapsed(totalElapsed)}`);
}

if (require.main === module) main().catch((error) => {
  console.error(
    `❌ エラーが発生しました: ${
      error instanceof Error ? error.message : String(error)
    }`
  );
  process.exit(1);
});

/**
 * GOOGLE_APPLICATION_CREDENTIALS_BASE64 が与えられた場合に
 * 一時ファイルへ展開し、GOOGLE_APPLICATION_CREDENTIALS をセットする。
 * 既に GOOGLE_APPLICATION_CREDENTIALS がある場合は何もしない。
 */
function setupCredentialsFromBase64(): void {
  const b64 = process.env.GOOGLE_APPLICATION_CREDENTIALS_BASE64;
  if (!b64) {
    console.warn(
      "警告: GOOGLE_APPLICATION_CREDENTIALS も GOOGLE_APPLICATION_CREDENTIALS_BASE64 も未設定です。ADC を用意してください。"
    );
    return;
  }

  // 内容ハッシュで安定したファイル名を生成し、再実行時に使い回す
  const hash = createHash("sha256").update(b64).digest("hex").slice(0, 16);
  const credDir = path.join(tmpdir(), "gcp-creds");
  if (!existsSync(credDir)) mkdirSync(credDir, { recursive: true });
  const credPath = path.join(credDir, `gcp-${hash}.json`);

  if (!existsSync(credPath)) {
    try {
      const json = Buffer.from(b64, "base64").toString("utf8");
      writeFileSync(credPath, json, { mode: 0o600 });
      console.log(`一時認証ファイルを作成しました: ${credPath}`);
    } catch (error) {
      console.error(
        `エラー: 認証ファイルの展開に失敗しました: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
      process.exit(1);
    }
  }

  process.env.GOOGLE_APPLICATION_CREDENTIALS = credPath;
}

/**
 * ローカルファイルが指定された場合に GCS へアップロードし、gs:// URI を返す
 */
async function uploadIfNeeded(params: {
  filePath: string;
  bucket: string;
  object?: string;
}): Promise<string> {
  const storage = new Storage();
  const objectName =
    params.object ?? `${Date.now()}-${path.basename(params.filePath)}`;

  console.log(
    `📤 ローカルファイルをアップロード中: ${params.filePath} -> gs://${params.bucket}/${objectName}`
  );

  try {
    await storage.bucket(params.bucket).upload(params.filePath, {
      destination: objectName,
      resumable: true,
      preconditionOpts: { ifGenerationMatch: 0 },
    });
  } catch (error) {
    console.error(
      `エラー: GCS へのアップロードに失敗しました: ${
        error instanceof Error ? error.message : String(error)
      }`
    );
    throw error;
  }

  return `gs://${params.bucket}/${objectName}`;
}

/**
 * 指定リージョンに Recognizer が無ければ作成し、名前を返す
 */
async function ensureRecognizer(params: {
  client: v2.SpeechClient;
  projectId: string;
  region: string;
  model: string;
  languageCodes: string[];
  defaultConfig: protos.google.cloud.speech.v2.IRecognitionConfig;
}): Promise<string> {
  const { client, projectId, region, model, languageCodes, defaultConfig } =
    params;
  const recognizerId = "chirp-auto";
  const name = `projects/${projectId}/locations/${region}/recognizers/${recognizerId}`;

  // 既存チェック
  try {
    await client.getRecognizer({ name });
    return name;
  } catch (error) {
    const gaxCode = (error as { code?: number }).code;
    if (gaxCode !== Status.NOT_FOUND && gaxCode !== 5) {
      console.warn(
        `Recognizer の取得に失敗しましたが再作成を試みます: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }

  console.log(
    `🔧 Recognizer を作成します: ${name} (model: ${model}, languages: ${languageCodes.join(
      ","
    )})`
  );

  const [operation] = await client.createRecognizer({
    parent: `projects/${projectId}/locations/${region}`,
    recognizerId,
    recognizer: {
      model,
      languageCodes,
      defaultRecognitionConfig: defaultConfig,
    },
  });

  await operation.promise();
  return name;
}

/**
 * プロジェクト ID を解決する
 * 1) 環境変数 GOOGLE_CLOUD_PROJECT / GCLOUD_PROJECT
 * 2) GOOGLE_APPLICATION_CREDENTIALS_BASE64 の JSON 内 project_id
 * 3) GOOGLE_APPLICATION_CREDENTIALS の JSON 内 project_id
 */
function resolveProjectId(): string | null {
  const envProject =
    process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT;
  if (envProject) return envProject;

  const b64 = process.env.GOOGLE_APPLICATION_CREDENTIALS_BASE64;
  if (b64) {
    try {
      const json = JSON.parse(Buffer.from(b64, "base64").toString("utf8"));
      if (typeof json.project_id === "string") return json.project_id;
    } catch {
      /* noop */
    }
  }

  const credPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
  if (credPath && existsSync(credPath)) {
    try {
      const json = JSON.parse(readFileSync(credPath, "utf8"));
      if (typeof json.project_id === "string") return json.project_id;
    } catch {
      /* noop */
    }
  }

  return null;
}

/**
 * リージョンに応じた SpeechClient を生成
 * 非 global の場合は regional endpoint を使用する
 */
function createSpeechClient(region: string): v2.SpeechClient {
  if (region && region !== "global") {
    return new v2.SpeechClient({
      apiEndpoint: `${region}-speech.googleapis.com`,
    });
  }
  return new v2.SpeechClient();
}

/**
 * 出力先設定を組み立てる
 * - ユーザー指定 --output-gcs があればそれを使用
 * - --file と --bucket がある場合は同じバケット配下に自動生成
 * - 上記以外は inline
 */
function buildOutputConfig(
  options: CliOptions,
  sourceUri: string
): {
  outputConfig: protos.google.cloud.speech.v2.IRecognitionOutputConfig;
  outputPrefix?: { bucket: string; prefix: string };
} {
  if (options.outputGcsUri) {
    const { bucket, prefix } = parseGcsUri(ensureTrailingSlash(options.outputGcsUri));
    return {
      outputConfig: { gcsOutputConfig: { uri: `gs://${bucket}/${prefix}` } },
      outputPrefix: { bucket, prefix },
    };
  }

  if (options.bucket) {
    const baseName = path.basename(sourceUri).replace(/\W+/g, "-");
    const prefix = `stt-output/${Date.now()}-${baseName}/`;
    return {
      outputConfig: { gcsOutputConfig: { uri: `gs://${options.bucket}/${prefix}` } },
      outputPrefix: { bucket: options.bucket, prefix },
    };
  }

  return { outputConfig: { inlineResponseConfig: {} } };
}

/**
 * GCS URI を bucket と prefix に分解
 */
function parseGcsUri(uri: string): { bucket: string; prefix: string } {
  const m = uri.match(/^gs:\/\/([^/]+)\/?(.*)$/);
  if (!m) {
    throw new Error(`無効な GCS URI です: ${uri}`);
  }
  return { bucket: m[1], prefix: m[2] ?? "" };
}

function ensureTrailingSlash(uri: string): string {
  return uri.endsWith("/") ? uri : `${uri}/`;
}

/** Refuse existing destinations before any upload/API work; wx also closes the write race. */
export function assertSafeMergedOutput(inputPath?: string, outputPath?: string): void {
  if (!outputPath) return;
  if (inputPath && path.resolve(inputPath) === path.resolve(outputPath)) {
    throw new Error("入力ファイルと --merged-output に同じパスは指定できません。");
  }
  try {
    lstatSync(outputPath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  throw new Error(`出力先は既に存在します。別の --merged-output を指定してください: ${outputPath}`);
}

export function validateOutputGcsUri(uri: string): void {
  const { prefix } = parseGcsUri(uri);
  if (!prefix.replace(/\//g, "").trim()) {
    throw new Error("--output-gcs はバケット直下ではなく専用のプレフィックスを指定してください。");
  }
}

function hasTranscript(alt?: protos.google.cloud.speech.v2.ISpeechRecognitionAlternative | null): boolean {
  return Boolean(alt?.transcript?.trim() || alt?.words?.some((word) => word.word?.trim()));
}

/** Read only the object identified by this file's API result, never an entire user prefix. */
export async function readTranscriptResults(
  fileResult?: protos.google.cloud.speech.v2.IBatchRecognizeFileResult,
): Promise<protos.google.cloud.speech.v2.ISpeechRecognitionResult[]> {
  const inline = fileResult?.inlineResult?.transcript?.results ?? fileResult?.transcript?.results;
  if (inline?.length) return inline;
  const resultUri = fileResult?.cloudStorageResult?.uri || fileResult?.uri;
  if (!resultUri) throw new Error("認識結果の JSON URI がありません。GCS データは保持します。");
  const { bucket, prefix: objectName } = parseGcsUri(resultUri);
  if (!objectName || objectName.endsWith("/")) throw new Error("認識結果の URI は JSON オブジェクトを指定する必要があります。");
  const [buffer] = await new Storage().bucket(bucket).file(objectName).download();
  const json: unknown = JSON.parse(buffer.toString("utf8"));
  if (!json || typeof json !== "object") throw new Error("認識結果 JSON の形式が不正です。");
  const record = json as { transcript?: { results?: unknown }; results?: unknown };
  const results = record.transcript?.results ?? record.results;
  if (!Array.isArray(results) || !results.some((result) => hasTranscript(result?.alternatives?.[0]))) {
    throw new Error("GCS の JSON に保存可能な文字起こし結果がありません。データは保持します。");
  }
  return results;
}

/**
 * ローカルファイルを必要に応じて分割し、GCS URI の配列を返す
 */
async function prepareLocalSources(params: {
  filePath: string;
  bucket: string;
  object?: string;
  chunkSeconds: number;
  ffmpegPath: string;
  reencode: "flac" | "wav";
  sampleRate: number;
}): Promise<string[]> {
  if (params.chunkSeconds > 0) {
    const uris = await splitAndUpload(params);
    if (uris.length > 0) return uris;
    console.warn("分割に失敗したため、元ファイルをそのままアップロードします。");
  }

  // 単一ファイルでも reencode をかける
  const maybeTranscoded = await transcodeSingle(params);

  const singleUri = await uploadIfNeeded({
    filePath: maybeTranscoded ?? params.filePath,
    bucket: params.bucket,
    object: params.object,
  });
  return [singleUri];
}

async function splitAndUpload(params: {
  filePath: string;
  bucket: string;
  object?: string;
  chunkSeconds: number;
  ffmpegPath: string;
  reencode: "flac" | "wav";
  sampleRate: number;
}): Promise<string[]> {
  if (!params.chunkSeconds || params.chunkSeconds <= 0) return [];

  const tmpDir = await mkdtemp(path.join(tmpdir(), "stt-chunks-"));
  const baseExt = params.reencode === "wav" ? ".wav" : ".flac";
  const pattern = path.join(tmpDir, `chunk-%03d${baseExt}`);

  console.log(
    `🔪 ffmpeg で分割します: ${params.chunkSeconds}s ごと -> ${pattern}`
  );

  try {
    await execFileAsync(params.ffmpegPath, [
      "-i",
      params.filePath,
      "-f",
      "segment",
      "-segment_time",
      String(params.chunkSeconds),
      "-ac",
      "1",
      "-ar",
      String(params.sampleRate),
      "-acodec",
      params.reencode === "wav" ? "pcm_s16le" : "flac",
      pattern,
    ]);
  } catch (error) {
    console.error(
      `ffmpeg 分割に失敗しました: ${
        error instanceof Error ? error.message : String(error)
      }`
    );
    return [];
  }

  let files: string[] = [];
  try {
    files = (await readdir(tmpDir))
      .filter((f) => f.startsWith("chunk-"))
      .map((f) => path.join(tmpDir, f))
      .sort();
  } catch {
    /* noop */
  }

  if (files.length === 0) {
    console.warn("ffmpeg 分割後のファイルが見つかりませんでした。");
    return [];
  }

  const uris: string[] = [];
  for (const file of files) {
    const uri = await uploadIfNeeded({
      filePath: file,
      bucket: params.bucket,
      object: params.object
        ? `${params.object}.${path.basename(file)}`
        : undefined,
    });
    uris.push(uri);
  }

  return uris;
}

async function transcodeSingle(params: {
  filePath: string;
  bucket: string;
  object?: string;
  chunkSeconds: number;
  ffmpegPath: string;
  reencode: "flac" | "wav";
  sampleRate: number;
}): Promise<string | null> {
  const tmpDir = await mkdtemp(path.join(tmpdir(), "stt-reenc-"));
  const outPath = path.join(
    tmpDir,
    `reenc${params.reencode === "wav" ? ".wav" : ".flac"}`
  );

  try {
    await execFileAsync(params.ffmpegPath, [
      "-i",
      params.filePath,
      "-ac",
      "1",
      "-ar",
      String(params.sampleRate),
      "-acodec",
      params.reencode === "wav" ? "pcm_s16le" : "flac",
      outPath,
    ]);
    return outPath;
  } catch (error) {
    console.error(
      `ffmpeg 再エンコードに失敗しました: ${
        error instanceof Error ? error.message : String(error)
      }`
    );
    return null;
  }
}

function execFileAsync(cmd: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = execFile(cmd, args, (error, _stdout, stderr) => {
      if (error) {
        reject(
          new Error(
            `${error.message}${stderr ? ` | stderr: ${stderr.trim()}` : ""}`
          )
        );
        return;
      }
      resolve();
    });
    child.stderr?.on("data", () => {
      /* suppress verbose ffmpeg logs */
    });
  });
}

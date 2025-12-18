import { parseArgs } from "node:util";
import { writeFileSync, statSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { Storage } from "@google-cloud/storage";
import { protos } from "@google-cloud/speech";

interface CliOptions {
  prefix: string;
  output?: string;
  timestamps: boolean;
}

function parseCli(): CliOptions {
  const { values } = parseArgs({
    options: {
      prefix: { type: "string", short: "p" },
      output: { type: "string", short: "o" },
      timestamps: { type: "boolean", short: "t" },
      help: { type: "boolean", short: "h" },
    },
  });

  if (values.help) {
    printHelp();
    process.exit(0);
  }
  if (!values.prefix) {
    console.error("エラー: --prefix で GCS プレフィックスまたはローカルパスを指定してください (例: gs://bucket/path/ または ./src/stt-output...)");
    printHelp();
    process.exit(1);
  }
  return {
    prefix: values.prefix,
    output: values.output as string | undefined,
    timestamps: Boolean(values.timestamps),
  };
}

function printHelp() {
  console.log(`
音声文字起こし結果(JSON)をマージして整形表示します。

使い方:
  npm run stt:merge -- --prefix gs://bucket/stt-output/xxxx/ [--output merged.txt] [--timestamps]

オプション:
  --prefix, -p      GCS 上の出力プレフィックス (末尾 / 推奨)
  --output, -o      マージ結果を書き出すローカルファイル (省略時は標準出力のみ)
  --timestamps, -t  時刻を mm:ss 形式で表示する
  --help, -h        ヘルプ表示
`);
}

function durationToSec(d?: protos.google.protobuf.IDuration | null): number {
  const s = Number(d?.seconds ?? 0);
  const n = Number(d?.nanos ?? 0);
  return s + n / 1_000_000_000;
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m.toString().padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}

async function main() {
  const opts = parseCli();
  const gcsMode = opts.prefix.startsWith("gs://");
  const { files, mode } = gcsMode
    ? await listGcsFiles(opts.prefix)
    : listLocalFiles(opts.prefix);

  if (files.length === 0) {
    console.error(
      mode === "gcs"
        ? "指定プレフィックスに transcript JSON が見つかりません。"
        : "指定パスに transcript JSON が見つかりません。"
    );
    process.exit(1);
  }

  console.log(`検出ファイル (${mode}): ${files.length} 件`);

  const lines: string[] = [];

  for (const file of files) {
    const buf =
      mode === "gcs"
        ? await downloadGcs(file as { bucket: string; name: string })
        : readFileSync(file as string);
    const json = JSON.parse(buf.toString()) as {
      transcript?: { results?: protos.google.cloud.speech.v2.ISpeechRecognitionResult[] };
      results?: protos.google.cloud.speech.v2.ISpeechRecognitionResult[] | any;
    };

    // v2 inline: results[], GCS transcript: transcript.results
    const results =
      json.transcript?.results ??
      (Array.isArray(json.results) ? json.results : []);

    for (const result of results) {
      const alt = result.alternatives?.[0];
      if (!alt) continue;

      // 単語ごとの出力では読みにくいので、話者が同じ間は文をまとめてから出力する
      type Segment = {
        speaker: string;
        start?: number;
        end?: number;
        text: string;
      };

      const segments: Segment[] = [];

      const joinWord = (base: string, word: string): string => {
        if (!base) return word;
        const prev = base[base.length - 1];
        const curr = word[0];
        const needsSpace = /[A-Za-z0-9]/.test(prev) && /[A-Za-z0-9]/.test(curr);
        return needsSpace ? `${base} ${word}` : `${base}${word}`;
      };

      for (const w of alt.words ?? []) {
        const speaker = w.speakerLabel ?? (w as any).speakerTag ?? "S?";
        const start = w.startOffset ? durationToSec(w.startOffset) : undefined;
        const end = w.endOffset ? durationToSec(w.endOffset) : undefined;
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
        lines.push(alt.transcript);
        continue;
      }

      for (const seg of segments) {
        if (opts.timestamps && seg.start !== undefined && seg.end !== undefined) {
          const start = formatTime(seg.start);
          const end = formatTime(seg.end);
          lines.push(`[${seg.speaker} ${start}-${end}] ${seg.text}`);
        } else {
          lines.push(`[${seg.speaker}] ${seg.text}`);
        }
      }
    }
  }

  const outputText = lines.join("\n");
  if (opts.output) {
    writeFileSync(opts.output, outputText, "utf8");
    console.log(`書き出し: ${opts.output}`);
  }
  console.log("\n=== マージ結果 ===");
  console.log(outputText);
}

function parseGcsPrefix(uri: string): { bucket: string; prefix: string } {
  const m = uri.match(/^gs:\/\/([^/]+)\/?(.*)$/);
  if (!m) throw new Error(`GCS URI が不正です: ${uri}`);
  return { bucket: m[1], prefix: m[2] ?? "" };
}

async function listGcsFiles(prefixUri: string): Promise<{
  files: { bucket: string; name: string }[];
  mode: "gcs";
}> {
  const { bucket, prefix } = parseGcsPrefix(prefixUri);
  const storage = new Storage();
  const [files] = await storage.bucket(bucket).getFiles({ prefix });
  const target = files
    .map((f) => ({ bucket, name: f.name }))
    .filter((n) => n.name.endsWith(".json"))
    .sort((a, b) => a.name.localeCompare(b.name));
  return { files: target, mode: "gcs" };
}

function listLocalFiles(p: string): { files: string[]; mode: "local" } {
  const st = statSync(p);
  if (st.isFile()) {
    return { files: [p], mode: "local" };
  }
  const files = readdirSync(p)
    .filter((f) => f.endsWith(".json"))
    .map((f) => path.join(p, f))
    .sort();
  return { files, mode: "local" };
}

async function downloadGcs(f: { bucket: string; name: string }) {
  const storage = new Storage();
  const [buf] = await storage.bucket(f.bucket).file(f.name).download();
  return buf;
}

main().catch((err) => {
  console.error(
    `❌ エラー: ${err instanceof Error ? err.message : String(err)}`
  );
  process.exit(1);
});

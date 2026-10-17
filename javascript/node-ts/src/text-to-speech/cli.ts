import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import OpenAI from "openai";
import {
  textToSpeech,
  VOICE_TYPES,
  MODEL_TYPES,
  AUDIO_FORMATS,
  type VoiceType,
  type ModelType,
  type AudioFormat,
} from "./index.js";

interface CliOptions {
  text?: string;
  inputFile?: string;
  output: string;
  voice: VoiceType;
  model: ModelType;
  format: AudioFormat;
  speed: number;
}

/**
 * CLIオプションをパースする
 */
export function parseCliOptions(): CliOptions {
  const { values } = parseArgs({
    options: {
      text: { type: "string", short: "t" },
      input: { type: "string", short: "i" },
      output: { type: "string", short: "o" },
      voice: { type: "string", short: "v" },
      model: { type: "string", short: "m" },
      format: { type: "string", short: "f" },
      speed: { type: "string", short: "s" },
      help: { type: "boolean", short: "h" },
    },
  });

  if (values.help) {
    printHelp();
    process.exit(0);
  }

  // text か input のいずれかは必須
  if (!values.text && !values.input) {
    console.error("エラー: --text または --input を指定してください");
    printHelp();
    process.exit(1);
  }

  // output は必須
  if (!values.output) {
    console.error("エラー: --output を指定してください");
    printHelp();
    process.exit(1);
  }

  const voice = parseChoice(values.voice, "nova", VOICE_TYPES, "voice");
  const model = parseChoice(values.model, "tts-1", MODEL_TYPES, "model");
  const format = parseChoice(values.format, "mp3", AUDIO_FORMATS, "format");
  const speed = values.speed === undefined ? 1.0 : Number(values.speed);

  if (!Number.isFinite(speed) || speed < 0.25 || speed > 4.0) {
    throw new Error("--speed は 0.25 から 4.0 の有限な数値で指定してください");
  }

  return {
    text: values.text,
    inputFile: values.input,
    output: values.output,
    voice,
    model,
    format,
    speed,
  };
}

function parseChoice<T extends string>(
  value: string | undefined, fallback: T, choices: readonly T[], option: string,
): T {
  const choice = value ?? fallback;
  if (!choices.includes(choice as T)) {
    throw new Error(`--${option} は ${choices.join(", ")} のいずれかで指定してください`);
  }
  return choice as T;
}

/**
 * ヘルプメッセージを表示する
 */
function printHelp(): void {
  console.log(`
OpenAI Text-to-Speech CLI ツール

使い方:
  npm run tts -- [オプション]

オプション:
  -t, --text <text>       音声に変換するテキスト
  -i, --input <file>      テキストファイルのパス
  -o, --output <file>     出力音声ファイルのパス (必須)
  -v, --voice <voice>     音声の種類 (デフォルト: nova)
                          選択肢: alloy, echo, fable, onyx, nova, shimmer
  -m, --model <model>     使用するモデル (デフォルト: tts-1)
                          選択肢: tts-1, tts-1-hd
  -f, --format <format>   出力フォーマット (デフォルト: mp3)
                          選択肢: mp3, opus, aac, flac, wav, pcm
  -s, --speed <speed>     再生速度 (デフォルト: 1.0, 範囲: 0.25-4.0)
  -h, --help              このヘルプを表示

環境変数:
  OPENAI_API_KEY          OpenAI API キー (必須)

例:
  # テキストを直接指定
  npm run tts -- -t "こんにちは、世界" -o output.mp3

  # ファイルから読み込み
  npm run tts -- -i input.txt -o output.mp3

  # 高品質モデルとカスタム音声を使用
  npm run tts -- -t "Hello, World" -o output.mp3 -m tts-1-hd -v alloy

  # 速度を変更
  npm run tts -- -t "こんにちは" -o output.mp3 -s 1.5
  `);
}

/**
 * メイン処理
 */
export async function main(): Promise<void> {
  try {
    const options = parseCliOptions();

    // OpenAI API キーの確認
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      console.error("エラー: OPENAI_API_KEY 環境変数が設定されていません");
      process.exit(1);
    }

    // テキストの取得
    let text: string;
    if (options.text) {
      text = options.text;
    } else if (options.inputFile) {
      try {
        text = readFileSync(options.inputFile, "utf-8");
      } catch (error) {
        console.error(
          `エラー: 入力ファイルの読み込みに失敗しました: ${
            error instanceof Error ? error.message : "不明なエラー"
          }`
        );
        process.exit(1);
      }
    } else {
      console.error("エラー: テキストが指定されていません");
      process.exit(1);
    }

    console.log("🎙️  音声を生成中...");
    console.log(`テキスト: ${text.slice(0, 100)}${text.length > 100 ? "..." : ""}`);
    console.log(`音声: ${options.voice}`);
    console.log(`モデル: ${options.model}`);
    console.log(`フォーマット: ${options.format}`);
    console.log(`速度: ${options.speed}x`);

    // OpenAI クライアントを初期化
    const client = new OpenAI({ apiKey });

    // 音声を生成
    const result = await textToSpeech(client, text, options.output, {
      voice: options.voice,
      model: options.model,
      format: options.format,
      speed: options.speed,
    });

    console.log(`✅ 音声ファイルを生成しました: ${result.outputPath}`);
    if (result.sizeBytes) {
      const sizeKB = (result.sizeBytes / 1024).toFixed(2);
      console.log(`ファイルサイズ: ${sizeKB} KB`);
    }
  } catch (error) {
    console.error(
      `❌ エラーが発生しました: ${
        error instanceof Error ? error.message : "不明なエラー"
      }`
    );
    process.exit(1);
  }
}

// エントリーポイント
if (require.main === module) main().catch((error) => {
  console.error(
    "致命的なエラーが発生しました:",
    error instanceof Error ? error.message : error
  );
  process.exit(1);
});

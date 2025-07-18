import { parse } from "std/flags/mod.ts";
import { AppConfig } from "../types/index.ts";

export interface CliArgs {
  _: string[];
  output?: string;
  api?: string;
  pattern?: string;
  concurrency?: number;
  verbose?: boolean;
  help?: boolean;
}

export function parseArgs(args: string[]): CliArgs {
  const parsed = parse(args, {
    string: ["output", "api", "pattern", "concurrency"],
    boolean: ["verbose", "help"],
    alias: {
      o: "output",
      a: "api", 
      p: "pattern",
      c: "concurrency",
      v: "verbose",
      h: "help",
    },
    default: {
      api: "vision",
      concurrency: "3",
      verbose: false,
    },
  });

  return {
    _: parsed._.map(String),
    output: parsed.output,
    api: parsed.api,
    pattern: parsed.pattern,
    concurrency: parsed.concurrency ? parseInt(parsed.concurrency, 10) : 3,
    verbose: parsed.verbose,
    help: parsed.help,
  };
}

export function validateArgs(args: CliArgs): AppConfig | null {
  if (args.help || args._.length === 0) {
    printHelp();
    return null;
  }

  const inputDir = args._[0] as string;

  if (!args.api || !["vision", "documentai"].includes(args.api)) {
    console.error(`エラー: 無効なAPIタイプ: ${args.api}`);
    console.error("使用可能なオプション: vision, documentai");
    return null;
  }

  if (args.concurrency && (args.concurrency < 1 || args.concurrency > 10)) {
    console.error(`エラー: 並行処理数は1から10の間で指定してください: ${args.concurrency}`);
    return null;
  }

  return {
    inputDir,
    outputDir: args.output,
    apiType: args.api as "vision" | "documentai",
    filePattern: args.pattern,
    concurrency: args.concurrency,
    verbose: args.verbose,
  };
}

export function printHelp(): void {
  console.log(`
PDF テキスト抽出ツール

使用方法:
  deno run --allow-read --allow-write --allow-env --allow-net pdf-extract <input-dir> [options]

引数:
  <input-dir>    PDF ファイルを含むディレクトリパス

オプション:
  -o, --output <dir>        出力ディレクトリ (デフォルト: 入力ディレクトリと同じ)
  -a, --api <type>          使用するAPI (vision | documentai) (デフォルト: vision)
  -p, --pattern <pattern>   ファイルパターン (例: "*.pdf", "invoice_*.pdf")
  -c, --concurrency <num>   並行処理数 (1-10) (デフォルト: 3)
  -v, --verbose             詳細なログ出力
  -h, --help                ヘルプを表示

環境変数:
  GOOGLE_APPLICATION_CREDENTIALS    Google Cloud サービスアカウントキーのパス
  GOOGLE_CLOUD_PROJECT             Google Cloud プロジェクトID

例:
  # Vision API を使用してすべての PDF を処理
  deno run --allow-all pdf-extract ./pdfs -o ./output

  # Document AI を使用して特定のパターンのファイルを処理
  deno run --allow-all pdf-extract ./pdfs -a documentai -p "invoice_*.pdf"

  # 詳細ログ付きで並行処理数を指定
  deno run --allow-all pdf-extract ./pdfs -v -c 5
  `);
}
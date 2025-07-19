import { parse } from "std/flags/mod.ts";
import { AppConfig } from "../types/index.ts";

export interface CliArgs {
  _: string[];
  output?: string;
  library?: string;
  pattern?: string;
  concurrency?: number;
  verbose?: boolean;
  help?: boolean;
  merge?: boolean;
  "merge-separator"?: string;
}

export function parseArgs(args: string[]): CliArgs {
  const parsed = parse(args, {
    string: ["output", "library", "pattern", "concurrency", "merge-separator"],
    boolean: ["verbose", "help", "merge"],
    alias: {
      o: "output",
      l: "library", 
      p: "pattern",
      c: "concurrency",
      v: "verbose",
      h: "help",
      m: "merge",
    },
    default: {
      library: "pdf-ts",
      concurrency: "3",
      verbose: false,
      merge: false,
      "merge-separator": "\n\n========================================\n\n",
    },
  });

  return {
    _: parsed._.map(String),
    output: parsed.output,
    library: parsed.library,
    pattern: parsed.pattern,
    concurrency: parsed.concurrency ? parseInt(parsed.concurrency, 10) : 3,
    verbose: parsed.verbose,
    help: parsed.help,
    merge: parsed.merge,
    "merge-separator": parsed["merge-separator"],
  };
}

export function validateArgs(args: CliArgs): AppConfig | null {
  if (args.help || args._.length === 0) {
    printHelp();
    return null;
  }

  const inputDir = args._[0] as string;

  if (!args.library || !["pdf-ts", "pdf-parse"].includes(args.library)) {
    console.error(`エラー: 無効なライブラリタイプ: ${args.library}`);
    console.error("使用可能なオプション: pdf-ts, pdf-parse");
    return null;
  }

  if (args.concurrency && (args.concurrency < 1 || args.concurrency > 10)) {
    console.error(`エラー: 並行処理数は1から10の間で指定してください: ${args.concurrency}`);
    return null;
  }

  return {
    inputDir,
    outputDir: args.output,
    libraryType: args.library as "pdf-ts" | "pdf-parse",
    filePattern: args.pattern,
    concurrency: args.concurrency,
    verbose: args.verbose,
    merge: args.merge,
    mergeSeparator: args["merge-separator"],
  };
}

export function printHelp(): void {
  console.log(`
PDF テキスト抽出ツール（テキスト層版）

使用方法:
  deno run --allow-read --allow-write --allow-env --allow-net pdf-text-extract <input-dir> [options]

引数:
  <input-dir>    PDF ファイルを含むディレクトリパス

オプション:
  -o, --output <dir>        出力ディレクトリ (デフォルト: 入力ディレクトリと同じ)
  -l, --library <type>      使用するライブラリ (pdf-ts | pdf-parse) (デフォルト: pdf-ts)
  -p, --pattern <pattern>   ファイルパターン (例: "*.pdf", "invoice_*.pdf")
  -c, --concurrency <num>   並行処理数 (1-10) (デフォルト: 3)
  -m, --merge               すべての抽出されたテキストを1つのファイルにマージ
  --merge-separator <text>  マージ時のファイル間セパレータ (デフォルト: 区切り線)
  -v, --verbose             詳細なログ出力
  -h, --help                ヘルプを表示

説明:
  このツールは PDF ファイル内のテキスト層を抽出します。
  OCR は使用せず、既存のテキスト情報のみを取得します。
  画像のみの PDF の場合は、テキストを抽出できません。

例:
  # pdf-ts を使用してすべての PDF を処理
  deno run --allow-all pdf-text-extract ./pdfs -o ./output

  # pdf-parse を使用して特定のパターンのファイルを処理
  deno run --allow-all pdf-text-extract ./pdfs -l pdf-parse -p "invoice_*.pdf"

  # 詳細ログ付きで並行処理数を指定
  deno run --allow-all pdf-text-extract ./pdfs -v -c 5

  # すべてのテキストを1つのファイルにマージ
  deno run --allow-all pdf-text-extract ./pdfs -m -o ./output

  # カスタムセパレータでマージ
  deno run --allow-all pdf-text-extract ./pdfs -m --merge-separator "\\n--- 次のファイル ---\\n"
  `);
}
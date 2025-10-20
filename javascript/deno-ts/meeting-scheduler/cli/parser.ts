import { CliOptions, Person } from "../types/index.ts";
import { parseDateTime } from "../utils/date-utils.ts";

export function parseCliArgs(args: string[]): CliOptions {
  const options: CliOptions = {
    startDate: new Date(),
    endDate: new Date(),
    duration: parseInt(Deno.env.get("DEFAULT_DURATION") || "60", 10),
    participants: [],
    businessHoursOnly: true,
    timezone: Deno.env.get("DEFAULT_TIMEZONE") || "Asia/Tokyo",
    outputFormat: "text",
    useOpenAI: false,
    verbose: false,
    showAll: false,
    limit: undefined,
    rawSlots: false,
    minDuration: 30, // デフォルト30分
    textNoIndex: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    switch (arg) {
      case "--start":
      case "-s":
        if (i + 1 < args.length) {
          options.startDate = parseDateTime(args[++i]);
        }
        break;

      case "--end":
      case "-e":
        if (i + 1 < args.length) {
          options.endDate = parseDateTime(args[++i]);
        }
        break;

      case "--duration":
      case "-d":
        if (i + 1 < args.length) {
          options.duration = parseInt(args[++i], 10);
        }
        break;

      case "--participant":
      case "-p":
        if (i + 1 < args.length) {
          const participantStr = args[++i];
          options.participants.push(parseParticipant(participantStr));
        }
        break;

      case "--participants-file":
        if (i + 1 < args.length) {
          const filePath = args[++i];
          options.participants.push(...parseParticipantsFile(filePath));
        }
        break;

      case "--all-day":
        options.businessHoursOnly = false;
        break;

      case "--timezone":
        if (i + 1 < args.length) {
          options.timezone = args[++i];
        }
        break;

      case "--format":
      case "-f":
        if (i + 1 < args.length) {
          const format = args[++i];
          if (format === "json" || format === "markdown" || format === "text") {
            options.outputFormat = format;
          }
        }
        break;

      case "--openai":
        options.useOpenAI = true;
        break;

      case "--verbose":
      case "-v":
        options.verbose = true;
        break;

      case "--show-all":
        options.showAll = true;
        break;

      case "--limit":
        if (i + 1 < args.length) {
          options.limit = parseInt(args[++i], 10);
          if (options.limit <= 0) {
            throw new Error("--limit は1以上の数値を指定してください");
          }
        }
        break;

      case "--raw-slots":
        options.rawSlots = true;
        break;

      case "--text-no-index":
        options.textNoIndex = true;
        break;

      case "--min-duration":
      case "-m":
        if (i + 1 < args.length) {
          options.minDuration = parseInt(args[++i], 10);
          if (options.minDuration < 0) {
            throw new Error("--min-duration は0以上の数値を指定してください");
          }
        }
        break;

      case "--help":
      case "-h":
        printHelp();
        Deno.exit(0);
    }
  }

  // デフォルト値の設定
  if (options.endDate <= options.startDate) {
    options.endDate = new Date(options.startDate.getTime() + 7 * 24 * 60 * 60 * 1000); // 1週間後
  }

  if (options.participants.length === 0) {
    throw new Error("参加者を最低1人指定してください");
  }

  return options;
}

function parseParticipant(str: string): Person {
  // フォーマット: "名前:email:source:sourceId"
  // source: "google" または "hubspot"
  const parts = str.split(":");
  
  if (parts.length < 3) {
    throw new Error(
      `無効な参加者形式: ${str}\n` +
      `期待される形式: 名前:email:source[:sourceId]\n` +
      `source: "google" または "hubspot"\n` +
      `sourceId: 省略時はemailが使用されます（Googleの場合のみ）`
    );
  }

  const source = parts[2].toLowerCase();
  if (source !== "google" && source !== "hubspot") {
    throw new Error(`無効なソース: ${source}。"google" または "hubspot" を指定してください。`);
  }

  // sourceIdが省略された場合、Googleならemailを使用、HubSpotはエラー
  let sourceId = parts[3];
  if (!sourceId) {
    if (source === "google") {
      sourceId = parts[1]; // emailをcalendarIdとして使用
    } else {
      throw new Error("HubSpotユーザーにはミーティングリンクの指定が必須です。");
    }
  }

  return {
    name: parts[0],
    email: parts[1],
    source: source as "google" | "hubspot",
    sourceId,
  };
}

function parseParticipantsFile(filePath: string): Person[] {
  try {
    const content = Deno.readTextFileSync(filePath);
    const lines = content.split("\n").filter(line => line.trim() && !line.startsWith("#"));
    
    return lines.map(line => {
      const parts = line.split(",").map(p => p.trim());
      if (parts.length < 3) {
        throw new Error(`無効な行: ${line}\nフォーマット: 名前,email,source[,sourceId]`);
      }
      
      const source = parts[2].toLowerCase();
      if (source !== "google" && source !== "hubspot") {
        throw new Error(`無効なソース: ${source}`);
      }

      let sourceId = parts[3];
      if (!sourceId) {
        if (source === "google") {
          sourceId = parts[1];
        } else {
          throw new Error(`HubSpotユーザー ${parts[0]} にはユーザーIDが必要です`);
        }
      }
      
      return {
        name: parts[0],
        email: parts[1],
        source: source as "google" | "hubspot",
        sourceId,
      };
    });
  } catch (error) {
    throw new Error(`参加者ファイルの読み込みエラー: ${error}`);
  }
}

function printHelp(): void {
  console.log(`
会議スケジュール調整ツール

使用方法:
  deno run --allow-net --allow-env --allow-read meeting-scheduler/app.ts [オプション]

オプション:
  -s, --start <日時>         検索開始日時 (例: 2024-01-15T09:00:00)
  -e, --end <日時>           検索終了日時 (例: 2024-01-22T18:00:00)
  -d, --duration <分>        会議の長さ（分単位、デフォルト: 60）
  -m, --min-duration <分>    最小会議時間（--raw-slots時に使用）
  -p, --participant <情報>   参加者情報 (形式: 名前:email:source[:sourceId])
                              Google: sourceIdはcalendarId（省略時email使用）
                              HubSpot: sourceIdはmeeting-link（必須）
  --participants-file <path> CSVファイルから参加者を読み込み
  --all-day                  営業時間外も含める（デフォルトは9-18時のみ）
  --timezone <tz>            タイムゾーン（デフォルト: Asia/Tokyo）
  -f, --format <形式>        出力形式 (text/json/markdown、デフォルト: text)
  --openai                   OpenAI APIを使用して最適化
  --show-all                 全ての空き時間候補を表示
  --limit <数>               表示する候補数の上限（デフォルト: 5）
  --raw-slots                連続した空き時間ブロックを表示（例: 10:00-15:00）
  --text-no-index            テキスト出力で先頭の番号を非表示
  -v, --verbose              詳細ログを表示
  -h, --help                 このヘルプを表示

環境変数:
  GOOGLE_CLIENT_ID          Google OAuth2 クライアントID
  GOOGLE_CLIENT_SECRET      Google OAuth2 クライアントシークレット
  GOOGLE_REFRESH_TOKEN      Google OAuth2 リフレッシュトークン
  HUBSPOT_API_KEY           HubSpot APIキー
  OPENAI_API_KEY            OpenAI APIキー（--openai使用時）

使用例:
  # Google 2人、HubSpot 1人の参加者で空き時間を検索
  ./app.ts -s 2024-01-15T09:00:00 \\
    -p "田中太郎:tanaka@example.com:google" \\
    -p "山田花子:yamada@example.com:google" \\
    -p "安達裕哉:yuyadachi@workwonders.jp:hubspot:yuyadachi"

  # CSVファイルから参加者を読み込み、JSON形式で出力
  ./app.ts --participants-file participants.csv -f json

  # OpenAI APIを使用して最適な時間を提案
  ./app.ts -p "佐藤:sato@example.com" --openai -d 90

  # 全ての空き時間を表示
  ./app.ts -p "田中:tanaka@example.com:google" --show-all

  # 上位10件のみ表示
  ./app.ts -p "山田:yamada@example.com:google" --limit 10

  # 連続した空き時間ブロックを表示
  ./app.ts -p "佐藤:sato@example.com:google" --raw-slots

  # 60分以上の空き時間のみ表示
  ./app.ts -p "鈴木:suzuki@example.com:google" --raw-slots -m 60

参加者CSVファイル形式:
  # コメント行
  # 名前,メールアドレス,ソース(google/hubspot),ソースID
  田中太郎,tanaka@example.com,google,tanaka@example.com
  山田花子,yamada@example.com,google
  安達裕哉,yuyadachi@workwonders.jp,hubspot,yuyadachi
  `);
}

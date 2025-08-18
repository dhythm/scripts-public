import { CliOptions, Person } from "../types/index.ts";
import { parseDateTime } from "../utils/date-utils.ts";

export function parseCliArgs(args: string[]): CliOptions {
  const options: CliOptions = {
    startDate: new Date(),
    endDate: new Date(),
    duration: 60,
    participants: [],
    businessHoursOnly: true,
    timezone: "Asia/Tokyo",
    outputFormat: "text",
    useOpenAI: false,
    verbose: false,
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
  // フォーマット: "名前:email:calendarId:hubspotUserId"
  const parts = str.split(":");
  
  if (parts.length < 2) {
    throw new Error(`無効な参加者形式: ${str}\n期待される形式: 名前:email[:calendarId][:hubspotUserId]`);
  }

  return {
    name: parts[0],
    email: parts[1],
    calendarId: parts[2] || parts[1], // デフォルトはemailをcalendarIdとして使用
    hubspotUserId: parts[3] || undefined,
  };
}

function parseParticipantsFile(filePath: string): Person[] {
  try {
    const content = Deno.readTextFileSync(filePath);
    const lines = content.split("\n").filter(line => line.trim() && !line.startsWith("#"));
    
    return lines.map(line => {
      const parts = line.split(",").map(p => p.trim());
      if (parts.length < 2) {
        throw new Error(`無効な行: ${line}`);
      }
      
      return {
        name: parts[0],
        email: parts[1],
        calendarId: parts[2] || parts[1],
        hubspotUserId: parts[3] || undefined,
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
  -p, --participant <情報>   参加者情報 (形式: 名前:email[:calendarId][:hubspotUserId])
  --participants-file <path> CSVファイルから参加者を読み込み
  --all-day                  営業時間外も含める（デフォルトは9-18時のみ）
  --timezone <tz>            タイムゾーン（デフォルト: Asia/Tokyo）
  -f, --format <形式>        出力形式 (text/json/markdown、デフォルト: text)
  --openai                   OpenAI APIを使用して最適化
  -v, --verbose              詳細ログを表示
  -h, --help                 このヘルプを表示

環境変数:
  GOOGLE_CLIENT_ID          Google OAuth2 クライアントID
  GOOGLE_CLIENT_SECRET      Google OAuth2 クライアントシークレット
  GOOGLE_REFRESH_TOKEN      Google OAuth2 リフレッシュトークン
  HUBSPOT_API_KEY           HubSpot APIキー
  OPENAI_API_KEY            OpenAI APIキー（--openai使用時）

使用例:
  # 2人の参加者で1週間の空き時間を検索
  ./app.ts -s 2024-01-15T09:00:00 -p "田中太郎:tanaka@example.com" -p "山田花子:yamada@example.com"

  # CSVファイルから参加者を読み込み、JSON形式で出力
  ./app.ts --participants-file participants.csv -f json

  # OpenAI APIを使用して最適な時間を提案
  ./app.ts -p "佐藤:sato@example.com" --openai -d 90

参加者CSVファイル形式:
  # コメント行
  名前,メールアドレス,GoogleカレンダーID,HubSpotユーザーID
  田中太郎,tanaka@example.com,tanaka@example.com,12345
  山田花子,yamada@example.com,,67890
  `);
}
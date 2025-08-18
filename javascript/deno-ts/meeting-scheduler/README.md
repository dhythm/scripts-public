# 会議スケジュール調整ツール

複数人の Google Calendar と HubSpot の予定を統合し、全員が空いている時間を見つけて最適な会議時間を提案するツールです。

## 機能

- 🗓️ Google Calendar API から参加者の予定を取得
- 📅 HubSpot Meeting API から参加者の予定を取得
- 🔍 異なるカレンダーシステムを使う参加者間で空き時間を検出
- 🤖 OpenAI API を使用した最適な時間の提案（オプション）
- 📊 複数の出力形式（テキスト、JSON、Markdown）

## セットアップ

### 1. 環境変数の設定

以下の環境変数を設定してください：

```bash
# Google Calendar API（OAuth2）
export GOOGLE_CLIENT_ID="your-client-id"
export GOOGLE_CLIENT_SECRET="your-client-secret"
export GOOGLE_REFRESH_TOKEN="your-refresh-token"

# HubSpot API
export HUBSPOT_API_KEY="your-hubspot-api-key"

# OpenAI API（オプション）
export OPENAI_API_KEY="your-openai-api-key"
```

### 2. Google Calendar API の設定

1. [Google Cloud Console](https://console.cloud.google.com/) にアクセス
2. 新しいプロジェクトを作成または既存のプロジェクトを選択
3. Calendar API を有効化
4. OAuth 2.0 クライアント ID を作成（リダイレクトURIに`http://localhost`を追加）
5. `.env`ファイルにクライアントIDとシークレットを設定
6. リフレッシュトークンを取得：

```bash
# リフレッシュトークン取得ツールを実行
deno run --allow-net --allow-env --allow-read --allow-write \
  tools/get-google-refresh-token.ts

# 表示される手順に従って認証
# 取得したリフレッシュトークンは自動的に.envに保存されます
```

詳細な設定手順は[Google OAuth設定ガイド](docs/google-oauth-setup.md)を参照してください。

### 3. HubSpot API の設定

1. [HubSpot開発者アカウント](https://developers.hubspot.com/) にログイン
2. アプリまたは API キーを作成
3. 必要なスコープを設定（meetings:read）

## 使用方法

### 基本的な使用例

```bash
# Googleユーザー2人、HubSpotユーザー1人で1週間の空き時間を検索
deno run --allow-net --allow-env --allow-read meeting-scheduler/app.ts \
  -s 2024-01-15T09:00:00 \
  -p "田中太郎:tanaka@example.com:google" \
  -p "山田花子:yamada@example.com:google" \
  -p "佐藤次郎:sato@example.com:hubspot:12345"
```

### 参加者ファイルを使用

参加者が多い場合は CSV ファイルから読み込めます：

`participants.csv`:
```csv
# 名前,メールアドレス,ソース(google/hubspot),ソースID
田中太郎,tanaka@example.com,google,tanaka@example.com
山田花子,yamada@example.com,google
佐藤次郎,sato@example.com,hubspot,12345
鈴木一郎,suzuki@example.com,google,suzuki@gmail.com
```

```bash
deno run --allow-net --allow-env --allow-read meeting-scheduler/app.ts \
  --participants-file participants.csv \
  -f json
```

### OpenAI による最適化

```bash
deno run --allow-net --allow-env --allow-read meeting-scheduler/app.ts \
  -p "佐藤:sato@example.com:google" \
  -p "鈴木:suzuki@example.com:hubspot:67890" \
  --openai \
  -d 90  # 90分の会議
```

### 詳細オプション

```bash
deno run --allow-net --allow-env --allow-read meeting-scheduler/app.ts \
  -s 2024-01-15T09:00:00 \  # 開始日時
  -e 2024-01-22T18:00:00 \  # 終了日時
  -d 60 \                    # 会議時間（分）
  -p "名前:email:source:sourceId" \
  --all-day \                # 営業時間外も含める
  --timezone "America/New_York" \
  -f markdown \              # 出力形式
  --openai \                 # AI最閕化
  --verbose                  # 詳細ログ
```

## コマンドラインオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `-s, --start` | 検索開始日時 | 現在時刻 |
| `-e, --end` | 検索終了日時 | 開始から7日後 |
| `-d, --duration` | 会議の長さ（分） | 60 |
| `-p, --participant` | 参加者情報 (名前:email:source[:sourceId]) | 必須 |
| `--participants-file` | 参加者CSVファイル | - |
| `--all-day` | 営業時間外も含める | false |
| `--timezone` | タイムゾーン | Asia/Tokyo |
| `-f, --format` | 出力形式 (text/json/markdown) | text |
| `--openai` | OpenAI APIで最適化 | false |
| `-v, --verbose` | 詳細ログ表示 | false |

### 環境変数（.envファイル）

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `GOOGLE_CLIENT_ID` | Google OAuth2 クライアントID | 必須（Google使用時） |
| `GOOGLE_CLIENT_SECRET` | Google OAuth2 クライアントシークレット | 必須（Google使用時） |
| `GOOGLE_REFRESH_TOKEN` | Google OAuth2 リフレッシュトークン | 必須（Google使用時） |
| `HUBSPOT_API_KEY` | HubSpot APIキー | 必須（HubSpot使用時） |
| `OPENAI_API_KEY` | OpenAI APIキー | 必須（--openai使用時） |
| `OPENAI_MODEL` | 使用するOpenAIモデル | gpt-4o-mini |
| `DEFAULT_TIMEZONE` | デフォルトのタイムゾーン | Asia/Tokyo |
| `DEFAULT_DURATION` | デフォルトの会議時間（分） | 60 |

## 出力例

### テキスト形式（デフォルト）

```
=== 会議候補時間 ===
合計 15 件の候補が見つかりました。

おすすめトップ5:

1. 2024/01/16 10:00 - 2024/01/16 11:00
   スコア: 95/100
   理由: 午前中の集中しやすい時間, 業務時間内, 平日

2. 2024/01/17 14:00 - 2024/01/17 15:00
   スコア: 85/100
   理由: 午後の生産的な時間, 業務時間内, 平日
```

### JSON形式

```json
{
  "totalCandidates": 15,
  "topCandidates": [
    {
      "start": "2024-01-16T01:00:00.000Z",
      "end": "2024-01-16T02:00:00.000Z",
      "score": 95,
      "reasons": ["午前中の集中しやすい時間", "業務時間内", "平日"]
    }
  ]
}
```

### Markdown形式

```markdown
# 会議候補時間

合計 15 件の候補が見つかりました。

## おすすめトップ5

### 1. 2024/01/16 10:00
- **時間**: 2024/01/16 10:00 - 2024/01/16 11:00
- **スコア**: 95/100
- **理由**:
  - 午前中の集中しやすい時間
  - 業務時間内
  - 平日
```

## トラブルシューティング

### 参加者の指定方法

- **source**: "google" または "hubspot" を指定
- **sourceId**: 
  - Googleの場合: カレンダーID（省略時はemailが使用される）
  - HubSpotの場合: ユーザーID（必須）

### Google Calendar API エラー

- リフレッシュトークンの有効期限切れ → 再度 OAuth フローを実行
- API が有効化されていない → Google Cloud Console で Calendar API を有効化
- スコープ不足 → `https://www.googleapis.com/auth/calendar.readonly` スコープを追加

### HubSpot API エラー

- APIキーが無効 → HubSpot 開発者ポータルで新しいキーを生成
- レート制限 → API呼び出し頻度を調整

### 空き時間が見つからない

- 期間を広げる（`--end` オプション）
- 会議時間を短くする（`--duration` オプション）
- 営業時間外も含める（`--all-day` オプション）

## ライセンス

MIT
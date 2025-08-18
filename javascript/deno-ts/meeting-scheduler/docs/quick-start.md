# クイックスタートガイド

## 🚀 5分で始める会議スケジュール調整

### Step 1: 環境準備（初回のみ）

```bash
# deno-tsディレクトリに移動
cd javascript/deno-ts

# .envファイルを作成
cp meeting-scheduler/.env.example meeting-scheduler/.env
```

### Step 2: Google Calendar設定（Googleユーザーがいる場合）

#### 2.1 Google Cloud Consoleで認証情報を作成

1. https://console.cloud.google.com/ にアクセス
2. 「APIとサービス」→「認証情報」→「認証情報を作成」→「OAuthクライアントID」
3. アプリケーションの種類: ウェブアプリケーション
4. リダイレクトURI: `http://localhost`を追加
5. 作成されたクライアントIDとシークレットをコピー

#### 2.2 .envファイルに設定

```bash
# .envファイル
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
```

#### 2.3 リフレッシュトークンを取得

```bash
# deno taskを使用（推奨）
deno task meeting:auth

# または直接実行
deno run --allow-net --allow-env --allow-read --allow-write \
  meeting-scheduler/tools/get-google-refresh-token.ts

# ブラウザでURLを開いて認証
# 認証コードをツールに入力
# リフレッシュトークンが自動的に.envに保官されます
```

### Step 3: HubSpot設定（HubSpotユーザーがいる場合）

1. https://developers.hubspot.com/ にログイン
2. アプリを作成またはAPIキーを取得
3. .envファイルに追加:

```bash
HUBSPOT_API_KEY=your-hubspot-api-key
```

### Step 4: 実行！

#### 基本的な使用例

```bash
# Googleユーザー2人の空き時間を検索
deno task meeting \
  -p "田中:tanaka@gmail.com:google" \
  -p "山田:yamada@gmail.com:google"
```

#### 混在パターン（Google + HubSpot）

```bash
# GoogleとHubSpotユーザーの混在
deno task meeting \
  -p "田中:tanaka@gmail.com:google" \
  -p "佐藤:sato@example.com:hubspot:12345"
```

#### CSVファイルから読み込み

```bash
# participants.csv を作成
cat > participants.csv << EOF
田中太郎,tanaka@gmail.com,google
山田花子,yamada@gmail.com,google
佐藤次郎,sato@example.com,hubspot,12345
EOF

# 実行
deno task meeting --participants-file meeting-scheduler/participants.csv
```

## 📊 出力形式

### テキスト（デフォルト）
```bash
deno task meeting -p "田中:tanaka@gmail.com:google"
```

### JSON形式
```bash
deno task meeting -p "田中:tanaka@gmail.com:google" -f json
```

### Markdown形式
```bash
deno task meeting -p "田中:tanaka@gmail.com:google" -f markdown
```

## 🤖 AI最適化を使う

```bash
# OpenAI APIキーを.envに設定
echo "OPENAI_API_KEY=sk-..." >> .env

# --openaiオプションを追加
deno task meeting \
  -p "田中:tanaka@gmail.com:google" \
  -p "山田:yamada@gmail.com:google" \
  --openai
```

## 🔧 オプション

| オプション | 説明 | 例 |
|----------|------|-----|
| `-s` | 開始日時 | `-s 2024-01-15T09:00:00` |
| `-e` | 終了日時 | `-e 2024-01-22T18:00:00` |
| `-d` | 会議時間（分） | `-d 90` |
| `--all-day` | 営業時間外も含む | `--all-day` |
| `--verbose` | 詳細ログ | `--verbose` |

## ❓ よくある質問

### Q: 「リフレッシュトークンが返されませんでした」エラー

A: Googleアカウント設定でアプリのアクセス権を削除してから再認証:
1. https://myaccount.google.com/permissions にアクセス
2. アプリを削除
3. もう一度認証

### Q: HubSpotのユーザーIDはどこで確認？

A: HubSpot管理画面の「設定」→「ユーザーとチーム」で確認できます。

### Q: 営業時間外も検索したい

A: `--all-day`オプションを追加してください。

## 📚 詳細ドキュメント

- [Google OAuth設定ガイド](google-oauth-setup.md)
- [メインREADME](../README.md)
# Google Calendar API OAuth2 設定ガイド

このガイドでは、Google Calendar APIを使用するためのOAuth2認証の設定方法を説明します。

## 前提条件

- Googleアカウント
- Google Cloud Platformプロジェクト

## Step 1: Google Cloud Consoleでプロジェクトを設定

### 1.1 プロジェクトの作成または選択

1. [Google Cloud Console](https://console.cloud.google.com/)にアクセス
2. 既存のプロジェクトを選択するか、新しいプロジェクトを作成

### 1.2 Google Calendar APIを有効化

1. ナビゲーションメニューから「APIとサービス」→「ライブラリ」を選択
2. 「Google Calendar API」を検索
3. 「有効にする」をクリック

## Step 2: OAuth2.0 認証情報の作成

### 2.1 OAuth同意画面の設定

1. 「APIとサービス」→「OAuth同意画面」を選択
2. ユーザータイプを選択（通常は「外部」）
3. 必要な情報を入力：
   - アプリ名: 「Meeting Scheduler」など
   - ユーザーサポートメール: あなたのメールアドレス
   - デベロッパー連絡先: あなたのメールアドレス
4. スコープの追加：
   - `.../auth/calendar.readonly`
   - `.../auth/calendar.events.readonly`
5. テストユーザーを追加（必要に応じて）

### 2.2 OAuth2.0クライアントIDの作成

1. 「APIとサービス」→「認証情報」を選択
2. 「認証情報を作成」→「OAuthクライアントID」を選択
3. アプリケーションの種類: 「ウェブアプリケーション」を選択
4. 名前: 「Meeting Scheduler Client」など
5. 承認済みのリダイレクトURI:
   - `http://localhost` を追加
   - `http://localhost:8080` を追加（オプション）
6. 「作成」をクリック
7. 表示されたクライアントIDとクライアントシークレットをコピー

## Step 3: .envファイルの設定

### 3.1 クライアント認証情報を.envに追加

```bash
# .envファイルを作成（まだない場合）
cp .env.example .env

# エディタで.envを開いて以下を設定
GOOGLE_CLIENT_ID=your-client-id-here
GOOGLE_CLIENT_SECRET=your-client-secret-here
```

## Step 4: リフレッシュトークンの取得

### 4.1 自動取得ツールを使用

```bash
# リフレッシュトークン取得ツールを実行
deno run --allow-net --allow-env --allow-read --allow-write \
  tools/get-google-refresh-token.ts
```

### 4.2 手動での取得手順

1. ツールが表示するURLをブラウザで開く
2. Googleアカウントでログイン
3. アプリケーションへのアクセスを許可
4. リダイレクトされたURL（`http://localhost/?code=XXXXX&scope=...`）から`code=`の後の部分をコピー
5. ツールに認証コードを入力
6. 表示されたリフレッシュトークンを.envファイルに保存

### 4.3 .envファイルの最終確認

```bash
# .envファイル
GOOGLE_CLIENT_ID=your-actual-client-id
GOOGLE_CLIENT_SECRET=your-actual-client-secret
GOOGLE_REFRESH_TOKEN=your-actual-refresh-token
```

## Step 5: 動作確認

```bash
# テスト実行
deno run --allow-net --allow-env --allow-read \
  meeting-scheduler/app.ts \
  -p "test:test@example.com:google" \
  --verbose
```

## トラブルシューティング

### 「リフレッシュトークンが返されませんでした」エラー

- 原因: すでに同じアプリケーションで認証済み
- 解決策:
  1. [Googleアカウント設定](https://myaccount.google.com/permissions)にアクセス
  2. 「Meeting Scheduler」のアクセス権を削除
  3. もう一度認証プロセスを実行

### 「invalid_grant」エラー

- 原因: 認証コードの有効期限切れ（数分で失効）
- 解決策: Step 4を最初からやり直す

### 「redirect_uri_mismatch」エラー

- 原因: リダイレクトURIが一致していない
- 解決策: Google Cloud ConsoleでリダイレクトURIに`http://localhost`が追加されているか確認

## セキュリティに関する注意事項

- **重要**: `.env`ファイルは絶対にGitにコミットしないでください
- クライアントシークレットとリフレッシュトークンは秘密情報として扱ってください
- 本番環境では、より安全な認証方法（サービスアカウントなど）の使用を検討してください

## 参考リンク

- [Google Calendar API Documentation](https://developers.google.com/calendar/api/v3/reference)
- [Google OAuth2 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [Google Cloud Console](https://console.cloud.google.com/)
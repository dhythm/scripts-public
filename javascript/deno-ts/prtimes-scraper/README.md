# PRTIMESスクレイパー

PRTIMESの新着リリース情報を取得して、JSON形式で保存するツールです。

## 機能

- Playwrightを利用してPRTIMESの新着リリース一覧をレンダリング後に取得
- 「もっと見る」ボタンを自動クリックし、初期表示以外のリリースも収集
- タイトル、企業名、URL、公開日時、相対時間、企業ID、サムネイル画像、タグなどを抽出
- JSON形式でファイルに保存
- オプションで企業APIからURL、業種、所在地、資本金、SNS（X/Facebook/YouTube など）を取得して統合

## 使い方

### 事前準備

初回実行時にPlaywright用ブラウザのダウンロードが必要です。

```bash
npx playwright install chromium
```

### 基本的な実行

```bash
deno run --allow-net --allow-read --allow-write --allow-env --allow-run --allow-ffi --allow-sys main.ts
```

デフォルトで `prtimes_releases.json` ファイルに結果が保存されます。

### オプション

```bash
# 出力ファイル名を指定
deno run --allow-net --allow-read --allow-write --allow-env --allow-run --allow-ffi --allow-sys main.ts -o output.json

# 詳細ログを表示
deno run --allow-net --allow-read --allow-write --allow-env --allow-run --allow-ffi --allow-sys main.ts -v

# ヘルプを表示
deno run --allow-net --allow-read --allow-write --allow-env --allow-run --allow-ffi --allow-sys main.ts -h
```

## テスト

```bash
deno test --allow-net --allow-env --allow-run --allow-read --allow-write --allow-ffi --allow-sys
```

## 出力例

```json
[
  {
    "title": "新製品のリリースのお知らせ",
    "company": "株式会社サンプル",
    "url": "https://prtimes.jp/main/html/rd/p/000000001.000000001.html",
    "date": "2025-01-15T10:00:00+0900",
    "relativeTime": "5分前",
    "releaseId": "000000001.000000001",
    "thumbnailUrl": "https://prtimes.jp/i/12345/1/thumb/118x78/sample.png",
    "companyId": "12345"
  }
]
```

### 企業URL付き（`-c`オプション使用時）

```json
[
  {
    "title": "新製品のリリースのお知らせ",
    "company": "株式会社サンプル",
    "url": "https://prtimes.jp/main/html/rd/p/000000001.000000001.html",
    "date": "2025-01-15T10:00:00+0900",
    "companyId": "12345",
    "companyUrl": "https://example.com",
    "companyInfo": {
      "url": "https://example.com",
      "industry": "情報通信業",
      "address": "東京都千代田区1-2-3",
      "capital": "1億円",
      "foundationDate": "2010-04",
      "representative": "山田太郎",
      "xUrl": "https://x.com/example",
      "facebookUrl": "https://www.facebook.com/example",
      "youtubeUrl": "https://www.youtube.com/@example",
      "followerCount": 1234
    }
  }
]
```

## 制限事項

- Playwrightの実行にはChromiumブラウザのダウンロードが必要です
- 企業APIで取得できないSNS（Instagramなど）は空欄になります
- 大量の企業情報を取得する場合はAPIへのアクセス数が増えるため、リクエスト間隔にご注意ください

## 必要な権限

- `--allow-net`: PRTIMESのウェブサイトへのアクセスに必要
- `--allow-read`: Playwrightがキャッシュやブラウザを読み込むために必要
- `--allow-write`: 結果の保存およびPlaywrightのキャッシュ書き込みに必要
- `--allow-env`: Playwright内で環境変数を参照するために必要
- `--allow-run`: PlaywrightがChromiumを起動するために必要
- `--allow-ffi` / `--allow-sys`: Playwrightの内部処理で必要となる場合があります

## 技術スタック

- Deno
- TypeScript
- Playwright
- cheerio (HTMLパーサー)

## ライセンス

MIT

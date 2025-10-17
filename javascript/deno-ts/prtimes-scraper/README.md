# PRTIMESスクレイパー

PRTIMESの新着リリース情報を取得して、JSON形式で保存するツールです。

## 機能

- PRTIMESの新着リリース一覧ページをスクレイピング
- タイトル、企業名、URL、公開日時を抽出
- JSON形式でファイルに保存

## 使い方

### 基本的な実行

```bash
deno run --allow-net --allow-write --allow-env main.ts
```

デフォルトで `prtimes_releases.json` ファイルに結果が保存されます。

### オプション

```bash
# 出力ファイル名を指定
deno run --allow-net --allow-write --allow-env main.ts -o output.json

# 詳細ログを表示
deno run --allow-net --allow-write --allow-env main.ts -v

# ヘルプを表示
deno run --allow-net --allow-write --allow-env main.ts -h
```

## テスト

```bash
deno test --allow-net --allow-env
```

## 出力例

```json
[
  {
    "title": "新製品のリリースのお知らせ",
    "company": "株式会社サンプル",
    "url": "https://prtimes.jp/main/html/rd/p/000000001.000000001.html",
    "date": "2025.01.15 10:00"
  }
]
```

## 必要な権限

- `--allow-net`: PRTIMESのウェブサイトへのアクセスに必要
- `--allow-write`: 結果をファイルに保存するために必要
- `--allow-env`: cheerioライブラリの依存関係で必要

## 技術スタック

- Deno
- TypeScript
- cheerio (HTMLパーサー)

## ライセンス

MIT

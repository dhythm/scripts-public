# Deno TypeScript プロジェクト

Deno を使用した TypeScript ツール集です。

## ディレクトリ構造

```
deno-ts/
├── src/                # サンプルアプリケーション
│   └── main.ts         # デモ用のメインファイル
├── meeting-scheduler/  # 会議の日程調整ツール（Google Calendar + HubSpot）
├── prtimes-scraper/    # PR TIMES のスクレイパー
├── deno.json           # Deno設定ファイル
├── deno.lock           # ロックファイル
└── README.md           # このファイル
```

## ツール一覧

### 1. meeting-scheduler - 会議の日程調整ツール

参加者の Google Calendar の空き状況から、会議の候補日時を提案します。

```bash
deno task meeting --help
```

詳細は [meeting-scheduler/README.md](meeting-scheduler/README.md) を参照してください。

### 2. prtimes-scraper - PR TIMES スクレイパー

詳細は [prtimes-scraper/README.md](prtimes-scraper/README.md) を参照してください。

PDFのテキスト抽出・OCRは Python の `pdfocr` に統合しました（[python/README.md](../../python/README.md) を参照）。

## 開発

### コマンド一覧

```bash
# 開発サーバー起動（サンプルアプリ）
deno task dev

# 型チェック
deno task check

# コードフォーマット
deno fmt

# Lint
deno lint

# テスト
deno task test
```

### 新しいツールの追加

新しいツールを追加する場合は、専用のディレクトリを作成し、関連ファイルをまとめてください：

```bash
mkdir my-tool
cd my-tool
# ツールのファイルを作成
```

`deno.json` にタスクを追加：

```json
{
  "tasks": {
    "my-tool": "deno run --allow-read my-tool/main.ts"
  }
}
```

## 要件

- Deno 1.40.0 以上
- 各ツール固有の要件は、それぞれのREADMEを参照してください
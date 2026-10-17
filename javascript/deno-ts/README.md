# Deno TypeScript プロジェクト

Deno を使用した TypeScript ツール集です。

## ディレクトリ構造

```
deno-ts/
├── meeting-scheduler/  # 会議の日程調整ツール（Google Calendar + HubSpot）
├── deno.json           # Deno設定ファイル
├── deno.lock           # ロックファイル
└── README.md           # このファイル
```

## ツール一覧

### meeting-scheduler - 会議の日程調整ツール

参加者の Google Calendar と HubSpot の空き状況から、会議の候補日時を提案します。

```bash
deno task meeting --help
```

詳細は [meeting-scheduler/README.md](meeting-scheduler/README.md) を参照してください。

PDFのテキスト抽出・OCRは Python の `pdfocr` に統合しました（[python/README.md](../../python/README.md) を参照）。

## 開発

```bash
# 型チェック
deno task meeting:check

# テスト
deno task test

# コードフォーマット / Lint
deno fmt
deno lint
```

新しいツールを追加する場合は、専用のディレクトリを作成し、`deno.json` にタスクを追加してください。

## 要件

- Deno 2.x
- 各ツール固有の要件は、それぞれのREADMEを参照してください

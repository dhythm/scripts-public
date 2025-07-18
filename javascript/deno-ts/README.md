# Deno TypeScript プロジェクト

Deno を使用した TypeScript ツール集です。

## ディレクトリ構造

```
deno-ts/
├── src/              # サンプルアプリケーション
│   └── main.ts       # デモ用のメインファイル
├── check-pdf/        # PDFテキストチェックツール
│   ├── check-pdf-text.ts
│   ├── check-pdf-text-simple.ts
│   └── README.md
├── pdf-extract/      # PDF テキスト抽出ツール（Google Cloud連携）
│   ├── pdf-extract.ts
│   ├── app.ts
│   ├── cli/
│   ├── clients/
│   ├── handlers/
│   ├── processors/
│   ├── types/
│   ├── utils/
│   └── README.md
├── deno.json         # Deno設定ファイル
├── deno.lock         # ロックファイル
└── README.md         # このファイル
```

## ツール一覧

### 1. check-pdf - PDFテキストチェックツール

PDFファイルにテキストレイヤーが含まれているかを確認します。

```bash
# pdf-libバージョン
deno task check-pdf <PDFファイルパス>

# 軽量バージョン（外部ライブラリ不要）
deno task check-pdf-simple <PDFファイルパス>
```

詳細は [check-pdf/README.md](check-pdf/README.md) を参照してください。

### 2. pdf-extract - PDFテキスト抽出ツール

Google Cloud Vision API または Document AI を使用してPDFからテキストを抽出します。

```bash
# 基本的な使用方法
deno task pdf-extract <入力ディレクトリ>

# Document AI を使用
deno task pdf-extract <入力ディレクトリ> -a documentai

# 詳細オプション付き
deno task pdf-extract <入力ディレクトリ> -o <出力ディレクトリ> -v
```

詳細は [pdf-extract/README.md](pdf-extract/README.md) を参照してください。

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
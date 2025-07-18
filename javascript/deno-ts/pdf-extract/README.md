# PDF テキスト抽出ツール

Google Cloud Vision API または Document AI を使用して PDF ファイルからテキストを抽出する Deno アプリケーションです。

## 必要な環境

- Deno 1.40.0 以上
- Google Cloud Platform アカウント
- Vision API または Document AI の有効化
- サービスアカウントキー

## セットアップ

### 1. Google Cloud の設定

1. [Google Cloud Console](https://console.cloud.google.com) にアクセス
2. Vision API または Document AI を有効化
3. サービスアカウントを作成し、JSONキーをダウンロード
4. Document AI を使用する場合は、プロセッサを作成してIDを取得

### 2. 環境変数の設定

```bash
# サービスアカウントキーのパス
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Google Cloud プロジェクトID（オプション - サービスアカウントから自動取得可能）
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Document AI を使用する場合（オプション）
export DOCUMENT_AI_PROCESSOR_ID="your-processor-id"
```

### 3. 認証方法

以下のいずれかの方法で認証できます：

#### 方法1: Google Cloud APIキー（推奨）
```bash
export GOOGLE_API_KEY="your-api-key"
```

#### 方法2: gcloud CLIを使用
```bash
# gcloud CLIでログイン
gcloud auth application-default login
```

#### 方法3: サービスアカウント（上記の環境変数設定）

## 使用方法

### 基本的な使用方法

```bash
# Vision API を使用してすべての PDF を処理
deno task pdf-extract ./pdfs

# または直接実行
deno run --allow-read --allow-write --allow-env --allow-net src/pdf-extract.ts ./pdfs
```

### オプション

```bash
# Document AI を使用
deno task pdf-extract ./pdfs -a documentai

# 出力ディレクトリを指定
deno task pdf-extract ./pdfs -o ./output

# ファイルパターンを指定
deno task pdf-extract ./pdfs -p "invoice_*.pdf"

# 並行処理数を変更（デフォルト: 3）
deno task pdf-extract ./pdfs -c 5

# 詳細ログを表示
deno task pdf-extract ./pdfs -v

# ヘルプを表示
deno task pdf-extract -h
```

### マージ機能

複数のPDFから抽出したテキストを1つのファイルにまとめることができます：

```bash
# すべてのテキストを merged_output.txt に結合
deno task pdf-extract ./pdfs -m

# カスタムセパレータでマージ
deno task pdf-extract ./pdfs -m --merge-separator "\n--- 次のファイル ---\n"

# 出力ディレクトリを指定してマージ
deno task pdf-extract ./pdfs -m -o ./output
```

マージされたファイルは、出力ディレクトリに `merged_output.txt` として保存されます。
ファイルはアルファベット順にソートされ、各ファイルの内容の前にファイル名がヘッダーとして挿入されます。

### 複数のオプションを組み合わせる

```bash
# Document AI を使用し、特定のパターンのファイルを5並列で処理
deno task pdf-extract ./pdfs -a documentai -p "report_*.pdf" -c 5 -v

# すべてのPDFを処理してマージ
deno task pdf-extract ./pdfs -m -v -o ./output
```

## 出力

- 抽出されたテキストは、元のPDFファイルと同じ名前で `.txt` 拡張子のファイルとして保存されます
- デフォルトでは元のPDFファイルと同じディレクトリに保存されます
- `-o` オプションで出力ディレクトリを指定できます

## トラブルシューティング

### 認証エラー

```
エラー: 環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていません
```

認証方法のセクションを参照して、適切な認証方法を設定してください。

#### Maximum call stack size exceeded エラー

大きなPDFファイルを処理する際に発生する場合があります。このエラーは内部的に処理されるため、通常は問題ありません。

### レート制限エラー

API のレート制限に達した場合、自動的にリトライされます。`-c` オプションで並行処理数を減らすことで、レート制限を回避できます。

### Document AI エラー

```
エラー: 環境変数 DOCUMENT_AI_PROCESSOR_ID が設定されていません
```

Document AI を使用する場合は、プロセッサIDを環境変数に設定する必要があります。

## 開発

### 型チェック

```bash
deno task pdf-extract:check
```

### コードフォーマット

```bash
deno fmt
```

### Lint

```bash
deno lint
```
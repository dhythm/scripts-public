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
# すべてのテキストを merged_output.txt に結合（個別ファイルは作成されません）
deno task pdf-extract ./pdfs -m

# カスタムセパレータでマージ
deno task pdf-extract ./pdfs -m --merge-separator "\n--- 次のファイル ---\n"

# 出力ディレクトリを指定してマージ
deno task pdf-extract ./pdfs -m -o ./output
```

**注意**: `-m` オプションを使用すると、個別のテキストファイルは作成されず、`merged_output.txt` のみが出力されます。
マージされたファイルは、出力ディレクトリに保存されます。
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

## Document AI を使用したPDF文字起こしツール

### 概要
`document-ai-pdf-extract.ts` は、Google Cloud Document AI を使用してPDFファイルからテキストを抽出する単独のTypeScriptスクリプトです。

### 使用方法

```bash
# 直接実行（実行権限が付与されている場合）
./document-ai-pdf-extract.ts <PDFファイルパス>

# または Deno コマンドで実行
deno run --allow-env --allow-read --allow-net document-ai-pdf-extract.ts <PDFファイルパス>
```

### 例

```bash
# PDFファイルからテキストを抽出
./document-ai-pdf-extract.ts ../../000928313.pdf
```

### 機能
- 関数型プログラミング（classを使用せず）
- 単一ファイルに全ての処理を含む
- 完全な型定義付きTypeScript
- Google Cloud サービスアカウント認証
- Document AI APIによるOCR処理
- 自動リトライ機能（レート制限・一時的エラー対応）
- エラーハンドリング
- 進行状況の表示

### 必要な環境変数
```bash
# Google Cloud サービスアカウントキーのパス
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Document AI プロセッサID
export DOCUMENT_AI_PROCESSOR_ID="your-processor-id"

# プロジェクトID（オプション：サービスアカウントキーに含まれる場合は不要）
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### デバッグモード

認証エラーやその他の問題が発生した場合、デバッグモードで詳細情報を確認できます：

```bash
# デバッグ情報を表示して実行
DEBUG=1 ./document-ai-pdf-extract.ts <PDFファイルパス>
```

デバッグモードでは以下の情報が表示されます：
- JWT payloadの内容
- Token URI
- トークンレスポンス
- エラーの詳細

### 注意事項
- Document AI APIの使用には Google Cloud プロジェクトが必要です
- プロセッサIDは Google Cloud Console の Document AI セクションで作成・取得できます
- APIの使用量に応じて課金が発生する可能性があります

## Document AI SDK版 （推奨）

### 概要
`document-ai-pdf-extract-sdk.ts` は、Google Cloud Document AI の公式 SDK を使用したPDF文字起こしツールです。
認証の複雑さをSDKが吸収してくれるため、よりシンプルで安定した実装となっています。

### 使用方法

```bash
# 実行（実行権限が付与されている場合）
./document-ai-pdf-extract-sdk.ts <PDFファイルパス>

# または Deno コマンドで実行
deno run --allow-env --allow-read --allow-net --allow-sys document-ai-pdf-extract-sdk.ts <PDFファイルパス>
```

### 例

```bash
# PDFファイルからテキストを抽出
./document-ai-pdf-extract-sdk.ts ../../000928313.pdf

# デバッグモードで実行
DEBUG=1 ./document-ai-pdf-extract-sdk.ts ../../000928313.pdf
```

### 必要な環境変数
```bash
# Google Cloud サービスアカウントキーのパス
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Google Cloud プロジェクトID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Document AI プロセッサID
export DOCUMENT_AI_PROCESSOR_ID="your-processor-id"

# リージョン（オプション、デフォルト: us）
export DOCUMENT_AI_LOCATION="us"  # または "eu"
```

### 特徴
- Google Cloud Document AI 公式 SDK を使用
- 認証処理がSDKに任せられるためシンプル
- EUリージョンのサポート
- エラーハンドリングの改善
- デバッグモードで詳細情報表示

### SDK版と手動実装版の違い

| 項目 | SDK版 (`-sdk.ts`) | 手動実装版 (`.ts`) |
|------|-------------------|-------------------|
| 認証 | SDKが自動処理 | JWTを手動で生成 |
| 依存関係 | npmパッケージあり | 依存関係なし |
| コード量 | 少ない | 多い |
| メンテナンス | 簡単 | 複雑 |
| 推奨度 | 高（推奨） | 低 |

通常はSDK版の使用を推奨します。

## gcloud CLI版 （最も安定）

### 概要
`document-ai-pdf-extract-gcloud.ts` は、gcloud CLI を使用して認証を行うシンプルな実装です。
SDK版で問題が発生する場合は、こちらを使用してください。

### 事前準備
```bash
# gcloud CLI のインストール（まだの場合）
# https://cloud.google.com/sdk/docs/install

# gcloud での認証
gcloud auth application-default login

# プロジェクトの設定（オプション）
gcloud config set project YOUR_PROJECT_ID
```

### 使用方法
```bash
# 実行
./document-ai-pdf-extract-gcloud.ts <PDFファイルパス>

# または
deno run --allow-env --allow-read --allow-net --allow-run document-ai-pdf-extract-gcloud.ts <PDFファイルパス>
```

### 例
```bash
# PDFファイルからテキストを抽出
DOCUMENT_AI_PROCESSOR_ID=d92387f0d5deee12 ./document-ai-pdf-extract-gcloud.ts ../../000928313.pdf

# デバッグモードで実行
DEBUG=1 DOCUMENT_AI_PROCESSOR_ID=d92387f0d5deee12 ./document-ai-pdf-extract-gcloud.ts ../../000928313.pdf
```

### 必要な環境変数
```bash
# Document AI プロセッサID（必須）
export DOCUMENT_AI_PROCESSOR_ID="your-processor-id"

# Google Cloud プロジェクトID（オプション、gcloudから自動取得）
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### 特徴
- gcloud CLI を使用したシンプルな認証
- サービスアカウントキーの設定不要
- Deno との互換性問題がない
- 最も安定して動作

## 3つの実装の比較

| 項目 | SDK版 | 手動実装版 | gcloud CLI版 |
|------|--------|------------|---------------|
| ファイル名 | `document-ai-pdf-extract-sdk.ts` | `document-ai-pdf-extract.ts` | `document-ai-pdf-extract-gcloud.ts` |
| 認証方法 | SDKが自動処理 | JWTを手動生成 | gcloud CLI 経由 |
| 依存関係 | npmパッケージ | なし | gcloud CLI |
| コード量 | 少ない | 多い | 中程度 |
| Deno互換性 | 問題あり（TLSエラー） | 良好 | 完璧 |
| 安定性 | 低 | 中 | 高 |
| 推奨度 | 低（Denoでは） | 中 | **高（推奨）** |

### 推奨使用順序

1. **gcloud CLI版** - 最も安定して動作し、設定も簡単
2. **手動実装版** - gcloudが使えない場合の代替案
3. **SDK版** - Node.js環境でのみ推奨

### トラブルシューティング

#### SDK版でTLSエラーが発生する場合
```
Error: Client network socket disconnected before secure TLS connection was established
```
→ gcloud CLI版を使用してください

#### 手動実装版で id_token エラーが発生する場合
```
認証エラー: access_tokenではなくid_tokenが返されました
```
→ gcloud CLI版を使用してください

#### gcloud CLI版で gcloud が見つからない場合
```
エラー: gcloud CLI がインストールされていません
```
→ [gcloud CLI をインストール](https://cloud.google.com/sdk/docs/install)してください
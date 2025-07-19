# PDF Text Extract

PDFファイル内のテキスト層を抽出するためのDenoスクリプトです。OCRは使用せず、既存のテキスト情報のみを高速に取得します。

## 特徴

- 🚀 高速処理（OCR不要）
- 📁 ディレクトリ一括処理
- 🔀 並行処理対応
- 📄 マージオプション
- 🎯 ファイルパターンマッチング
- 📚 複数ライブラリサポート（pdf-ts、pdf-parse）

## 必要環境

- Deno 1.40以降

## インストール不要

Denoスクリプトのため、特別なインストールは不要です。

## 使用方法

### 基本的な使用

```bash
# すべてのPDFを処理
deno run --allow-all pdf-text-extract.ts ./pdfs

# 出力ディレクトリを指定
deno run --allow-all pdf-text-extract.ts ./pdfs -o ./output
```

### オプション

```
-o, --output <dir>        出力ディレクトリ (デフォルト: 入力ディレクトリと同じ)
-l, --library <type>      使用するライブラリ (pdf-ts | pdf-parse) (デフォルト: pdf-ts)
-p, --pattern <pattern>   ファイルパターン (例: "*.pdf", "invoice_*.pdf")
-c, --concurrency <num>   並行処理数 (1-10) (デフォルト: 3)
-m, --merge               すべての抽出されたテキストを1つのファイルにマージ
--merge-separator <text>  マージ時のファイル間セパレータ (デフォルト: 区切り線)
-v, --verbose             詳細なログ出力
-h, --help                ヘルプを表示
```

### 使用例

```bash
# pdf-parseライブラリを使用
deno run --allow-all pdf-text-extract.ts ./pdfs -l pdf-parse

# 特定のパターンのファイルのみ処理
deno run --allow-all pdf-text-extract.ts ./pdfs -p "report_*.pdf"

# 並行処理数を増やして高速化
deno run --allow-all pdf-text-extract.ts ./pdfs -c 5

# すべてのテキストを1つのファイルにマージ
deno run --allow-all pdf-text-extract.ts ./pdfs -m -o ./output

# カスタムセパレータでマージ
deno run --allow-all pdf-text-extract.ts ./pdfs -m --merge-separator "\n--- 次のファイル ---\n"
```

## ライブラリの違い

### pdf-ts
- 依存関係が最小
- Promiseベース
- UTF-8対応
- 軽量で高速

### pdf-parse
- より詳細な情報（ページ数など）を取得可能
- 表や改行の復元にやや弱い
- 広く使われている実績あり

## 注意事項

- このツールはPDF内のテキスト層を抽出します
- 画像のみのPDF（スキャンされたPDFなど）からはテキストを抽出できません
- OCRが必要な場合は、別途`pdf-extract`スクリプトをご利用ください

## エラー時の対処

- `テキスト層なし`と表示される場合、そのPDFは画像のみの可能性があります
- メモリ不足エラーが発生する場合は、並行処理数を減らしてください（-c 1）

## ライセンス

MIT
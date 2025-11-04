# scripts-public

公開用ユーティリティスクリプト集

## セットアップ

### 依存関係のインストール

[uv](https://docs.astral.sh/uv/)を使用して依存関係をインストールします：

```bash
uv sync
```

## Python スクリプト

### PDF → Markdown 変換

縦書きPDFから読みやすいMarkdownファイルを生成します。

#### 実行方法

```bash
# uvを使用（推奨）
uv run python python/pdf_to_markdown.py <PDFファイルパス> [出力ファイルパス]

# または直接実行
python3 python/pdf_to_markdown.py <PDFファイルパス> [出力ファイルパス]
```

#### 使用例

```bash
# 基本的な使い方（出力ファイルは自動生成）
uv run python python/pdf_to_markdown.py document.pdf

# 出力ファイル名を指定
uv run python python/pdf_to_markdown.py document.pdf output.md
```

#### 特徴

- **縦書き対応**: `pdfminer.six`の縦書き検出機能を使用
- **自動構造化**: 章・節を自動認識してMarkdown見出しに変換
- **特殊文字の正規化**: 縦書き記号（︑、︒など）を横書きに自動変換
- **重複除去**: 同じセクションの重複を自動検出して除外
- **フォールバック**: `pdfminer.six` → `pdfplumber` → `PyPDF2` の順に試行

#### 依存ライブラリ

- `pdfminer.six`: メインの抽出エンジン（縦書き対応）
- `pdfplumber`: フォールバック用
- `PyPDF2`: フォールバック用

### その他のPDFスクリプト

#### OCRによるテキスト抽出

画像ベースのPDFからテキストを抽出する場合：

```bash
python3 python/pdf_ocr.py <PDFファイルパス>
```

**注意**: OCR機能にはTesseractのインストールが必要です：

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-jpn
```

#### シンプルなテキスト抽出

```bash
python3 python/pdf_to_text.py <PDFファイルパス>
```

## ライセンス

MIT License
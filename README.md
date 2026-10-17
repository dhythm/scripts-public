# scripts-public

公開用ユーティリティスクリプト集

## セットアップ

### 作業ディレクトリと依存関係

このリポジトリには独立した実行環境があります。ルートの `uv sync` は `python/` 以下の全ツールを準備するコマンドではありません。

| 場所 | 用途 | 準備 |
|---|---|---|
| ルート | 主にPDFのローカル抽出・SVG用の小さいPython環境 | `uv sync` |
| `python/` | OCR・サイト調査・画像・音声などのPython CLI | `cd python` → `uv sync` |
| `javascript/node-ts/` | 情報源調査、TTSのNode CLI | 同ディレクトリで `npm ci` |
| `javascript/deno-ts/` | PDF抽出、会議候補、PRTIMES等のDeno CLI | 同ディレクトリのREADME・`deno.json`参照 |

各Python環境は別の `pyproject.toml` と `uv.lock` を使います。`site-inspector` のentry pointは `python/` 側にのみあります。
音声検証で使用中の環境や共有依存を置き換える必要はありません。既存の仮想環境に対する一括更新は避け、対象ツールの手順を選んでください。

### ルートの依存関係のインストール

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
uv run python python/pdf_to_markdown.py <PDFファイルパス> [-o 出力ファイルパス]

# または直接実行
python3 python/pdf_to_markdown.py <PDFファイルパス> [-o 出力ファイルパス]
```

#### 使用例

```bash
# 基本的な使い方（出力ファイルは自動生成）
uv run python python/pdf_to_markdown.py document.pdf

# 出力ファイル名を指定
uv run python python/pdf_to_markdown.py document.pdf -o output.md
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

画像ベースのPDFからテキストを抽出する場合は、OCR依存を持つ `python/` 側で準備・実行します：

```bash
cd python
uv sync
uv run python pdf_ocr.py <PDFファイルパス>
```

**注意**: OCRにはTesseractと言語データに加え、`pdf2image`が呼ぶPopplerが必要です：

```bash
# macOS
brew install tesseract tesseract-lang poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-jpn poppler-utils
```

#### シンプルなテキスト抽出

```bash
python3 python/pdf_to_text.py <PDFファイルパス>
```

## ライセンス

MIT License
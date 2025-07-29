# Python

## Getting Started

```sh
uv sync
```

```sh
uv add ruff
uv run ruff check
```

## PDF OCR (日本語対応)

PDFファイルからOCRを使用してテキストを抽出するツールです。

### セットアップ

1. 依存関係のインストール
```sh
uv sync
```

2. 日本語OCRのセットアップ
```sh
./setup_japanese_ocr.sh
```

### 使用方法

```sh
uv run python pdf_ocr.py <PDFファイル>
```

例:
```sh
uv run python pdf_ocr.py 000928313.pdf
```

### 特徴

- 日本語と英語の両方に対応
- 画像ベースのPDF（スキャンされたPDF）からもテキスト抽出可能
- 画像前処理により、OCRの精度を向上
- 各ページごとにテキストを抽出し、`.ocr.txt`ファイルとして保存
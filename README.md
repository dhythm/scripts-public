# scripts-public

公開用ユーティリティスクリプト集

## セットアップ

| 場所 | 用途 | 準備 |
|---|---|---|
| `python/` | PDF OCR・文字起こし・画像→SVG・サイト調査・音声などのPython CLI | `cd python` → `uv sync` |
| `javascript/deno-ts/` | 会議の日程調整（meeting-scheduler）のDeno CLI | 同ディレクトリのREADME・`deno.json`参照 |
| `shell/` | 単体のシェルスクリプト | 不要 |

Pythonのツールは `python/` の `pyproject.toml` と `uv.lock` で管理しています（[uv](https://docs.astral.sh/uv/) を使用）。
各ツールの使い方は [python/README.md](python/README.md) を参照してください。

## Python スクリプト

### PDF → テキスト / Markdown（pdfocr）

PDFのテキスト抽出とOCRは `python/pdfocr` に統合しています。テキスト層があるページはそのまま抽出し、
スキャン画像のページだけをローカルのOCR（NDLOCR-Lite / YomiToku）にかけます。クラウドAPIは使いません。

```bash
cd python
uv sync
uv run python -m pdfocr setup ndl          # 初回のみ
uv run python -m pdfocr book.pdf --format md
```

詳細は [python/README.md](python/README.md) の「PDF OCR (pdfocr)」を参照してください。

## ライセンス

MIT License
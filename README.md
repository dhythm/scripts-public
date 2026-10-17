# scripts-public

公開用ユーティリティスクリプト集

## セットアップ

### 作業ディレクトリと依存関係

このリポジトリには独立した実行環境があります。ルートの `uv sync` は `python/` 以下の全ツールを準備するコマンドではありません。

| 場所 | 用途 | 準備 |
|---|---|---|
| ルート | SVG用の小さいPython環境 | `uv sync` |
| `python/` | PDF OCR・文字起こし・サイト調査・画像・音声などのPython CLI | `cd python` → `uv sync` |
| `javascript/node-ts/` | 情報源調査、TTSのNode CLI | 同ディレクトリで `npm ci` |
| `javascript/deno-ts/` | 会議の日程調整（meeting-scheduler）のDeno CLI | 同ディレクトリのREADME・`deno.json`参照 |

各Python環境は別の `pyproject.toml` と `uv.lock` を使います。`site-inspector` のentry pointは `python/` 側にのみあります。
音声検証で使用中の環境や共有依存を置き換える必要はありません。既存の仮想環境に対する一括更新は避け、対象ツールの手順を選んでください。

### ルートの依存関係のインストール

[uv](https://docs.astral.sh/uv/)を使用して依存関係をインストールします：

```bash
uv sync
```

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
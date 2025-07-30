# Python

## Getting Started

```sh
uv sync
```

```sh
uv add ruff
uv run ruff check
```

## 音声文字起こしツール

音声ファイルからテキストを抽出する2つのツールを提供しています。

**注意**: 現在、Python 3.10以降では`openai-whisper`の依存関係の問題により、`transcribe_faster_whisper.py`のみが使用可能です。`transcribe_whisper.py`を使用する場合は、`WHISPER_SETUP.md`を参照してください。

### 1. OpenAI Whisper版 (transcribe_whisper.py)

OpenAI公式のWhisperライブラリを使用した音声文字起こしツールです。（Python 3.9が必要）

#### 使用方法

```sh
# 基本的な使用方法
uv run python transcribe_whisper.py input.mp3

# モデルと言語を指定
uv run python transcribe_whisper.py input.mp3 --model medium --language ja

# JSON形式でタイムスタンプ付きで出力
uv run python transcribe_whisper.py input.wav --output result.json --format json --timestamps

# SRT字幕形式で出力
uv run python transcribe_whisper.py input.mp4 --output subtitles.srt --format srt
```

#### 特徴

- 複数のモデルサイズ（tiny, base, small, medium, large）をサポート
- 日本語を含む多言語対応
- 自動言語検出機能
- テキスト、JSON、SRT形式での出力
- タイムスタンプ付き出力オプション
- 音声ファイルの自動変換（mp3, wav, m4a, flac等に対応）

### 2. Faster-Whisper版 (transcribe_faster_whisper.py)

CTranslate2ベースの高速実装を使用した音声文字起こしツールです。

#### 使用方法

```sh
# 基本的な使用方法
uv run python transcribe_faster_whisper.py input.mp3

# モデルと言語を指定
uv run python transcribe_faster_whisper.py input.mp3 --model large-v2 --language ja

# GPUを使用して高速処理
uv run python transcribe_faster_whisper.py input.wav --device cuda --compute_type float16

# 単語レベルのタイムスタンプ付きJSON出力
uv run python transcribe_faster_whisper.py input.mp4 --format json --word_timestamps --output result.json

# WebVTT字幕形式で出力
uv run python transcribe_faster_whisper.py input.mp3 --format vtt --output subtitles.vtt
```

#### 特徴

- OpenAI Whisperより高速な処理
- GPU/CPU自動選択
- メモリ効率的な処理
- 単語レベルのタイムスタンプ生成
- Voice Activity Detection (VAD) フィルター
- テキスト、JSON、SRT、WebVTT形式での出力
- 信頼度情報の出力オプション

### 共通オプション

両ツール共通の主要オプション：

- `--model`: 使用するモデルサイズ
- `--language`: 音声の言語コード（ja, en等）
- `--output`: 出力ファイルパス
- `--format`: 出力形式（text, json, srt等）
- `--timestamps`: タイムスタンプを含める
- `--task`: transcribe（文字起こし）またはtranslate（英語翻訳）

### インストール

音声文字起こしツールを使用する場合は、以下のコマンドで依存関係をインストールしてください：

```sh
uv sync
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
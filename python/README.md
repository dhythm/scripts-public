# Python

## Getting Started

```sh
uv sync
```

```sh
uv add ruff
uv run ruff check
```

## Google PDF パイプライン

Google検索APIとPlaywrightを組み合わせ、PDFの取得から文字抽出・LLM整形までを自動化するスクリプト `google_pdf_pipeline.py` を追加しました。

### 必要な環境変数

- `GOOGLE_API_KEY`
- `GOOGLE_CSE_ID`
- `OPENAI_API_KEY`（LLM整形を行う場合）

`python/.env.example` を `python/.env` にコピーし、値を設定してください（ファイルはリポジトリ直下ではなく `python/` ディレクトリ内に配置しています）。

### 実行例

```sh
# Google検索で上位1件のPDFを処理し、そのまま標準出力にJSONで表示
uv run python google_pdf_pipeline.py "site:prtimes.jp プレスリリース PDF" -n 1

# OCRフォールバックを無効化し、結果をファイルに保存
uv run python google_pdf_pipeline.py "教育委員会 PDF 報告書" -n 2 --disable-ocr --output results.json

# 追加の整形指示を与えてLLMに要約させる
uv run python google_pdf_pipeline.py "サステナビリティ レポート PDF" --prompt "要約の最後に推奨アクションを2つ挙げてください。"
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

#### 基本的な使用方法

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

#### ユースケース別の使用方法

##### 1. 音声が小さい・遠い場合

```sh
# ノーマライズを使用（音量を自動調整）
uv run python transcribe_faster_whisper.py recording.m4a --normalize

# ノーマライズ + VAD閾値を下げる（より敏感に）
uv run python transcribe_faster_whisper.py recording.m4a --normalize --vad_threshold 0.25
```

##### 2. ノイズが多い環境の場合

```sh
# ノイズ除去 + ノーマライズ（推奨）
uv run python transcribe_faster_whisper.py noisy_audio.m4a --denoise --normalize --noise_reduce_amount 0.5

# ノイズ除去の強度を調整（0.0-1.0、デフォルト0.8）
uv run python transcribe_faster_whisper.py noisy_audio.m4a --denoise --noise_reduce_amount 0.6
```

##### 3. 音声が小さくノイズも多い場合

```sh
# ノイズ除去 + ノーマライズ + VAD調整の組み合わせ
uv run python transcribe_faster_whisper.py difficult_audio.m4a \
  --denoise \
  --normalize \
  --noise_reduce_amount 0.5 \
  --vad_threshold 0.3
```

##### 4. 手動でゲインを調整したい場合

```sh
# +15dB のゲインを追加
uv run python transcribe_faster_whisper.py quiet_audio.m4a --gain 15

# ゲインを追加してから VAD 調整
uv run python transcribe_faster_whisper.py quiet_audio.m4a --gain 12 --vad_threshold 0.25
```

##### 5. VADパラメータの細かい調整

```sh
# 短い発話も拾う（最小音声長を調整）
uv run python transcribe_faster_whisper.py audio.m4a \
  --vad_min_speech_duration 100 \
  --vad_min_silence_duration 800

# 音声の切れを防ぐ（パディングを増やす）
uv run python transcribe_faster_whisper.py audio.m4a --vad_speech_pad 600
```

#### 主要オプション

##### 音声前処理オプション

| オプション | 説明 | デフォルト | 推奨値 |
|-----------|------|-----------|--------|
| `--normalize` | 音声をピークノーマライズ（-3.0 dBFS） | なし | 音声が小さい場合に推奨 |
| `--gain` | ゲイン調整(dB)。正の値で音量アップ | 0 | 10-15 dB |
| `--denoise` | ノイズ除去を適用 | なし | ノイズが多い場合に推奨 |
| `--noise_reduce_amount` | ノイズ除去の強度（0.0-1.0） | 0.8 | 0.5-0.6（強すぎると音声劣化） |

##### VAD（Voice Activity Detection）オプション

| オプション | 説明 | デフォルト | 推奨値 |
|-----------|------|-----------|--------|
| `--vad_threshold` | VAD閾値（0-1）。低いほど敏感 | 0.30 | 0.25-0.35（小さい音声用） |
| `--vad_min_speech_duration` | 最小音声長(ms) | 100 | 100-300 |
| `--vad_min_silence_duration` | 最小無音長(ms) | 1000 | 800-1500 |
| `--vad_speech_pad` | 音声の前後パディング(ms) | 400 | 400-600 |
| `--no_vad_filter` | VADフィルタを無効化 | - | デバッグ時のみ |

##### 出力オプション

| オプション | 説明 |
|-----------|------|
| `--format` | 出力形式（text/json/srt/vtt） |
| `--timestamps` | タイムスタンプを含める |
| `--word_timestamps` | 単語レベルのタイムスタンプ |
| `--confidence` | 信頼度情報を含める |

##### モデル・処理オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--model` | モデルサイズ（tiny/base/small/medium/large-v3） | base |
| `--language` | 言語コード（ja/en等） | 自動検出 |
| `--device` | デバイス（cuda/cpu/auto） | auto |
| `--beam_size` | ビームサーチの幅 | 5 |
| `--initial_prompt` | 初期プロンプト（文脈を与える） | - |

#### パラメータ調整のコツ

1. **まず`--normalize`を試す**
   - 音声が小さい場合の最も簡単な解決策
   - ノイズも増幅されるので注意

2. **ノイズが多い場合は`--denoise`**
   - `--noise_reduce_amount 0.5`程度から始める
   - 強すぎると音声の質が劣化するので注意

3. **VAD閾値の調整**
   - 音声が削減されすぎる場合：`--vad_threshold 0.25`
   - ノイズを拾いすぎる場合：`--vad_threshold 0.35`

4. **処理の順序**
   - ノイズ除去 → ゲイン調整 → ノーマライズ → VAD
   - この順序で自動的に処理されます

#### トラブルシューティング

##### 問題: 文字起こし結果が意味不明・精度が極端に低い

**考えられる原因:**
- 元の音声ファイルの品質が極端に悪い
- 音声が小さすぎる（ピークが-20 dBFS以下）
- S/N比（信号対雑音比）が非常に悪い
- 録音距離が遠すぎる

**試すべき設定（優先順）:**

1. **より大きいモデルを使用**
   ```sh
   # mediumモデルで精度向上
   uv run python transcribe_faster_whisper.py recording.m4a \
     --model medium \
     --language ja
   ```

2. **VADを無効化して全体を処理**
   ```sh
   # VADによる音声の削減を防ぐ
   uv run python transcribe_faster_whisper.py recording.m4a \
     --model medium \
     --language ja \
     --no_vad_filter
   ```

3. **ノイズ除去を追加（弱め）**
   ```sh
   # ノイズ除去の強度を弱めに設定
   uv run python transcribe_faster_whisper.py recording.m4a \
     --model medium \
     --language ja \
     --denoise \
     --normalize \
     --noise_reduce_amount 0.3 \
     --no_vad_filter
   ```

**重要な注意点:**
- **元の音声品質が悪い場合、どの設定でも満足できる結果は得られません**
- 人間が聞いても理解できないレベルの音声は、AIでも正確な文字起こしは不可能です
- まず音声ファイルを実際に聞いて、人間が理解できるかを確認してください

##### 問題: 文字数が少なすぎる（音声が削減されすぎる）

**解決策:**

1. **VAD閾値を下げる**
   ```sh
   uv run python transcribe_faster_whisper.py recording.m4a \
     --normalize \
     --vad_threshold 0.20
   ```

2. **VADを完全に無効化**
   ```sh
   uv run python transcribe_faster_whisper.py recording.m4a \
     --normalize \
     --no_vad_filter
   ```

##### 問題: ノイズを拾いすぎる（不要な音まで文字起こしされる）

**解決策:**

1. **VAD閾値を上げる**
   ```sh
   uv run python transcribe_faster_whisper.py recording.m4a \
     --vad_threshold 0.40
   ```

2. **最小音声長を調整**
   ```sh
   uv run python transcribe_faster_whisper.py recording.m4a \
     --vad_min_speech_duration 300
   ```

##### 録音品質の改善（今後のために）

文字起こし精度を上げるには、**録音時の品質改善が最も重要**です：

- **マイクと話者の距離**: 30cm以内が理想
- **録音レベル**: ピークが-12 dBFS～-6 dBFS
- **環境**: 静かな部屋（エアコン・ファンを止める）
- **フォーマット**: 16-bit/48kHz以上
- **マイク**: できれば外付けマイク、ピンマイクを使用

#### 特徴

- OpenAI Whisperより高速な処理
- GPU/CPU自動選択
- メモリ効率的な処理
- 単語レベルのタイムスタンプ生成
- Voice Activity Detection (VAD) フィルター（小さい音声に最適化済み）
- 音声前処理機能（ノーマライズ、ゲイン調整、ノイズ除去）
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

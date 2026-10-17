# Python

## Getting Started

```sh
uv sync
```

## 音声の無音区間を削除（remove_silence.py）

室内収録など、発話と無音が明確な音声から長い無音区間を削除し、発話部分だけを連結します。文字起こしやAI APIは使用しません。

```sh
# 同じ場所に input_speech.m4a を出力
uv run python remove_silence.py input.m4a

# 出力先を指定
uv run python remove_silence.py input.wav -o output.wav

# まず検出区間だけを確認
uv run python remove_silence.py input.m4a --dry-run --list-segments
```

既定では、`-40 dBFS`以下の状態が1秒以上続いた区間を無音とみなし、発話の前後を250 msずつ残します。生活音まで残る場合は、しきい値を上げてください。

```sh
# より積極的に静かな生活音を削る
uv run python remove_silence.py input.m4a --silence-threshold -35

# 長い間だけ削り、発話前後の余白を広めに残す
uv run python remove_silence.py input.m4a --min-silence 1500 --keep-silence 400
```

主なオプション:

| オプション | 既定値 | 説明 |
|---|---:|---|
| `--silence-threshold` | `-40` | 無音とみなす音量（dBFS）。高くするほど積極的に削除 |
| `--min-silence` | `1000` | 削除対象にする無音の最短時間（ms） |
| `--keep-silence` | `250` | 発話の前後に残す余白（ms） |
| `--min-speech` | `150` | 残す音の最短時間（ms）。短い物音の除外に利用 |
| `--fade` | `10` | 連結点のクロスフェード（ms） |

MP3・M4AなどWAV以外の読み書きにはffmpegが必要です。

## バズった英語投稿の収集 (fetch_viral_english_posts.py)

xAI Grok API の **X Search** (`x_search`) を使い、海外で伸びた英語投稿を収集します。X API の Bearer トークンは不要で、`XAI_API_KEY` のみで動作します。

### 必要な環境変数

- `XAI_API_KEY` — [xAI Console](https://console.x.ai/) で発行

`python/.env.example` を `python/.env` にコピーし、キーを設定してください。

### 実行例

```sh
# 汎用（直近7日・いいね2万相当を目安）
uv run python fetch_viral_english_posts.py fetch

# AI 界隈のバズ投稿を10件
uv run python fetch_viral_english_posts.py fetch -p ai -n 10

# スタートアップ界隈 + トピック指定
uv run python fetch_viral_english_posts.py fetch -p startup -t "YC" --days 14

# JSON / CSV に保存
uv run python fetch_viral_english_posts.py fetch -p media -o viral.json --csv viral.csv

# プリセット一覧
uv run python fetch_viral_english_posts.py list-presets
```

### プリセット

| preset | 内容 |
|--------|------|
| `general` | 英語・高エンゲージメント投稿（汎用） |
| `ai` | AI / GPT / agents |
| `startup` | startup / SaaS / founder |
| `media` | メディア付きの超バズ投稿 |
| `verified` | Verified アカウントの伸び投稿 |

### 主なオプション

| オプション | 説明 | 既定値 |
|-----------|------|--------|
| `--preset`, `-p` | 上記プリセット | `general` |
| `--topic`, `-t` | 追加キーワード | なし |
| `--count`, `-n` | 取得件数（最大30） | `10` |
| `--min-likes` | 目安の最低いいね数 | `20000` |
| `--days` | 直近 N 日を検索 | `7` |
| `--from-date` / `--to-date` | 日付範囲（YYYY-MM-DD） | なし |
| `--model` | xAI モデル | `grok-4.3` |
| `-o` / `--csv` | JSON / CSV 出力先 | なし |

> **補足**
> - `x_search` はセマンティック検索のため、`min_faves:` 演算子はプロンプトのガイドとして使います。いいね数・view 数はモデル推定値です。
> - **xAI API（`XAI_API_KEY`）は X への投稿・リポストはできません。** 検索・要約のみです。投稿には X Developer API + OAuth（例: `xurl`）が別途必要です。

```sh
uv add ruff
uv run ruff check
```

## 音声文字起こしツール

音声・動画ファイルをローカルで文字起こしするツールです。

### Qwen3-ASR / Faster-Whisper版 (transcribe_faster_whisper.py)

2つのエンジンを切り替えて使える音声文字起こしツールです（ファイル名は後方互換のため据え置き）。

| エンジン | 指定 | 特徴 |
|---------|------|------|
| Qwen3-ASR（MLX） | デフォルト（`--engine qwen3`） | Apple SiliconのGPUで動作し高速（35分の音声を約1.5分）。句読点付き。**Apple Silicon Mac専用** |
| faster-whisper | `--engine whisper` | CTranslate2ベース。MacではCPU実行のため遅い。単語タイムスタンプ・細かい字幕・翻訳が必要な場合に使用 |

`--engine` を省略して `--model large-v3` などWhisperのモデル名を指定した場合は、従来どおりfaster-whisperを使用します。

#### 基本的な使用方法

```sh
# 基本的な使用方法（Qwen3-ASR）
uv run python transcribe_faster_whisper.py input.mp3

# 固有名詞・専門用語をヒントとして与える（両エンジン共通）
uv run python transcribe_faster_whisper.py input.mp3 --hotwords "Kubernetes 岡田 議事録"

# faster-whisper でモデルと言語を指定
uv run python transcribe_faster_whisper.py input.mp3 --engine whisper --model large-v3 --language ja

# GPUを使用して高速処理
uv run python transcribe_faster_whisper.py input.wav --device cuda --compute_type float16

# 単語レベルのタイムスタンプ付きJSON出力
uv run python transcribe_faster_whisper.py input.mp4 --format json --word_timestamps --output result.json

# WebVTT字幕形式で出力
uv run python transcribe_faster_whisper.py input.mp3 --format vtt --output subtitles.vtt
```

文字起こしの前に音声レベル（ピーク・発話レベル・背景ノイズ・SN比・クリップ率）を解析し、
ノイズ除去の要否と強度を自動で決めます。SN比が10dB以上ならノイズ除去は行いません（ノイズ除去は音声も削り、認識精度を下げることがあるため）。
元音声のクリップや発話レベルの小ささも警告します。

Qwen3-ASRは約30秒単位で区切って出力するため、`--format srt/vtt` の字幕としては1行が長くなります。字幕用途には `--engine whisper` を推奨します。
以下のオプションはfaster-whisper専用で、Qwen3-ASRでは無視されます（指定すると警告を表示）:
`--beam_size`, `--temperature`, `--vad_*`, `--no_speech_threshold`, `--no_condition_on_previous_text`, `--word_timestamps`, `--device`, `--compute_type`, `--cpu_threads`, `--num_workers`

faster-whisperでVADが有効な場合、文字起こしを始める前に元の長さ・VAD後の見込み・削除秒数と削減率を表示します。削りすぎだと判断した場合は `Ctrl+C` で中断し、VADパラメータを調整して再実行できます。

#### ユースケース別の使用方法

##### 1. 音声が小さい・遠い場合

```sh
# デフォルトでノーマライズと小声向けVADを使用
uv run python transcribe_faster_whisper.py recording.m4a

# さらにVAD閾値を下げる（より敏感に）
uv run python transcribe_faster_whisper.py recording.m4a --vad_threshold 0.15
```

##### 2. ノイズが多い環境の場合

```sh
# 通常は指定不要（SN比からノイズ除去の要否を自動判定）
uv run python transcribe_faster_whisper.py noisy_audio.m4a

# 強制的にノイズ除去を有効化
uv run python transcribe_faster_whisper.py noisy_audio.m4a --denoise --noise_reduce_amount 0.5

# ノイズ除去の強度を調整（0.0-1.0、デフォルト0.8）
uv run python transcribe_faster_whisper.py noisy_audio.m4a --denoise --noise_reduce_amount 0.6
```

##### 3. 音声が小さくノイズも多い場合

```sh
# ノイズ除去 + ノーマライズ + VAD調整の組み合わせ
uv run python transcribe_faster_whisper.py difficult_audio.m4a \
  --denoise \
  --noise_reduce_amount 0.5 \
  --vad_threshold 0.15
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
| `--normalize` / `--no_normalize` | 音声をピークノーマライズ（-3.0 dBFS） | 有効 | 無効化したい場合のみ `--no_normalize` |
| `--gain` | ゲイン調整(dB)。正の値で音量アップ | 0 | 10-15 dB |
| `--denoise` / `--no_denoise` | ノイズ除去を強制的に有効化 / 無効化 | 自動判定（SN比10dB未満で強度0.5、5dB未満で0.8） | 通常は指定不要 |
| `--noise_reduce_amount` | `--denoise` 指定時のノイズ除去の強度（0.0-1.0） | 0.8 | 0.5-0.6（強すぎると音声劣化） |

##### 認識精度オプション

| オプション | 説明 | 対応エンジン |
|-----------|------|------------|
| `--engine` | `qwen3`（デフォルト）または `whisper` | - |
| `--model` | qwen3: Hugging FaceのモデルID（デフォルト `Qwen/Qwen3-ASR-1.7B`）、whisper: `large-v3` など（デフォルト `large-v3`） | 両方 |
| `--hotwords` | 固有名詞・専門用語をスペース区切りで指定 | 両方 |
| `--initial_prompt` | 文脈を与える初期プロンプト | 両方 |
| `--no_condition_on_previous_text` | 直前の認識結果を文脈に使わない。同じ文の繰り返しが出る場合に有効 | whisper |

##### VAD（Voice Activity Detection）オプション

| オプション | 説明 | デフォルト | 推奨値 |
|-----------|------|-----------|--------|
| `--vad_threshold` | VAD閾値（0-1）。低いほど敏感 | 自動選択 | 通常は指定不要 |
| `--no_vad_auto_threshold` | 閾値の自動選択を無効化（0.20を使用） | - | 事前スキャンを省きたい場合 |
| `--vad_min_speech_duration` | 最小音声長(ms) | 50 | 30-100 |
| `--vad_min_silence_duration` | 最小無音長(ms) | 800 | 600-1200 |
| `--vad_speech_pad` | 音声の前後パディング(ms) | 800 | 800-1200 |
| `--no_vad_filter` | VADフィルタを無効化 | - | デバッグ時のみ |

`--vad_threshold` を指定しない場合、文字起こしの前に軽量なVADスキャン（54分の音声で30秒程度）を
複数の閾値で実行し、取りこぼしがない範囲で最も無音を削れる閾値を自動選択します。
これにより、小さい発話が無音扱いで欠落する問題と、無音区間を全て処理する非効率の両方を回避します。

##### 出力オプション

| オプション | 説明 |
|-----------|------|
| `--format` | 出力形式（text/json/srt/vtt） |
| `--timestamps` | タイムスタンプを含める |
| `--word_timestamps` | 単語レベルのタイムスタンプ |
| `--confidence` | 信頼度情報を含める |

#### パラメータ調整のコツ

1. **ノーマライズはデフォルトで有効**
   - 音声が小さい場合の基本対策として自動適用
   - ノイズも増幅されるので注意

2. **ノイズが多い場合は`--denoise`**
   - `--noise_reduce_amount 0.5`程度から始める
   - 強すぎると音声の質が劣化するので注意

3. **VAD閾値の調整**
   - 通常は事前スキャンによる自動選択に任せる（指定不要）
   - 手動で調整する場合、削減されすぎなら`--vad_threshold 0.05`、ノイズを拾いすぎなら`--vad_threshold 0.35`

4. **処理の順序**
   - ノイズ除去 → ゲイン調整 → ノーマライズ → VAD
   - この順序で自動的に処理されます

5. **句読点・相槌を残したい場合は`--initial_prompt`**
   - Whisperはデフォルトでは句読点を打たず、相槌や言い淀みを省略しがち
   - 句読点付きの例文をプロンプトに与えると出力スタイルが誘導され、文字数が2割程度増える

   ```bash
   uv run python transcribe_faster_whisper.py recording.m4a \
     --initial_prompt "うん、今日はですね、ちょっと雑談っぽく話します。まあ、あの、そういう感じで。"
   ```

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
     --vad_threshold 0.15 \
     --vad_speech_pad 1000 \
     --no_speech_threshold 0.95
   ```

2. **VADを完全に無効化**
   ```sh
   uv run python transcribe_faster_whisper.py recording.m4a \
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

- Qwen3-ASR（Apple SiliconのGPU）による高速・句読点付きの文字起こし（デフォルト）
- faster-whisper（`--engine whisper`）への切り替え
- メモリ効率的な処理
- 単語レベルのタイムスタンプ生成
- Voice Activity Detection (VAD) フィルター（小さい音声に最適化済み）
- 音声前処理機能（ノーマライズ、ゲイン調整、ノイズ除去）
- テキスト、JSON、SRT、WebVTT形式での出力
- 信頼度情報の出力オプション

### インストール

音声文字起こしツールを使用する場合は、以下のコマンドで依存関係をインストールしてください：

```sh
uv sync
```

## PDF OCR (pdfocr)

PDFをローカルだけでテキスト/Markdownにするツールです。クラウドAPIは使いません。
ページごとにテキスト層の品質を判定し、使えるページはそのまま抽出、スキャン画像や文字化けしたページだけをOCRします。

| エンジン | 指定 | 特徴 |
|---------|------|------|
| NDLOCR-Lite | デフォルト（`--engine ndl`） | 国立国会図書館のOCR。縦書きの和書に強く、GPUなしで高速。ルビ・柱・ノンブルは本文と別の領域として検出し、文字認識の対象から外す設計。CC BY 4.0 |
| YomiToku | `--engine yomitoku` | 読み順の推定や表に強い。ルビは除去する。Apple SiliconではGPU（mps）を使用。個人利用は無料（商用は有料、CC BY-NC-SA 4.0） |

どちらも依存パッケージのバージョンがこのプロジェクトと衝突するため、`~/.cache/` 配下の専用の仮想環境にインストールして呼び出します。

### セットアップ（初回のみ）

```sh
uv run python -m pdfocr setup ndl        # NDLOCR-Lite（約700MB）
uv run python -m pdfocr setup yomitoku   # YomiToku を使う場合（PyTorchを含むため数GB）
```

### 使用方法

```sh
# テキストとして保存（book.txt）
uv run python -m pdfocr book.pdf

# Markdownとして保存（章・節の見出し、原書の著作権表記を整形）
uv run python -m pdfocr book.pdf --format md

# YomiToku で全ページOCR
uv run python -m pdfocr scan.pdf --engine yomitoku --ocr always

# 既存の出力を上書き
uv run python -m pdfocr book.pdf --force
```

### 主なオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--format` | `txt` または `md` | `txt` |
| `--engine` | `ndl` または `yomitoku` | `ndl` |
| `--ocr` | `auto`（テキスト層が使えないページだけOCR）/ `always`（全ページOCR）/ `never`（テキスト層のみ） | `auto` |
| `--dpi` | OCR時にページを画像化する解像度 | 300 |
| `--keep-margins` | テキスト層の抽出で、ページ上下の帯（上12%・下13%）にある柱とページ番号を除外しない | 除外する |
| `-o`, `--output` | 出力先 | PDFと同じ場所 |
| `--force` | 既存の出力を上書き | 上書きしない |

### 処理の内容

- **テキスト層の判定**: 文字がほとんどない、`(cid:NNN)` や `U+FFFD` が多い、想定外の文字（文字化け）が多いページはOCRに回します。
- **柱とページ番号の除去（テキスト層）**: ページ上下の帯に完全に収まっていて、数字だけのもの（ページ番号）か、2ページ以上に繰り返し現れるもの（柱）だけを除きます。帯から始まる縦書きの列や、一度だけ現れる章見出しは残ります。
- **共通の後処理**: 連続する重複ページの除去、縦書き用の約物（﹁﹂︑︒ など）の変換、日本語の文字間に入った半角スペースの除去。
- **安全な保存**: 一時ファイルに書いてから置き換えるため、途中で失敗しても既存の出力は壊れません。入力PDF自体やそのハードリンクには出力できません。テキストが全く抽出できなかった場合は保存せず、終了コード1で終了します。

## Google Text-to-Speech 変換 (google_tts.py)

Google Cloud Text-to-Speech APIで日本語テキストを音声ファイルに変換するスクリプトです。

### 事前準備

1. Google CloudプロジェクトでText-to-Speech APIを有効化
2. サービスアカウント鍵(JSON)を取得し、`GOOGLE_APPLICATION_CREDENTIALS` 環境変数に設定するか、実行時に `--credentials` で指定
3. 依存関係を同期

```sh
uv sync
```

### 実行例

```sh
# 直接文字列を読み上げてMP3を生成
uv run python google_tts.py --text "本日は晴天なり。" --output demo.mp3

# テキストファイルを読み上げ、話速とピッチを調整
uv run python google_tts.py --text-file scripts/sample.txt --rate 1.2 --pitch 2.0 --format wav

# サービスアカウント鍵を指定して実行
uv run python google_tts.py --text "こんにちは" --credentials ~/.config/gcloud/tts-key.json
```

### 主なオプション

| オプション | 説明 | 既定値 |
|-------------|------|--------|
| `--voice` | 使用するボイス名 (例: `ja-JP-Neural2-C`) | `ja-JP-Neural2-C` |
| `--format` | 出力形式 (`mp3`, `wav`, `ogg`) | `mp3` |
| `--rate` | 話速 (0.25〜4.0) | `1.0` |
| `--pitch` | ピッチ調整 (-20〜20推奨) | `0.0` |
| `--sample-rate` | サンプルレートをHzで指定 | なし |

## 音声ファイル結合 (merge_audio.py)

複数の音声ファイルを無音を挟んで結合するツールです。

### 使用例

```sh
# slide1.mp3〜slide9.mp3を2秒の無音を挟んで結合
uv run python merge_audio.py slide*.mp3 -o combined.mp3

# 無音時間を3秒に変更
uv run python merge_audio.py slide*.mp3 -o combined.mp3 --silence 3.0

# ソートを無効化（引数の順序で結合）
uv run python merge_audio.py slide1.mp3 slide3.mp3 slide2.mp3 -o output.mp3 --no-sort
```

### 主なオプション

| オプション | 説明 | 既定値 |
|-------------|------|--------|
| `files` | 結合する音声ファイル（複数指定可、glob対応） | 必須 |
| `--output, -o` | 出力ファイルパス | 必須 |
| `--silence, -s` | ファイル間に挿入する無音の長さ（秒） | `2.0` |
| `--no-sort` | 自然順ソートを無効化 | `False` |

### 特徴

- pydubによる複数形式対応（mp3, wav, ogg, m4a, flac）
- natsortによる自然順ソート（slide1, slide2, ..., slide10の順序を保証）
- 無音時間を柔軟に設定可能（0〜60秒）

## 音声変換（Voice Conversion）

Seed-VC を使用した zero-shot 音声変換ツール `voice_convert.py` です。短い参照音声（1〜30秒）から声質を借りて、入力音声を別の声に変換します。学習不要で動作します。

セットアップ時に Seed-VC 専用の仮想環境（`~/.cache/seed-vc/.venv`）が自動作成されます。プロジェクト本体の venv とは依存関係を分離しているため、互いのパッケージバージョンが競合しません。

### セットアップ

```sh
# Seed-VC のクローン + 専用 venv 作成 + 依存関係インストール
uv run python voice_convert.py setup

# インストール先を指定する場合
uv run python voice_convert.py setup --path /path/to/seed-vc

# 既存のインストールを更新（git pull + 依存関係再インストール）
uv run python voice_convert.py setup --update
```

### 実行例

```sh
# 単一ファイルの音声変換
uv run python voice_convert.py convert input.wav reference.wav -o output.wav

# m4a などの形式もそのまま指定可能（自動で WAV に変換されます）
uv run python voice_convert.py convert input.m4a reference.wav -o output.wav

# 前処理付き（ノイズ除去 + ノーマライズ）
uv run python voice_convert.py convert input.wav reference.wav --denoise --normalize

# バッチ処理（ディレクトリ内の全音声ファイルを変換）
uv run python voice_convert.py convert ./inputs/ reference.wav -o ./outputs/

# CUDA GPU で高速・高品質に処理
uv run python voice_convert.py convert input.wav reference.wav --device cuda --diffusion-steps 50 --fp16

# Seed-VC 付属のデモ参照音声を使って試す
uv run python voice_convert.py convert input.wav ~/.cache/seed-vc/examples/reference/s1p1.wav -o output.wav
```

### 主要パラメーター

| パラメーター | 説明 | デフォルト |
|-----------|------|-----------|
| `source` | 変換元の音声ファイル（またはディレクトリ） | 必須 |
| `reference` | 参照音声ファイル（1〜30秒推奨） | 必須 |
| `--output, -o` | 出力ファイルパス | `{source_stem}_converted.wav` |
| `--seed-vc-path` | Seed-VC のインストールパス（環境変数 `SEED_VC_PATH` も可） | `~/.cache/seed-vc` |
| `--device` | 使用デバイス (auto/cuda/cpu) | `auto` |
| `--diffusion-steps` | 拡散ステップ数（大きいほど高品質・低速） | `25` |
| `--length-adjust` | 長さ調整係数（<1.0: 早口、>1.0: ゆっくり） | `1.0` |
| `--inference-cfg-rate` | 推論 CFG レート | `0.7` |
| `--fp16` | FP16 精度を使用（CUDA 推奨） | `False` |
| `--normalize` | 音声をノーマライズ | `False` |
| `--denoise` | ノイズ除去を適用 | `False` |
| `--gain` | ゲイン調整 (dB) | `0.0` |

### デバイスについて

- **CUDA GPU**: もっとも高速。`--fp16` との併用を推奨
- **CPU**: GPU がない環境ではこちらが使われます。処理時間は音声の長さに依存しますが、数分程度かかることがあります
- **MPS (Apple Silicon)**: Seed-VC が使用する PyTorch 2.4 の `torch.autocast` が MPS 未対応のため、自動的に CPU にフォールバックします

### 参照音声のコツ

- **長さ**: 10〜30秒が最適。1秒未満や30秒超は警告が表示されます
- **品質**: 単一話者、BGM やノイズが少ないクリーンな音声が理想的
- **形式**: WAV 推奨（mp3, m4a, flac 等も自動変換されます）
- **デモ音声**: セットアップ後、`~/.cache/seed-vc/examples/reference/` にサンプル音声が含まれています

## OpenVoice による音声 to 音声変換

`open_voice_converter.py` は、既存の読み上げ音声の抑揚や間をできるだけ保ったまま、参照音声の声質へ変換するための CLI です。  
TTS のアクセントが不自然なケースで、先に人間が読んだ音声や別エンジンで生成した音声を用意し、最後に声質だけを差し替える検証に向いています。

### 前提

- OpenVoice 本体を clone して依存関係を入れる
- OpenVoice V2 の `checkpoints_v2/converter` を配置する
- 必要ファイルは `config.json` と `checkpoint.pth`
- 参照音声は OpenVoice の `resources/*.mp3` も利用可能

### セットアップ

```sh
cd python
uv run python open_voice_converter.py setup
```

OpenVoice V2 の公式 checkpoint は別配布です。`setup` 後に `checkpoints_v2_0417.zip` を取得し、`converter` 配下のファイルを以下へ配置してください。

```sh
~/.cache/openvoice/checkpoints_v2/converter/config.json
~/.cache/openvoice/checkpoints_v2/converter/checkpoint.pth
```

公式案内:

- Usage: https://github.com/myshell-ai/OpenVoice/blob/main/docs/USAGE.md
- Checkpoint: https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip

### 参照音声の取得例

```sh
cd python
curl -L \
  https://raw.githubusercontent.com/myshell-ai/OpenVoice/main/resources/example_reference.mp3 \
  -o reference.mp3
```

### 基本的な使い方

```sh
cd python
uv run python open_voice_converter.py convert \
  ./input.m4a \
  ./reference.mp3 \
  -o ./output_openvoice.wav
```

### 主なオプション

```sh
uv run python open_voice_converter.py convert \
  ./input.m4a \
  ./reference.mp3 \
  -o ./output_openvoice.wav \
  --tau 0.25 \
  --gain 4 \
  --device cpu
```

- `reference_*` は目標の声質サンプルです。1話者・数秒から十数秒程度を推奨します。
- `--tau` は声質変換の強さです。低いほど元音声の質感を残しやすくなります。
- `--use-vad` は話者埋め込み抽出時に音声区間だけを拾いやすくします。
- CPU 実行は時間がかかるため、まず `--tau 0.2` / `0.25` / `0.35` で比較すると調整しやすいです。

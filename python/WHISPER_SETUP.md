# Whisper音声文字起こしツールのセットアップガイド

## 問題と解決策

現在、openai-whisperとその依存関係（特にnumba）がPython 3.10以降をサポートしていないため、通常のインストールができません。

## 解決策

### オプション1: Python 3.9を使用する（推奨）

1. Python 3.9をインストール:
```bash
uv python install 3.9
```

2. Python 3.9用の仮想環境を作成:
```bash
uv venv --python 3.9
source .venv/bin/activate
```

3. 依存関係を個別にインストール:
```bash
# Faster-Whisperのみをインストール（Python 3.10+対応）
pip install faster-whisper pydub tqdm

# または、OpenAI Whisperのみをインストール（Python 3.9必須）
pip install openai-whisper pydub tqdm
```

### オプション2: Dockerを使用する

Dockerを使用すれば、環境の互換性問題を回避できます：

```dockerfile
# Dockerfile.whisper
FROM python:3.9-slim

WORKDIR /app

RUN pip install openai-whisper faster-whisper pydub tqdm

COPY transcribe_*.py .

ENTRYPOINT ["python"]
```

使用方法:
```bash
docker build -f Dockerfile.whisper -t whisper-transcribe .
docker run -v $(pwd):/data whisper-transcribe transcribe_whisper.py /data/input.mp3
```

### オプション3: Faster-Whisperのみを使用する（Python 3.10+対応）

Faster-WhisperはPython 3.10以降でも動作するため、これのみを使用：

```bash
# pyproject.tomlを編集して、openai-whisperを削除
# faster-whisperのみを残す

uv sync
```

### オプション4: pipenvやpoetryを使用

別のパッケージマネージャーを使用：

```bash
# pipenvの場合
pipenv --python 3.9
pipenv install openai-whisper faster-whisper pydub tqdm

# poetryの場合
poetry env use python3.9
poetry add openai-whisper faster-whisper pydub tqdm
```

## 推奨される使用方法

最も簡単な方法は、Faster-Whisperのみを使用することです：

1. pyproject.tomlからopenai-whisperを削除
2. Python 3.10以降で`uv sync`を実行
3. `transcribe_faster_whisper.py`を使用

Faster-Whisperは以下の利点があります：
- より高速な処理
- メモリ効率が良い
- Python 3.10以降をサポート
- OpenAI Whisperと同等の精度

## 手動インストール

必要に応じて、以下のコマンドで手動インストールも可能：

```bash
# Faster-Whisperのみ（Python 3.10+）
pip install faster-whisper pydub tqdm

# OpenAI Whisperのみ（Python 3.9）
pip install openai-whisper pydub tqdm
```
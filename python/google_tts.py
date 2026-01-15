#!/usr/bin/env python3
"""Google Cloud Text-to-Speechで日本語テキストを音声化するユーティリティ。"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Literal

import typer
from google.api_core.exceptions import GoogleAPICallError, InvalidArgument
from google.cloud import texttospeech

app = typer.Typer(add_completion=False, help="日本語テキストを音声ファイルに変換します。")

AudioFormat = Literal["mp3", "wav", "ogg"]
AUDIO_ENCODING_MAP: dict[AudioFormat, texttospeech.AudioEncoding] = {
    "mp3": texttospeech.AudioEncoding.MP3,
    "wav": texttospeech.AudioEncoding.LINEAR16,
    "ogg": texttospeech.AudioEncoding.OGG_OPUS,
}

# テキスト分割の区切り文字（優先度順）
SPLIT_DELIMITERS = [
    "\n",  # 改行（最優先）
    "。",  # 句点
    "！",
    "!",
    "？",
    "?",  # 感嘆符・疑問符
    "、",  # 読点
    "」",
    "）",
    "】",
    "』",  # 括弧閉じ
]


def find_split_point(text: str, max_bytes: int) -> int:
    """テキスト内で最適な分割ポイント（文字インデックス）を見つける。

    Args:
        text: 分割対象のテキスト
        max_bytes: 最大バイト数

    Returns:
        分割すべき文字位置（インデックス）。分割点が見つからない場合は -1
    """
    if len(text.encode("utf-8")) <= max_bytes:
        return len(text)

    # バイト数制限内で最大の文字位置を二分探索
    left, right = 0, len(text)
    while left < right:
        mid = (left + right + 1) // 2
        if len(text[:mid].encode("utf-8")) <= max_bytes:
            left = mid
        else:
            right = mid - 1

    max_char_index = left
    if max_char_index == 0:
        return -1

    # 区切り文字を優先度順に探索
    for delimiter in SPLIT_DELIMITERS:
        # max_char_index から逆方向に区切り文字を探す
        search_text = text[:max_char_index]
        last_pos = search_text.rfind(delimiter)
        if last_pos != -1:
            # 区切り文字の次の位置で分割（区切り文字を含める）
            return last_pos + 1

    # 区切り文字が見つからない場合は -1
    return -1


def split_text_for_tts(text: str, max_bytes: int = 4800) -> list[str]:
    """テキストを TTS API のバイト制限に収まるチャンクに分割する。

    Args:
        text: 分割対象のテキスト
        max_bytes: 1チャンクの最大バイト数（デフォルト: 4800）

    Returns:
        分割されたテキストのリスト

    Raises:
        ValueError: 単一の文が max_bytes を超え、分割できない場合
    """
    if not text:
        return []

    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining.encode("utf-8")) <= max_bytes:
            chunks.append(remaining)
            break

        split_pos = find_split_point(remaining, max_bytes)
        if split_pos == -1 or split_pos == 0:
            raise ValueError(
                f"テキストを分割できません。{max_bytes}バイト以内に区切り文字がありません。"
            )

        chunks.append(remaining[:split_pos])
        remaining = remaining[split_pos:]

    return chunks


def merge_audio_files(
    audio_paths: list[Path],
    output_path: Path,
    audio_format: AudioFormat,
) -> Path:
    """複数の音声ファイルを1つに結合する。

    Args:
        audio_paths: 結合する音声ファイルのパスリスト
        output_path: 出力ファイルパス
        audio_format: 出力形式

    Returns:
        結合後のファイルパス

    Raises:
        ValueError: 空のリストが渡された場合
    """
    from pydub import AudioSegment
    import shutil

    if not audio_paths:
        raise ValueError("音声ファイルが指定されていません")

    if len(audio_paths) == 1:
        shutil.copy(audio_paths[0], output_path)
        return output_path

    # 形式に応じた読み込み方法
    format_map = {
        "mp3": "mp3",
        "wav": "wav",
        "ogg": "ogg",
    }
    pydub_format = format_map.get(audio_format, "mp3")

    # 最初のファイルを読み込み
    combined = AudioSegment.from_file(str(audio_paths[0]), format=pydub_format)

    # 残りのファイルを結合
    for path in audio_paths[1:]:
        segment = AudioSegment.from_file(str(path), format=pydub_format)
        combined += segment

    # 出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), format=pydub_format)

    return output_path


def _load_text(text: Optional[str], text_file: Optional[Path]) -> str:
    if text and text_file:
        raise typer.BadParameter("--text と --text-file は同時に指定できません。")
    if text:
        return text
    if text_file:
        if not text_file.exists():
            raise typer.BadParameter(f"テキストファイルが見つかりません: {text_file}")
        return text_file.read_text(encoding="utf-8").strip()
    raise typer.BadParameter("--text か --text-file のいずれかを指定してください。")


def _ensure_credentials(credentials_path: Optional[Path]) -> None:
    if credentials_path:
        if not credentials_path.exists():
            raise typer.BadParameter(f"認証ファイルが見つかりません: {credentials_path}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        typer.echo(
            "警告: GOOGLE_APPLICATION_CREDENTIALS が未設定です。\n"
            "gcloud CLI で認証済み、もしくは --credentials でサービスアカウント鍵を指定してください。"
        )


def synthesize_speech(
    text: str,
    output_path: Path,
    *,
    language_code: str = "ja-JP",
    voice_name: str = "ja-JP-Neural2-C",
    audio_format: AudioFormat = "mp3",
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    sample_rate_hz: Optional[int] = None,
) -> Path:
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
    )

    audio_kwargs = {
        "audio_encoding": AUDIO_ENCODING_MAP[audio_format],
        "speaking_rate": speaking_rate,
        "pitch": pitch,
    }
    if sample_rate_hz:
        audio_kwargs["sample_rate_hertz"] = sample_rate_hz

    audio_config = texttospeech.AudioConfig(**audio_kwargs)

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.audio_content)
    return output_path


def synthesize_long_speech(
    text: str,
    output_path: Path,
    *,
    language_code: str = "ja-JP",
    voice_name: str = "ja-JP-Neural2-C",
    audio_format: AudioFormat = "mp3",
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    sample_rate_hz: Optional[int] = None,
    show_progress: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Path:
    """長いテキストを分割して音声合成し、結合する。

    Args:
        text: 合成するテキスト
        output_path: 出力ファイルパス
        language_code: 言語コード
        voice_name: 音声名
        audio_format: 出力形式
        speaking_rate: 話速
        pitch: ピッチ
        sample_rate_hz: サンプルレート
        show_progress: 進捗表示を行うか
        max_retries: API 失敗時のリトライ回数
        retry_delay: リトライ間隔（秒）

    Returns:
        出力ファイルパス
    """
    # テキストを分割
    chunks = split_text_for_tts(text, max_bytes=4800)

    if len(chunks) == 1:
        # 分割不要な場合は直接処理
        return synthesize_speech(
            text=chunks[0],
            output_path=output_path,
            language_code=language_code,
            voice_name=voice_name,
            audio_format=audio_format,
            speaking_rate=speaking_rate,
            pitch=pitch,
            sample_rate_hz=sample_rate_hz,
        )

    # 一時ディレクトリで分割音声を生成
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        audio_paths: list[Path] = []

        for i, chunk in enumerate(chunks):
            if show_progress:
                typer.echo(f"音声合成中: {i + 1}/{len(chunks)} チャンク")

            chunk_path = tmp_path / f"chunk_{i:04d}.{audio_format}"

            # リトライ付きで API 呼び出し
            for attempt in range(max_retries):
                try:
                    synthesize_speech(
                        text=chunk,
                        output_path=chunk_path,
                        language_code=language_code,
                        voice_name=voice_name,
                        audio_format=audio_format,
                        speaking_rate=speaking_rate,
                        pitch=pitch,
                        sample_rate_hz=sample_rate_hz,
                    )
                    break
                except (GoogleAPICallError, InvalidArgument) as e:
                    if attempt == max_retries - 1:
                        raise
                    typer.echo(
                        f"API エラー (試行 {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(retry_delay * (2**attempt))

            audio_paths.append(chunk_path)

        # 音声ファイルを結合
        if show_progress:
            typer.echo("音声ファイルを結合中...")

        merge_audio_files(audio_paths, output_path, audio_format)

    return output_path


@app.command()
def run(
    text: Optional[str] = typer.Option(None, "--text", help="直接文字列を指定"),
    text_file: Optional[Path] = typer.Option(None, "--text-file", exists=False, help="テキストファイルを指定"),
    output: Path = typer.Option(Path("tts_output.mp3"), "--output", help="出力ファイルパス"),
    voice_name: str = typer.Option("ja-JP-Neural2-C", "--voice", help="Google TTS のボイス名"),
    language_code: str = typer.Option("ja-JP", "--language", help="言語コード"),
    audio_format: AudioFormat = typer.Option("mp3", "--format", case_sensitive=False, help="出力形式 (mp3/wav/ogg)"),
    speaking_rate: float = typer.Option(1.0, "--rate", min=0.25, max=4.0, help="話速 (0.25-4.0)"),
    pitch: float = typer.Option(0.0, "--pitch", help="ピッチ調整 (-20.0 から 20.0 推奨)"),
    sample_rate_hz: Optional[int] = typer.Option(None, "--sample-rate", help="サンプルレート (Hz)"),
    credentials_path: Optional[Path] = typer.Option(None, "--credentials", help="サービスアカウントJSONのパス"),
    max_retries: int = typer.Option(3, "--max-retries", help="API失敗時のリトライ回数"),
    no_progress: bool = typer.Option(False, "--no-progress", help="進捗表示を無効化"),
) -> None:
    """Google Text-to-Speech で日本語の音声ファイルを生成します。"""
    _ensure_credentials(credentials_path)
    text_content = _load_text(text, text_file)

    normalized_format = audio_format.lower()
    if normalized_format not in AUDIO_ENCODING_MAP:
        raise typer.BadParameter("--format は mp3 / wav / ogg から選択してください。")

    if output.suffix.lower() != f".{normalized_format}":
        output = output.with_suffix(f".{normalized_format}")

    # バイト数チェックして適切な関数を呼び出す
    text_bytes = len(text_content.encode("utf-8"))

    try:
        if text_bytes <= 4800:
            # 短いテキストは直接処理
            result_path = synthesize_speech(
                text=text_content,
                output_path=output,
                language_code=language_code,
                voice_name=voice_name,
                audio_format=normalized_format,  # type: ignore[arg-type]
                speaking_rate=speaking_rate,
                pitch=pitch,
                sample_rate_hz=sample_rate_hz,
            )
        else:
            # 長いテキストは分割処理
            typer.echo(f"テキストが {text_bytes} バイトのため、分割処理を行います。")
            result_path = synthesize_long_speech(
                text=text_content,
                output_path=output,
                language_code=language_code,
                voice_name=voice_name,
                audio_format=normalized_format,  # type: ignore[arg-type]
                speaking_rate=speaking_rate,
                pitch=pitch,
                sample_rate_hz=sample_rate_hz,
                show_progress=not no_progress,
                max_retries=max_retries,
            )
    except (GoogleAPICallError, InvalidArgument) as error:
        typer.secho(f"APIエラーが発生しました: {error}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from error

    typer.secho(f"音声ファイルを出力しました: {result_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()

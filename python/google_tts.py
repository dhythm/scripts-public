#!/usr/bin/env python3
"""Google Cloud Text-to-Speechで日本語テキストを音声化するユーティリティ。"""

from __future__ import annotations

import os
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
) -> None:
    """Google Text-to-Speech で日本語の音声ファイルを生成します。"""
    _ensure_credentials(credentials_path)
    text_content = _load_text(text, text_file)

    normalized_format = audio_format.lower()
    if normalized_format not in AUDIO_ENCODING_MAP:
        raise typer.BadParameter("--format は mp3 / wav / ogg から選択してください。")

    if output.suffix.lower() != f".{normalized_format}":
        output = output.with_suffix(f".{normalized_format}")

    try:
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
    except (GoogleAPICallError, InvalidArgument) as error:
        typer.secho(f"APIエラーが発生しました: {error}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from error

    typer.secho(f"音声ファイルを出力しました: {result_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()

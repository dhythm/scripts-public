#!/usr/bin/env python3
"""複数の音声ファイルを無音を挟んで結合するユーティリティ。"""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer
from natsort import natsorted
from pydub import AudioSegment

app = typer.Typer(add_completion=False, help="複数の音声ファイルを無音を挟んで結合します。")


def merge_audio_with_silence(
    audio_paths: List[Path],
    output_path: Path,
    silence_duration_ms: int,
) -> Path:
    """複数の音声ファイルを無音を挟んで結合する。

    Args:
        audio_paths: 結合する音声ファイルのパスリスト
        output_path: 出力ファイルパス
        silence_duration_ms: ファイル間に挿入する無音の長さ（ミリ秒）

    Returns:
        結合後のファイルパス

    Raises:
        ValueError: 空のリストが渡された場合
        FileNotFoundError: 音声ファイルが見つからない場合
    """
    if not audio_paths:
        raise ValueError("音声ファイルが指定されていません")

    # ファイルの存在確認
    for path in audio_paths:
        if not path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")

    # 出力形式を拡張子から判定
    output_format = output_path.suffix.lstrip(".").lower()
    if output_format not in ("mp3", "wav", "ogg", "m4a", "flac"):
        output_format = "mp3"

    # 無音セグメントを作成
    silence = AudioSegment.silent(duration=silence_duration_ms)

    # 最初のファイルを読み込み
    combined = AudioSegment.from_file(str(audio_paths[0]))
    typer.echo(f"  [1/{len(audio_paths)}] {audio_paths[0].name}")

    # 残りのファイルを無音を挟んで結合
    for i, path in enumerate(audio_paths[1:], start=2):
        typer.echo(f"  [{i}/{len(audio_paths)}] {path.name}")
        segment = AudioSegment.from_file(str(path))
        combined = combined + silence + segment

    # 出力ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 出力
    combined.export(str(output_path), format=output_format)

    return output_path


@app.command()
def run(
    files: List[Path] = typer.Argument(
        ...,
        help="結合する音声ファイル（複数指定可）",
        exists=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="出力ファイルパス",
    ),
    silence: float = typer.Option(
        2.0,
        "--silence",
        "-s",
        help="ファイル間に挿入する無音の長さ（秒）",
        min=0.0,
        max=60.0,
    ),
    no_sort: bool = typer.Option(
        False,
        "--no-sort",
        help="自然順ソートを無効化（引数の順序で結合）",
    ),
) -> None:
    """複数の音声ファイルを無音を挟んで結合します。

    例:
        python merge_audio.py slide1.mp3 slide2.mp3 slide3.mp3 -o output.mp3

        python merge_audio.py slide*.mp3 -o output.mp3 --silence 3.0
    """
    if len(files) < 2:
        typer.secho("エラー: 2つ以上のファイルを指定してください。", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # ソート
    if no_sort:
        sorted_files = list(files)
    else:
        sorted_files = natsorted(files, key=lambda p: str(p))

    typer.echo(f"結合するファイル数: {len(sorted_files)}")
    typer.echo(f"無音時間: {silence}秒")
    typer.echo("")

    silence_ms = int(silence * 1000)

    try:
        result_path = merge_audio_with_silence(
            audio_paths=sorted_files,
            output_path=output,
            silence_duration_ms=silence_ms,
        )
        typer.echo("")
        typer.secho(f"出力完了: {result_path}", fg=typer.colors.GREEN)
    except (ValueError, FileNotFoundError) as e:
        typer.secho(f"エラー: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.secho(f"予期しないエラー: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()

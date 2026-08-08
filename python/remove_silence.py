#!/usr/bin/env python3
"""音声ファイルから無音区間を削除し、発話部分だけを連結する。"""

from __future__ import annotations

from pathlib import Path

import typer
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


app = typer.Typer(
    add_completion=False,
    help="音声の無音区間を削除し、発話部分だけの音声ファイルを作成します。",
)

SUPPORTED_OUTPUTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def output_format(path: Path) -> str:
    """出力パスから pydub/ffmpeg 用の形式名を返す。"""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_OUTPUTS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUTS))
        raise ValueError(f"未対応の出力形式です: {suffix or '(拡張子なし)'}（対応: {supported}）")
    return {".m4a": "mp4", ".aac": "adts"}.get(suffix, suffix.lstrip("."))


def find_speech_ranges(
    audio: AudioSegment,
    *,
    min_silence_ms: int,
    silence_threshold_dbfs: float,
    keep_silence_ms: int,
    min_speech_ms: int,
    seek_step_ms: int = 10,
) -> list[tuple[int, int]]:
    """非無音区間を検出し、余白を付けた範囲をミリ秒単位で返す。"""
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_threshold_dbfs,
        seek_step=seek_step_ms,
    )

    # 短い物音を除外してから余白を付ける。
    padded = [
        (max(0, start - keep_silence_ms), min(len(audio), end + keep_silence_ms))
        for start, end in ranges
        if end - start >= min_speech_ms
    ]
    if not padded:
        return []

    # 余白が重なった区間は統合し、同じ音を二重に連結しない。
    merged = [padded[0]]
    for start, end in padded[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def join_ranges(
    audio: AudioSegment,
    ranges: list[tuple[int, int]],
    *,
    fade_ms: int,
) -> AudioSegment:
    """指定区間を連結する。切断点には短いクロスフェードを適用する。"""
    if not ranges:
        raise ValueError("発話区間が見つかりませんでした")

    result = audio[ranges[0][0] : ranges[0][1]]
    for start, end in ranges[1:]:
        segment = audio[start:end]
        crossfade = min(fade_ms, len(result), len(segment))
        result = result.append(segment, crossfade=crossfade)
    return result


def default_output_path(input_path: Path) -> Path:
    """入力ファイルと同じディレクトリに既定の出力パスを作る。"""
    return input_path.with_name(f"{input_path.stem}_speech{input_path.suffix}")


@app.command()
def run(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="入力音声ファイル（wav/mp3/m4a/flac等）",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="出力先。省略時は <元の名前>_speech.<拡張子>",
    ),
    silence_threshold: float = typer.Option(
        -40.0,
        "--silence-threshold",
        help="この音量以下を無音とみなす値（dBFS）。生活音も切るなら値を上げます",
        min=-100.0,
        max=0.0,
    ),
    min_silence: int = typer.Option(
        1000,
        "--min-silence",
        help="削除対象にする無音の最短時間（ms）",
        min=50,
    ),
    keep_silence: int = typer.Option(
        250,
        "--keep-silence",
        help="各発話の前後に残す余白（ms）",
        min=0,
    ),
    min_speech: int = typer.Option(
        150,
        "--min-speech",
        help="残す音の最短時間（ms）。短い生活音の除外に使用します",
        min=0,
    ),
    fade: int = typer.Option(
        10,
        "--fade",
        help="連結点のクロスフェード時間（ms）",
        min=0,
        max=100,
    ),
    list_segments: bool = typer.Option(
        False,
        "--list-segments",
        help="検出した発話区間を表示します",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="検出結果だけを表示し、ファイルを出力しません",
    ),
) -> None:
    """無音を削り、発話区間のみをつないだ音声ファイルを作成します。"""
    output_path = output or default_output_path(input_path)

    try:
        if input_path.resolve() == output_path.resolve():
            raise ValueError("入力ファイルと出力ファイルには別のパスを指定してください")
        format_name = output_format(output_path)

        typer.echo(f"読み込み中: {input_path}")
        audio = AudioSegment.from_file(str(input_path))
        ranges = find_speech_ranges(
            audio,
            min_silence_ms=min_silence,
            silence_threshold_dbfs=silence_threshold,
            keep_silence_ms=keep_silence,
            min_speech_ms=min_speech,
        )
        result = join_ranges(audio, ranges, fade_ms=fade)

        if list_segments:
            for index, (start, end) in enumerate(ranges, start=1):
                typer.echo(f"  {index:>3}: {start / 1000:.3f} - {end / 1000:.3f} 秒")

        original_seconds = len(audio) / 1000
        result_seconds = len(result) / 1000
        removed_seconds = original_seconds - result_seconds
        removed_ratio = removed_seconds / original_seconds * 100 if original_seconds else 0

        typer.echo(f"検出した発話区間: {len(ranges)} 個")
        typer.echo(f"処理前: {original_seconds:.2f} 秒")
        typer.echo(f"処理後: {result_seconds:.2f} 秒")
        typer.echo(f"削減:   {removed_seconds:.2f} 秒 ({removed_ratio:.1f}%)")

        if dry_run:
            typer.echo("dry-run のためファイルは出力していません。")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.export(str(output_path), format=format_name)
        typer.secho(f"出力完了: {output_path}", fg=typer.colors.GREEN)
    except (ValueError, OSError) as exc:
        typer.secho(f"エラー: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.secho(
            f"音声処理に失敗しました: {exc}\nffmpegがインストールされているか確認してください。",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()

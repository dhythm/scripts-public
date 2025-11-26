#!/usr/bin/env python3
"""
Kotoba-Whisper v2.0による高性能日本語文字起こしツール

特徴:
- 日本語特化モデル（Whisper large-v3と同等精度、6.3倍高速）
- Demucsによる環境ノイズ除去（風切り音等に効果的）
- 音声・動画ファイル両対応
- VADによるハルシネーション防止

使用方法:
    python transcribe_kotoba.py <ファイル> [オプション]

例:
    python transcribe_kotoba.py input.mp3
    python transcribe_kotoba.py video.mp4 --format srt
    python transcribe_kotoba.py input.wav --no-denoise --format json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel
from tqdm import tqdm

# --- 定数 ---
DEFAULT_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"
FALLBACK_MODEL = "large-v3"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv"}

# Whisper設定（CPU最適化・精度重視）
TRANSCRIBE_OPTS = {
    "language": "ja",
    "beam_size": 5,
    "best_of": 5,
    "patience": 2.0,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.5,
    "condition_on_previous_text": False,  # ハルシネーション防止
    "vad_filter": True,
    "vad_parameters": {
        "threshold": 0.3,
        "min_speech_duration_ms": 100,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 400,
    },
}


class TranscriptionError(Exception):
    """文字起こしエラー"""

    pass


# --- ファイル判定 ---
def is_video_file(path: Path) -> bool:
    """動画ファイルかどうかを判定"""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    """音声ファイルかどうかを判定"""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def validate_input_file(path: Path) -> None:
    """入力ファイルの検証"""
    if not path.exists():
        raise TranscriptionError(f"ファイルが見つかりません: {path}")

    if not path.is_file():
        raise TranscriptionError(f"パスがファイルではありません: {path}")

    if not (is_audio_file(path) or is_video_file(path)):
        supported = sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)
        raise TranscriptionError(
            f"サポートされていないファイル形式: {path.suffix}\n"
            f"サポートされている形式: {', '.join(supported)}"
        )


# --- 動画から音声抽出 ---
def extract_audio_from_video(video_path: Path, output_path: Path) -> Path:
    """
    ffmpegで動画から音声を抽出

    Args:
        video_path: 動画ファイルパス
        output_path: 出力WAVファイルパス

    Returns:
        出力ファイルパス
    """
    print(f"動画から音声を抽出中: {video_path.name}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",  # 映像なし
        "-acodec",
        "pcm_s16le",  # 16-bit PCM
        "-ar",
        "16000",  # 16kHz（Whisper推奨）
        "-ac",
        "1",  # モノラル
        "-af",
        "loudnorm",  # ラウドネス正規化
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise TranscriptionError(f"音声抽出に失敗しました: {e.stderr}")
    except FileNotFoundError:
        raise TranscriptionError(
            "ffmpegがインストールされていません。\n"
            "インストール方法: brew install ffmpeg"
        )

    print("音声抽出完了")
    return output_path


# --- Demucsによるノイズ除去 ---
def apply_demucs_denoise(audio_path: Path, output_dir: Path, verbose: bool = False) -> Path:
    """
    Demucsで音声（vocals）を抽出してノイズを除去

    Args:
        audio_path: 入力音声ファイルパス
        output_dir: 出力ディレクトリ
        verbose: 詳細ログ

    Returns:
        ノイズ除去後の音声ファイルパス
    """
    print("Demucsによるノイズ除去を実行中...")

    try:
        import numpy as np
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        from pydub import AudioSegment
    except ImportError as e:
        raise TranscriptionError(
            f"Demucsの依存関係が不足しています: {e}\n"
            "インストール: uv sync"
        )

    # モデルをロード（htdemucs: Hybrid Transformer Demucs）
    if verbose:
        print("Demucsモデルをロード中...")
    model = get_model("htdemucs")
    model.eval()

    # pydubで音声を読み込み（TorchCodecの問題を回避）
    if verbose:
        print(f"音声ファイルを読み込み中: {audio_path}")
    audio = AudioSegment.from_file(str(audio_path))
    sr = audio.frame_rate

    # numpy配列に変換
    samples = np.array(audio.get_array_of_samples())

    # ステレオの場合は適切に reshape
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).T  # (2, time)
    else:
        samples = samples.reshape((1, -1))  # (1, time)

    # float32に正規化 (-1.0 ~ 1.0)
    if audio.sample_width == 2:  # 16-bit
        samples = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 1:  # 8-bit
        samples = samples.astype(np.float32) / 128.0
    else:
        samples = samples.astype(np.float32)

    # PyTorchテンソルに変換
    wav = torch.from_numpy(samples)

    # リサンプリング（Demucsは44100Hzを期待）
    if sr != model.samplerate:
        if verbose:
            print(f"リサンプリング: {sr}Hz → {model.samplerate}Hz")
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)

    # バッチ次元を追加
    wav = wav.unsqueeze(0)  # (1, channels, time)

    # ステレオに拡張（Demucsは2chを期待）
    if wav.shape[1] == 1:
        wav = wav.repeat(1, 2, 1)

    # 音源分離を実行
    if verbose:
        print("音源分離を実行中...")
    with torch.no_grad():
        sources = apply_model(model, wav, device="cpu")

    # vocals (index=3) を抽出
    # sources shape: (batch, sources, channels, time)
    # sources: drums(0), bass(1), other(2), vocals(3)
    vocals = sources[0, 3]  # (channels, time)

    # モノラルに変換
    vocals = vocals.mean(0, keepdim=True)  # (1, time)

    # 16kHzにリサンプリング（Whisper用）
    import torchaudio
    vocals = torchaudio.functional.resample(vocals, model.samplerate, 16000)

    # numpy配列に戻す
    vocals_np = vocals.numpy()

    # 16-bit intに変換
    vocals_np = (vocals_np * 32768.0).clip(-32768, 32767).astype(np.int16)

    # AudioSegmentに変換
    output_audio = AudioSegment(
        vocals_np.tobytes(),
        frame_rate=16000,
        sample_width=2,
        channels=1,
    )

    # 出力
    output_path = output_dir / f"{audio_path.stem}_denoised.wav"
    output_audio.export(str(output_path), format="wav")

    print("ノイズ除去完了")
    return output_path


# --- 文字起こし ---
def transcribe_audio(
    audio_path: Path,
    model_id: str = DEFAULT_MODEL,
    word_timestamps: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Kotoba-Whisperで文字起こしを実行

    Args:
        audio_path: 音声ファイルパス
        model_id: モデルID
        word_timestamps: 単語レベルタイムスタンプ
        verbose: 詳細ログ

    Returns:
        文字起こし結果
    """
    print(f"モデルをロード中: {model_id}")

    # モデルをロード（CPU用にint8で最適化）
    model = WhisperModel(
        model_id,
        device="cpu",
        compute_type="int8",
    )

    print("文字起こしを開始...")

    # 文字起こし実行
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=word_timestamps,
        **TRANSCRIBE_OPTS,
    )

    # セグメントをリストに変換
    segment_list = []
    full_text = []

    for segment in tqdm(segments, desc="セグメント処理"):
        seg_dict = {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "no_speech_prob": segment.no_speech_prob,
            "avg_logprob": segment.avg_logprob,
        }

        if word_timestamps and segment.words:
            seg_dict["words"] = [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                    "probability": word.probability,
                }
                for word in segment.words
            ]

        segment_list.append(seg_dict)
        full_text.append(segment.text)

    print(f"検出言語: {info.language} (確信度: {info.language_probability:.2%})")

    return {
        "text": "".join(full_text),
        "segments": segment_list,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
        "model": model_id,
    }


# --- 出力フォーマット ---
def format_timestamp(seconds: float) -> str:
    """秒をHH:MM:SS.mmm形式に変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_timestamp_srt(seconds: float) -> str:
    """秒をSRT形式（HH:MM:SS,mmm）に変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """秒をWebVTT形式（HH:MM:SS.mmm）に変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_as_text(result: dict, timestamps: bool = False) -> str:
    """テキスト形式で出力"""
    if not timestamps:
        return result["text"].strip()

    lines = []
    for seg in result["segments"]:
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"[{start} --> {end}] {text}")

    return "\n".join(lines)


def format_as_json(result: dict, timestamps: bool = False) -> str:
    """JSON形式で出力"""
    output = {
        "text": result["text"],
        "language": result["language"],
        "language_probability": result["language_probability"],
        "duration": result["duration"],
        "model": result["model"],
    }

    if timestamps:
        output["segments"] = [
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "words": seg.get("words"),
            }
            for seg in result["segments"]
        ]

    return json.dumps(output, ensure_ascii=False, indent=2)


def format_as_srt(result: dict) -> str:
    """SRT字幕形式で出力"""
    lines = []
    for i, seg in enumerate(result["segments"], 1):
        start = format_timestamp_srt(seg["start"])
        end = format_timestamp_srt(seg["end"])
        text = seg["text"].strip()

        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def format_as_vtt(result: dict) -> str:
    """WebVTT字幕形式で出力"""
    lines = ["WEBVTT", ""]

    for seg in result["segments"]:
        start = format_timestamp_vtt(seg["start"])
        end = format_timestamp_vtt(seg["end"])
        text = seg["text"].strip()

        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def save_output(content: str, path: Path) -> None:
    """出力をファイルに保存"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"出力を保存しました: {path}")


# --- メイン処理 ---
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kotoba-Whisper v2.0による高性能日本語文字起こしツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用
  %(prog)s input.mp3

  # 動画ファイル
  %(prog)s video.mp4

  # SRT字幕出力
  %(prog)s input.mp3 --format srt --output subtitles.srt

  # ノイズ除去なし（高速モード）
  %(prog)s input.wav --no-denoise

  # タイムスタンプ付きJSON
  %(prog)s input.mp3 --format json --timestamps
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="入力ファイル（音声または動画）",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="出力ファイルパス（指定しない場合は自動生成）",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json", "srt", "vtt"],
        default="text",
        help="出力形式（default: text）",
    )

    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Demucsノイズ除去を無効化（高速化）",
    )

    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="タイムスタンプを含める（text, json形式で有効）",
    )

    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="単語レベルのタイムスタンプを生成",
    )

    parser.add_argument(
        "--model",
        default="kotoba-v2",
        choices=["kotoba-v2", "large-v3", "large-v3-turbo"],
        help="使用するモデル（default: kotoba-v2）",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="詳細ログを出力",
    )

    args = parser.parse_args()

    # モデルID変換
    model_map = {
        "kotoba-v2": DEFAULT_MODEL,
        "large-v3": "large-v3",
        "large-v3-turbo": "large-v3-turbo",
    }
    model_id = model_map[args.model]

    try:
        # 入力ファイル検証
        validate_input_file(args.input)

        # 出力パス決定
        if args.output:
            output_path = args.output
        else:
            ext_map = {"text": ".txt", "json": ".json", "srt": ".srt", "vtt": ".vtt"}
            output_path = args.input.with_suffix(ext_map[args.format])

        # 一時ディレクトリで処理
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = args.input

            # 動画の場合は音声抽出
            if is_video_file(args.input):
                audio_path = temp_path / f"{args.input.stem}.wav"
                extract_audio_from_video(args.input, audio_path)

            # Demucsノイズ除去
            if not args.no_denoise:
                audio_path = apply_demucs_denoise(
                    audio_path, temp_path, verbose=args.verbose
                )

            # 文字起こし
            result = transcribe_audio(
                audio_path,
                model_id=model_id,
                word_timestamps=args.word_timestamps,
                verbose=args.verbose,
            )

        # 出力フォーマット
        if args.format == "text":
            content = format_as_text(result, timestamps=args.timestamps)
        elif args.format == "json":
            content = format_as_json(result, timestamps=args.timestamps)
        elif args.format == "srt":
            content = format_as_srt(result)
        elif args.format == "vtt":
            content = format_as_vtt(result)

        # 保存
        save_output(content, output_path)

        # 統計情報
        print("\n--- 統計情報 ---")
        print(f"文字数: {len(result['text'])} 文字")
        print(f"セグメント数: {len(result['segments'])}")
        if result.get("duration"):
            print(f"音声の長さ: {format_timestamp(result['duration'])}")
            if result.get("duration_after_vad"):
                vad_reduction = (
                    1 - result["duration_after_vad"] / result["duration"]
                ) * 100
                print(
                    f"VAD後の長さ: {format_timestamp(result['duration_after_vad'])} "
                    f"({vad_reduction:.1f}%削減)"
                )

        return 0

    except TranscriptionError as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        return 1
    except Exception as e:
        print(f"\n予期しないエラー: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

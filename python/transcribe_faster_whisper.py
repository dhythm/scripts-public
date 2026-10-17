#!/usr/bin/env python3
"""
Qwen3-ASR / Faster-Whisperを使用した音声/動画文字起こしツール

デフォルトはQwen3-ASR（MLX、Apple Silicon専用）。--engine whisper で faster-whisper を使用する。

使用方法:
    python transcribe_faster_whisper.py <音声/動画ファイル> [オプション]

例:
    python transcribe_faster_whisper.py input.m4a --hotwords "固有名詞 専門用語"
    python transcribe_faster_whisper.py input.mp3 --engine whisper --model large-v3 --language ja
    python transcribe_faster_whisper.py input.wav --device cuda --compute_type float16
    python transcribe_faster_whisper.py video.mp4 --model large-v2 --language ja
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps
from pydub import AudioSegment
from tqdm import tqdm

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

# VAD閾値の自動選択で試す候補（高い順）。高いほど無音を多く削れる
VAD_THRESHOLD_CANDIDATES = [0.50, 0.30, 0.15, 0.05]
# 最大検出量に対してこの割合以上を拾える閾値のうち、最も高いものを採用する
VAD_AUTO_COVERAGE = 0.95
# ピークノーマライズの目標値。0 dBFS に余裕を残してクリップを防ぐ
NORMALIZE_TARGET_PEAK_DBFS = -3.0


def compute_normalize_gain(
    peak_dbfs: float,
    target_peak_dbfs: float = NORMALIZE_TARGET_PEAK_DBFS
) -> float:
    """
    ピークを目標値に揃えるためのゲイン(dB)を計算する

    Args:
        peak_dbfs: 現在のピーク音量(dBFS)。無音の場合は -inf
        target_peak_dbfs: 目標ピーク音量(dBFS)

    Returns:
        適用すべきゲイン(dB)。無音の場合は 0
    """
    if not math.isfinite(peak_dbfs):
        return 0.0
    return target_peak_dbfs - peak_dbfs


def _to_dbfs(value: float) -> float:
    """振幅(0-1)をdBFSに変換する。0は -inf"""
    return 20 * math.log10(value) if value > 0 else float('-inf')


def analyze_audio_levels(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int = 50
) -> Dict[str, float]:
    """
    音声のレベルを解析し、前処理パラメータ決定の材料を返す

    フレームごとのRMSを計算し、下位10%を背景ノイズ、上位5%を発話レベルとみなす。

    Args:
        samples: -1.0〜1.0 に正規化されたモノラル音声
        sample_rate: サンプリングレート
        frame_ms: RMSを計算するフレーム長(ms)

    Returns:
        peak_dbfs, rms_dbfs, noise_floor_dbfs, speech_level_dbfs,
        snr_db, clipping_ratio を持つ辞書
    """
    if len(samples) == 0:
        raise ValueError("音声が空です")

    samples = np.asarray(samples, dtype=np.float32)
    abs_samples = np.abs(samples)

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = max(1, len(samples) // frame_len)
    frames = samples[:n_frames * frame_len].reshape(n_frames, -1)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    voiced_rms = frame_rms[frame_rms > 0]

    if len(voiced_rms) == 0:
        noise_floor_dbfs = speech_level_dbfs = float('-inf')
        snr_db = 0.0
    else:
        noise_floor_dbfs = _to_dbfs(float(np.percentile(voiced_rms, 10)))
        speech_level_dbfs = _to_dbfs(float(np.percentile(voiced_rms, 95)))
        snr_db = speech_level_dbfs - noise_floor_dbfs

    return {
        'peak_dbfs': _to_dbfs(float(abs_samples.max())),
        'rms_dbfs': _to_dbfs(float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))),
        'noise_floor_dbfs': noise_floor_dbfs,
        'speech_level_dbfs': speech_level_dbfs,
        'snr_db': snr_db,
        'clipping_ratio': float(np.mean(abs_samples >= 0.999)),
    }


# SN比がこれ未満ならノイズ除去を有効化する。ノイズ除去は音声も削り
# 認識精度を下げることがあるため、雑音が明らかに大きい場合に限定する
DENOISE_SNR_THRESHOLD_DB = 10.0
# SN比がこれ未満なら強めのノイズ除去をかける
STRONG_DENOISE_SNR_THRESHOLD_DB = 5.0
# 元音声のクリップ率がこれを超えたら警告する
CLIPPING_WARN_RATIO = 0.001
# 発話レベルがこれ未満なら録音が小さすぎると警告する
QUIET_SPEECH_WARN_DBFS = -40.0


def recommend_preprocessing(stats: Dict[str, float]) -> Dict:
    """
    音声レベルの解析結果から前処理パラメータを決める

    Args:
        stats: analyze_audio_levels の戻り値

    Returns:
        denoise, noise_reduce_amount, warnings を持つ辞書
    """
    snr_db = stats['snr_db']
    warnings = []

    if snr_db < STRONG_DENOISE_SNR_THRESHOLD_DB:
        denoise, noise_reduce_amount = True, 0.8
    elif snr_db < DENOISE_SNR_THRESHOLD_DB:
        denoise, noise_reduce_amount = True, 0.5
    else:
        denoise, noise_reduce_amount = False, 0.0

    if stats['clipping_ratio'] > CLIPPING_WARN_RATIO:
        warnings.append(
            f"元音声がクリップしています（{stats['clipping_ratio']:.2%}）。"
            "歪んだ箇所は誤認識しやすいため、録音レベルを下げて録り直すのが確実です。"
        )
    if stats['speech_level_dbfs'] < QUIET_SPEECH_WARN_DBFS:
        warnings.append(
            f"発話レベルが小さいです（{stats['speech_level_dbfs']:.1f} dBFS）。"
            "ノーマライズで持ち上げますが、マイクを近づけると精度が上がります。"
        )

    return {
        'denoise': denoise,
        'noise_reduce_amount': noise_reduce_amount,
        'warnings': warnings,
    }


def normalize_peak(
    audio: AudioSegment,
    target_peak_dbfs: float = NORMALIZE_TARGET_PEAK_DBFS
) -> AudioSegment:
    """
    ピーク音量を目標値に揃える

    平均音量(audio.dBFS)を基準にすると、ピークが 0 dBFS を超えて
    クリップし音声が歪むため、必ずピーク(audio.max_dBFS)を基準にする。
    """
    gain = compute_normalize_gain(audio.max_dBFS, target_peak_dbfs)
    if gain == 0.0:
        return audio
    return audio.apply_gain(gain)


def choose_vad_threshold(
    scan_results: List[tuple],
    coverage: float = VAD_AUTO_COVERAGE
) -> float:
    """
    VADスキャン結果から最適な閾値を選択する

    最も低い閾値で検出された音声量を「取りこぼしのない基準」とみなし、
    その coverage 以上を検出できる最も高い閾値を返す。
    高い閾値ほど無音のスキップ効率が良いため、カバレッジを満たす範囲で
    最大の閾値を選ぶことで「取りこぼし防止」と「効率」を両立する。

    Args:
        scan_results: (閾値, 検出音声秒数) のリスト
        coverage: 最大検出量に対する必要カバレッジ（0より大きく1以下）

    Returns:
        選択された閾値
    """
    if not scan_results:
        raise ValueError("スキャン結果が空です")
    if not 0 < coverage <= 1:
        raise ValueError(f"coverageは(0, 1]の範囲で指定してください: {coverage}")

    max_speech = max(speech for _, speech in scan_results)
    if max_speech == 0:
        # どの閾値でも音声が検出されない場合は最も高い閾値を返す
        return max(threshold for threshold, _ in scan_results)

    qualified = [
        threshold for threshold, speech in scan_results
        if speech >= coverage * max_speech
    ]
    if qualified:
        return max(qualified)
    # 数値誤差等でどれも満たさない場合は最も低い閾値（最大検出）を返す
    return min(threshold for threshold, _ in scan_results)


def chunks_to_segments(chunks: Optional[List[Dict]]) -> List[Dict]:
    """
    Qwen3-ASR のチャンク（約30秒単位）を共通のセグメント形式に変換する

    空文字のチャンク（無音区間）は除外し、idは1から振り直す。
    """
    segments = []
    for chunk in chunks or []:
        text = chunk.get('text', '')
        if not text.strip():
            continue
        segments.append({
            'id': len(segments) + 1,
            'start': chunk['start'],
            'end': chunk['end'],
            'text': text,
        })
    return segments


def build_qwen_context(initial_prompt: Optional[str], hotwords: Optional[str]) -> str:
    """initial_prompt と hotwords を Qwen3-ASR の context（語彙ヒント）にまとめる"""
    return ' '.join(part for part in (initial_prompt, hotwords) if part)


class FasterWhisperTranscriber:
    """Faster-Whisperを使用した音声文字起こしクラス"""

    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.flv'}
    SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS
    MODEL_SIZES = [
        'tiny', 'tiny.en', 'base', 'base.en', 
        'small', 'small.en', 'medium', 'medium.en',
        'large-v1', 'large-v2', 'large-v3', 'large',
        'large-v3-turbo', 'turbo', 'distil-large-v3'
    ]
    COMPUTE_TYPES = ['default', 'float16', 'int8_float16', 'int8']
    
    def __init__(
        self,
        model_name: str = 'base',
        device: str = 'auto',
        compute_type: str = 'default',
        cpu_threads: int = 0,
        num_workers: int = 1
    ):
        """
        Args:
            model_name: 使用するWhisperモデルのサイズ
            device: 使用するデバイス ('cuda', 'cpu', 'auto')
            compute_type: 計算精度
            cpu_threads: CPU使用時のスレッド数（0で自動）
            num_workers: 並列ワーカー数
        """
        if model_name not in self.MODEL_SIZES:
            raise ValueError(f"無効なモデルサイズ: {model_name}. 有効なオプション: {self.MODEL_SIZES}")
        
        # デバイスの決定
        if device == 'auto':
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"自動検出されたデバイス: {device}")
            except ImportError:
                device = 'cpu'
                print("PyTorchがインストールされていません。CPUを使用します。")
        
        print(f"モデル '{model_name}' をロード中...")
        print(f"デバイス: {device}, 計算精度: {compute_type}")
        
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers
        )
        
        self.model_name = model_name
        self.device = device
        print(f"モデル '{model_name}' のロードが完了しました。")
    
    def is_video_file(self, file_path: Path) -> bool:
        """動画ファイルかどうかを判定"""
        return file_path.suffix.lower() in self.SUPPORTED_VIDEO_FORMATS

    def extract_audio_from_video(self, video_path: Path, output_dir: Path) -> Path:
        """
        動画ファイルから音声を抽出

        Args:
            video_path: 動画ファイルのパス
            output_dir: 出力ディレクトリ

        Returns:
            抽出した音声ファイルのパス
        """
        print(f"動画ファイルから音声を抽出中: {video_path.name}")

        output_path = output_dir / f"{video_path.stem}_audio.wav"

        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',                    # 映像なし
            '-acodec', 'pcm_s16le',   # 16bit PCM
            '-ar', '16000',           # 16kHz (Whisper推奨)
            '-ac', '1',               # モノラル
            '-y',                     # 上書き確認なし
            str(output_path)
        ]

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"音声抽出完了: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpegエラー: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpegが見つかりません。ffmpegをインストールしてください。\n"
                "macOS: brew install ffmpeg\n"
                "Ubuntu: sudo apt install ffmpeg"
            )

    def validate_input_file(self, file_path: Path) -> None:
        """入力ファイル（音声/動画）の検証"""
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"パスがファイルではありません: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"サポートされていないファイル形式: {file_path.suffix}\n"
                f"サポートされている音声形式: {', '.join(sorted(self.SUPPORTED_AUDIO_FORMATS))}\n"
                f"サポートされている動画形式: {', '.join(sorted(self.SUPPORTED_VIDEO_FORMATS))}"
            )

        # ファイルサイズチェック（1GB以上は警告）
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 1024:
            print(f"警告: ファイルサイズが大きいです ({file_size_mb:.1f} MB)。処理に時間がかかる可能性があります。")

    def validate_audio_file(self, file_path: Path) -> None:
        """音声ファイルの検証（後方互換性のため残す）"""
        self.validate_input_file(file_path)
    
    def apply_noise_reduction(
        self,
        audio: AudioSegment,
        reduce_amount: float = 0.8,
        stationary: bool = False
    ) -> AudioSegment:
        """
        ノイズ除去を適用

        Args:
            audio: AudioSegmentオブジェクト
            reduce_amount: ノイズ除去の強度（0.0-1.0）。1.0が最大
            stationary: 定常ノイズ（ファンの音など）の場合はTrue

        Returns:
            ノイズ除去後のAudioSegment
        """
        if not NOISEREDUCE_AVAILABLE:
            print("警告: noisereduceがインストールされていません。ノイズ除去をスキップします。")
            return audio

        print(f"ノイズ除去を実行中... (強度: {reduce_amount:.1f}, 定常ノイズ: {stationary})")

        # AudioSegmentをnumpy配列に変換
        samples = np.array(audio.get_array_of_samples())

        # ステレオの場合はモノラルに変換
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1)

        # float32に正規化
        samples = samples.astype(np.float32)
        if audio.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio.sample_width == 1:  # 8-bit
            samples = samples / 128.0

        # ノイズ除去を適用
        # 最初の0.5秒をノイズサンプルとして使用
        sample_rate = audio.frame_rate
        noise_duration = min(0.5, len(samples) / sample_rate / 2)  # 最大0.5秒
        noise_len = int(noise_duration * sample_rate)

        if noise_len > 0:
            y_noise = samples[:noise_len]
            reduced = nr.reduce_noise(
                y=samples,
                sr=sample_rate,
                y_noise=y_noise,
                stationary=stationary,
                prop_decrease=reduce_amount
            )
        else:
            # ノイズサンプルなしで実行
            reduced = nr.reduce_noise(
                y=samples,
                sr=sample_rate,
                stationary=stationary,
                prop_decrease=reduce_amount
            )

        # 16-bit intに戻す
        reduced = (reduced * 32768.0).astype(np.int16)

        # AudioSegmentに戻す
        denoised_audio = AudioSegment(
            reduced.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

        print("ノイズ除去が完了しました。")
        return denoised_audio

    def scan_vad_speech_durations(
        self,
        wav_path: Path,
        thresholds: List[float],
        vad_parameters: Dict
    ) -> List[tuple]:
        """
        文字起こしせずにVADのみを閾値別に実行し、検出音声量を計測する

        Silero VADは軽量（54分の音声でも閾値1つあたり数秒程度）なため、
        本処理の前に実行して閾値を調整するコストは小さい。

        Args:
            wav_path: 音声ファイルのパス
            thresholds: 試す閾値のリスト
            vad_parameters: threshold以外のVADパラメータ

        Returns:
            (閾値, 検出音声秒数) のリスト
        """
        sampling_rate = 16000
        audio = decode_audio(str(wav_path), sampling_rate=sampling_rate)

        results = []
        for threshold in thresholds:
            options = VadOptions(
                threshold=threshold,
                min_speech_duration_ms=vad_parameters.get('min_speech_duration_ms', 50),
                min_silence_duration_ms=vad_parameters.get('min_silence_duration_ms', 800),
                speech_pad_ms=vad_parameters.get('speech_pad_ms', 800),
            )
            timestamps = get_speech_timestamps(audio, vad_options=options)
            speech_sec = sum(t['end'] - t['start'] for t in timestamps) / sampling_rate
            results.append((threshold, speech_sec))

        return results

    def _auto_select_vad_threshold(
        self,
        wav_path: Path,
        vad_parameters: Dict
    ) -> float:
        """
        事前VADスキャンで最適な閾値を選択し、結果を表示する

        スキャンに失敗した場合は従来のデフォルト値(0.20)にフォールバックする。
        """
        fallback_threshold = 0.20
        try:
            print("VAD閾値を自動選択中（事前スキャン）...")
            scan_start = time.time()
            scan_results = self.scan_vad_speech_durations(
                wav_path, VAD_THRESHOLD_CANDIDATES, vad_parameters
            )
            chosen = choose_vad_threshold(scan_results)

            max_speech = max(speech for _, speech in scan_results)
            for threshold, speech in scan_results:
                marker = ' ←選択' if threshold == chosen else ''
                print(f"  threshold={threshold:.2f}: 検出 {speech/60:.1f}分{marker}")
            if max_speech == 0:
                print("  警告: どの閾値でも音声が検出されませんでした。")
            print(f"事前スキャン完了 ({time.time() - scan_start:.1f}秒)")
            return chosen
        except Exception as e:
            print(f"警告: VAD事前スキャンに失敗しました ({e})。"
                  f"閾値 {fallback_threshold} を使用します。")
            return fallback_threshold

    def _auto_select_preprocessing(
        self,
        audio_path: Path,
        fallback_noise_reduce_amount: float
    ) -> tuple:
        """
        音声レベルを解析してノイズ除去の要否と強度を決め、結果を表示する

        解析に失敗した場合はノイズ除去なしにフォールバックする。

        Returns:
            (denoise, noise_reduce_amount)
        """
        try:
            print("音声レベルを解析中...")
            sampling_rate = 16000
            audio = decode_audio(str(audio_path), sampling_rate=sampling_rate)
            stats = analyze_audio_levels(audio, sampling_rate)
            rec = recommend_preprocessing(stats)

            print(f"  ピーク: {stats['peak_dbfs']:.1f} dBFS / "
                  f"発話レベル: {stats['speech_level_dbfs']:.1f} dBFS / "
                  f"背景ノイズ: {stats['noise_floor_dbfs']:.1f} dBFS")
            print(f"  SN比: {stats['snr_db']:.1f} dB / "
                  f"クリップ率: {stats['clipping_ratio']:.3%}")
            if rec['denoise']:
                print(f"  → ノイズ除去: 有効 (強度 {rec['noise_reduce_amount']:.1f})")
            else:
                print("  → ノイズ除去: 不要（認識精度を下げないため無効）")
            for warning in rec['warnings']:
                print(f"  警告: {warning}")
            return rec['denoise'], rec['noise_reduce_amount']
        except Exception as e:
            print(f"警告: 音声レベルの解析に失敗しました ({e})。ノイズ除去なしで続行します。")
            return False, fallback_noise_reduce_amount

    def convert_to_wav_if_needed(
        self,
        file_path: Path,
        normalize: bool = False,
        gain_db: float = 0,
        denoise: bool = False,
        noise_reduce_amount: float = 0.8
    ) -> Path:
        """
        必要に応じてWAV形式に変換し、音声を前処理

        Args:
            file_path: 音声ファイルのパス
            normalize: ピークノーマライズを適用するか
            gain_db: ゲイン調整(dB)。正の値で音量アップ、負の値で音量ダウン
            denoise: ノイズ除去を適用するか
            noise_reduce_amount: ノイズ除去の強度（0.0-1.0）

        Returns:
            変換後のWAVファイルのパス
        """
        # WAVファイルでも前処理が必要な場合は読み込む
        needs_processing = normalize or gain_db != 0 or denoise
        if file_path.suffix.lower() == '.wav' and not needs_processing:
            return file_path

        print("音声ファイルをWAV形式に変換中...")
        audio = AudioSegment.from_file(str(file_path))

        # ノイズ除去（他の処理の前に実行）
        if denoise:
            audio = self.apply_noise_reduction(
                audio,
                reduce_amount=noise_reduce_amount,
                stationary=False  # 非定常ノイズ対応
            )

        # ゲイン調整
        if gain_db != 0:
            print(f"ゲイン調整: {gain_db:+.1f} dB")
            audio = audio + gain_db

        # ノーマライズ（ピークを-3.0 dBFSに調整）
        if normalize:
            gain = compute_normalize_gain(audio.max_dBFS)
            print(f"ノーマライズ: {gain:+.1f} dB (ピーク: {audio.max_dBFS:.1f} → {NORMALIZE_TARGET_PEAK_DBFS:.1f} dBFS)")
            audio = normalize_peak(audio)

        # 16kHzにリサンプリング（Whisperの推奨）
        audio = audio.set_frame_rate(16000)

        # 一時ファイルパスを作成
        wav_path = file_path.with_suffix('.wav')
        temp_wav_path = wav_path.parent / f"_temp_{wav_path.name}"

        # WAV形式でエクスポート
        audio.export(str(temp_wav_path), format='wav')
        print("変換が完了しました。")

        return temp_wav_path

    def preview_vad(
        self,
        audio_path: Path,
        vad_parameters: Optional[Dict] = None
    ) -> Dict[str, float]:
        """文字起こし前にVAD適用後の音声長を見積もる。"""
        sampling_rate = 16000
        audio = decode_audio(str(audio_path), sampling_rate=sampling_rate)
        options = VadOptions(**(vad_parameters or {}))
        speech_chunks = get_speech_timestamps(audio, options)

        original_duration = len(audio) / sampling_rate
        duration_after_vad = sum(
            chunk['end'] - chunk['start'] for chunk in speech_chunks
        ) / sampling_rate
        removed_duration = original_duration - duration_after_vad
        reduction_percent = (
            removed_duration / original_duration * 100
            if original_duration > 0
            else 0.0
        )

        return {
            'duration': original_duration,
            'duration_after_vad': duration_after_vad,
            'removed_duration': removed_duration,
            'reduction_percent': reduction_percent,
        }
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = 'transcribe',
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        temperature: Union[float, List[float]] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.9,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        condition_on_previous_text: bool = True,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict] = None,
        vad_auto_threshold: bool = True,
        normalize: bool = True,
        gain_db: float = 0,
        denoise: Optional[bool] = False,
        noise_reduce_amount: float = 0.8
    ) -> Dict:
        """
        音声ファイルを文字起こし

        Args:
            audio_path: 音声ファイルのパス
            language: 言語コード（例: 'ja', 'en'）。Noneの場合は自動検出
            task: 'transcribe' または 'translate'
            beam_size: ビームサーチのビーム幅
            best_of: 候補数
            patience: ビームサーチのpatience
            length_penalty: 長さペナルティ
            temperature: サンプリング温度
            compression_ratio_threshold: 圧縮率の閾値
            log_prob_threshold: ログ確率の閾値
            no_speech_threshold: 無音検出の閾値
            initial_prompt: 初期プロンプト
            hotwords: 認識させたい固有名詞・専門用語（スペース区切り）
            condition_on_previous_text: 直前の認識結果を次の区間の文脈に使う。
                Falseにすると同じ文の繰り返し（ハルシネーション）が起きにくい
            word_timestamps: 単語レベルのタイムスタンプを生成
            vad_filter: Voice Activity Detectionフィルタを使用
            vad_parameters: VADパラメータ
            vad_auto_threshold: VAD閾値を事前スキャンで自動選択する
                （vad_parametersでthresholdが明示された場合は無効）
            normalize: 音声をピークノーマライズ
            gain_db: ゲイン調整(dB)
            denoise: ノイズ除去を適用。Noneの場合は音声を解析して自動判定
            noise_reduce_amount: ノイズ除去の強度（0.0-1.0）

        Returns:
            文字起こし結果の辞書
        """
        input_path = Path(audio_path)
        self.validate_input_file(input_path)

        # 動画ファイルの場合は音声を抽出
        video_temp_dir = None
        extracted_audio_path = None

        if self.is_video_file(input_path):
            video_temp_dir = tempfile.mkdtemp(prefix="whisper_video_")
            extracted_audio_path = self.extract_audio_from_video(
                input_path, Path(video_temp_dir)
            )
            audio_path = extracted_audio_path
        else:
            audio_path = input_path

        # デフォルトVADパラメータ（小さい音声に最適化）
        threshold_is_explicit = False
        if vad_filter:
            vad_parameters = dict(vad_parameters) if vad_parameters else {}
            threshold_is_explicit = 'threshold' in vad_parameters
            vad_parameters.setdefault('min_speech_duration_ms', 50)  # 短い発話も保持
            vad_parameters.setdefault('min_silence_duration_ms', 800)  # 短い無音で分割しない
            vad_parameters.setdefault('speech_pad_ms', 800)  # 音声の前後を保護

        # 一時ファイルパスを保持
        temp_file = None

        try:
            # ノイズ除去の要否を音声解析で自動判定
            if denoise is None:
                denoise, noise_reduce_amount = self._auto_select_preprocessing(
                    audio_path, noise_reduce_amount
                )

            # 必要に応じてWAV形式に変換（前処理を含む）
            wav_path = self.convert_to_wav_if_needed(
                audio_path,
                normalize=normalize,
                gain_db=gain_db,
                denoise=denoise,
                noise_reduce_amount=noise_reduce_amount
            )
            if wav_path != audio_path:
                temp_file = wav_path

            # VAD閾値の決定（明示指定がなければ事前スキャンで自動選択）
            if vad_filter and not threshold_is_explicit:
                if vad_auto_threshold:
                    vad_parameters['threshold'] = self._auto_select_vad_threshold(
                        wav_path, vad_parameters
                    )
                else:
                    vad_parameters['threshold'] = 0.20
            if vad_filter:
                print(f"VADパラメータ: threshold={vad_parameters['threshold']}")

            # 確定したVADパラメータで適用後の長さを事前見積もり
            if vad_filter:
                vad_preview = self.preview_vad(wav_path, vad_parameters)
                print("\n--- VAD事前確認 ---")
                print(
                    f"元の長さ: {self._format_timestamp(vad_preview['duration'])} "
                    f"({vad_preview['duration']:.1f}秒)"
                )
                print(
                    "VAD後の見込み: "
                    f"{self._format_timestamp(vad_preview['duration_after_vad'])} "
                    f"({vad_preview['duration_after_vad']:.1f}秒)"
                )
                print(
                    f"削除見込み: {self._format_timestamp(vad_preview['removed_duration'])} "
                    f"({vad_preview['removed_duration']:.1f}秒 / "
                    f"{vad_preview['reduction_percent']:.1f}%削減)"
                )
                print("設定に問題があれば Ctrl+C で中断してください。\n")

            # 文字起こし実行
            print("文字起こしを開始します...")
            if language:
                print(f"言語: {language}")
            else:
                print("言語を自動検出します...")
            
            return self._run_model(
                wav_path,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                condition_on_previous_text=condition_on_previous_text,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters
            )
            
        finally:
            # 一時ファイルの削除
            if temp_file and temp_file.exists():
                temp_file.unlink()

            # 動画から抽出した一時ファイルの削除
            if extracted_audio_path and extracted_audio_path.exists():
                extracted_audio_path.unlink()
            if video_temp_dir:
                import shutil
                shutil.rmtree(video_temp_dir, ignore_errors=True)
    
    def _run_model(
        self,
        wav_path: Path,
        language: Optional[str],
        task: str,
        beam_size: int,
        best_of: int,
        patience: float,
        length_penalty: float,
        temperature: Union[float, List[float]],
        compression_ratio_threshold: float,
        log_prob_threshold: float,
        no_speech_threshold: float,
        initial_prompt: Optional[str],
        hotwords: Optional[str],
        condition_on_previous_text: bool,
        word_timestamps: bool,
        vad_filter: bool,
        vad_parameters: Optional[Dict]
    ) -> Dict:
        """前処理済みの音声をモデルで文字起こしし、結果の辞書を返す"""
        # 文字起こし実行
        segments, info = self.model.transcribe(
            str(wav_path),
            language=language,
            task=task,
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters
        )
        
        # セグメントをリストに変換（プログレスバー付き）
        segment_list = []
        full_text = []
        
        print("セグメントを処理中...")
        for segment in tqdm(segments, desc="セグメント"):
            seg_dict = {
                'id': segment.id,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'no_speech_prob': segment.no_speech_prob,
                'avg_logprob': segment.avg_logprob,
                'compression_ratio': segment.compression_ratio,
            }
            
            # 単語レベルのタイムスタンプを追加
            if word_timestamps and segment.words:
                seg_dict['words'] = [
                    {
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    }
                    for word in segment.words
                ]
            
            segment_list.append(seg_dict)
            full_text.append(segment.text)
        
        # 検出された言語を表示
        if info.language:
            print(f"検出された言語: {info.language}")
        
        print("文字起こしが完了しました。")
        
        # 結果を辞書形式で返す
        result = {
            'text': ''.join(full_text),
            'segments': segment_list,
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration,
            'duration_after_vad': info.duration_after_vad,
            'all_language_probs': info.all_language_probs
        }
        
        return result

    def save_result(
        self,
        result: Dict,
        output_path: Path,
        output_format: str = 'text',
        include_timestamps: bool = False,
        include_confidence: bool = False
    ) -> None:
        """
        文字起こし結果を保存
        
        Args:
            result: 文字起こし結果
            output_path: 出力ファイルパス
            output_format: 出力形式 ('text', 'json', 'srt', 'vtt')
            include_timestamps: タイムスタンプを含めるか
            include_confidence: 信頼度情報を含めるか
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'text':
            # テキスト形式で保存
            with output_path.open('w', encoding='utf-8') as f:
                if include_timestamps and 'segments' in result:
                    for segment in result['segments']:
                        start = self._format_timestamp(segment['start'])
                        end = self._format_timestamp(segment['end'])
                        text = segment['text'].strip()
                        
                        if include_confidence and segment.get('no_speech_prob') is not None:
                            conf = f" [信頼度: {1 - segment['no_speech_prob']:.2f}]"
                            f.write(f"[{start} --> {end}]{conf} {text}\n")
                        else:
                            f.write(f"[{start} --> {end}] {text}\n")
                else:
                    f.write(result['text'].strip())
        
        elif output_format == 'json':
            # JSON形式で保存
            output_data = {
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'language_probability': result.get('language_probability', 0),
                'duration': result.get('duration', 0),
                'model': self.model_name,
                'device': self.device,
            }
            
            if include_timestamps and 'segments' in result:
                segments_data = []
                for seg in result['segments']:
                    seg_data = {
                        'id': seg['id'],
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text'].strip(),
                    }
                    
                    if include_confidence and 'no_speech_prob' in seg:
                        seg_data.update({
                            'no_speech_prob': seg['no_speech_prob'],
                            'avg_logprob': seg['avg_logprob'],
                            'compression_ratio': seg['compression_ratio'],
                        })
                    
                    if 'words' in seg:
                        seg_data['words'] = seg['words']
                    
                    segments_data.append(seg_data)
                
                output_data['segments'] = segments_data
            
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        elif output_format == 'srt':
            # SRT字幕形式で保存
            with output_path.open('w', encoding='utf-8') as f:
                if 'segments' in result:
                    for i, segment in enumerate(result['segments'], 1):
                        start = self._format_timestamp_srt(segment['start'])
                        end = self._format_timestamp_srt(segment['end'])
                        text = segment['text'].strip()
                        
                        f.write(f"{i}\n")
                        f.write(f"{start} --> {end}\n")
                        f.write(f"{text}\n\n")
        
        elif output_format == 'vtt':
            # WebVTT字幕形式で保存
            with output_path.open('w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                
                if 'segments' in result:
                    for segment in result['segments']:
                        start = self._format_timestamp_vtt(segment['start'])
                        end = self._format_timestamp_vtt(segment['end'])
                        text = segment['text'].strip()
                        
                        f.write(f"{start} --> {end}\n")
                        f.write(f"{text}\n\n")
        
        print(f"結果を保存しました: {output_path}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """秒をHH:MM:SS形式に変換"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    
    def _format_timestamp_srt(self, seconds: float) -> str:
        """秒をSRT形式のタイムスタンプに変換"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """秒をWebVTT形式のタイムスタンプに変換"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class Qwen3ASRTranscriber(FasterWhisperTranscriber):
    """
    Qwen3-ASR（MLX実装）を使用した音声文字起こしクラス

    前処理（ノーマライズ・ノイズ除去・動画からの音声抽出）と結果の保存は
    FasterWhisperTranscriber と共通で、推論部分（_run_model）のみ差し替える。
    MLXはApple SiliconのGPUを使うため、CPU実行のfaster-whisperより大幅に速い。
    """

    DEFAULT_MODEL = 'Qwen/Qwen3-ASR-1.7B'

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Args:
            model_name: Hugging FaceのモデルID（例: Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B）
        """
        try:
            from mlx_qwen3_asr import Session
        except ImportError:
            raise RuntimeError(
                "mlx-qwen3-asr がインストールされていません。\n"
                "Apple Silicon の Mac で uv sync を実行するか、--engine whisper を指定してください。"
            )

        print(f"モデル '{model_name}' をロード中...")
        print("デバイス: mlx (Apple Silicon GPU)")
        self.session = Session(model=model_name)
        self.model_name = model_name
        self.device = 'mlx'
        print(f"モデル '{model_name}' のロードが完了しました。")

    def _run_model(
        self,
        wav_path: Path,
        language: Optional[str] = None,
        task: str = 'transcribe',
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        **_whisper_only_options
    ) -> Dict:
        """Qwen3-ASRで文字起こしする。Whisper固有のデコードオプションは無視する"""
        if task != 'transcribe':
            raise ValueError(
                "Qwen3-ASR は翻訳(translate)に対応していません。--engine whisper を指定してください。"
            )

        context = build_qwen_context(initial_prompt, hotwords)
        if context:
            print(f"語彙ヒント(context): {context}")

        result = self.session.transcribe(
            str(wav_path),
            language=language,
            context=context,
            return_chunks=True,
        )

        if result.truncated:
            print("警告: 一部のチャンクがトークン上限で途中終了しました。末尾が欠けている可能性があります。")
        if result.finish_reason in ('repetition', 'mixed'):
            print("警告: 同じ文の繰り返しを検出して生成を打ち切った箇所があります。")
        if result.language:
            print(f"検出された言語: {result.language}")
        print("文字起こしが完了しました。")

        segments = chunks_to_segments(result.chunks)
        duration = max((c['end'] for c in result.chunks or []), default=0.0)
        return {
            'text': result.text,
            'segments': segments,
            'language': result.language,
            'language_probability': None,
            'duration': duration,
            'duration_after_vad': None,
            'all_language_probs': None,
        }


# Qwen3-ASR では効果のない、faster-whisper 専用のCLIオプション
WHISPER_ONLY_OPTIONS = [
    'beam_size', 'temperature', 'vad_filter', 'vad_threshold', 'vad_auto_threshold',
    'vad_min_speech_duration', 'vad_max_speech_duration', 'vad_min_silence_duration',
    'vad_speech_pad', 'no_speech_threshold', 'condition_on_previous_text',
    'word_timestamps', 'device', 'compute_type', 'cpu_threads', 'num_workers',
]


def resolve_engine(engine: Optional[str], model: Optional[str]) -> str:
    """
    使用するエンジンを決める

    --engine 省略時に Whisper のモデル名（large-v3 など）が指定されていれば
    whisper を選ぶ。従来の「--model large-v3」だけのコマンドを壊さないため。
    """
    if engine:
        return engine
    if model in FasterWhisperTranscriber.MODEL_SIZES:
        return 'whisper'
    return 'qwen3'


WHISPER_DEFAULT_MODEL = 'large-v3'


def resolve_model(engine: str, model: Optional[str]) -> str:
    """指定がなければエンジンごとのデフォルトモデルを返す"""
    if model:
        return model
    if engine == 'whisper':
        return WHISPER_DEFAULT_MODEL
    return Qwen3ASRTranscriber.DEFAULT_MODEL


def find_changed_options(values: Dict, defaults: Dict) -> List[str]:
    """デフォルト値から変更されたオプション名を返す"""
    return [name for name, value in values.items() if value != defaults.get(name)]


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Qwen3-ASR / Faster-Whisperを使用した音声/動画文字起こしツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法（Qwen3-ASR）
  %(prog)s input.mp3

  # 固有名詞・専門用語をヒントとして与える
  %(prog)s input.mp3 --hotwords "Kubernetes 岡田"

  # faster-whisper を使用
  %(prog)s input.mp3 --engine whisper --model large-v3 --language ja

  # GPUを使用して高速処理
  %(prog)s input.wav --device cuda --compute_type float16

  # 動画ファイルから文字起こし
  %(prog)s video.mp4 --model large-v2 --language ja

  # 単語レベルのタイムスタンプ付きJSON出力
  %(prog)s input.mp4 --format json --word_timestamps --output result.json

  # WebVTT字幕形式で出力
  %(prog)s input.mp3 --format vtt --output subtitles.vtt

サポートされる形式:
  音声: mp3, wav, m4a, flac, aac, ogg, wma
  動画: mp4, mkv, avi, mov, webm, wmv, flv (ffmpeg必須)
        """
    )
    
    parser.add_argument(
        'audio_file',
        type=Path,
        help='文字起こしする音声/動画ファイル'
    )
    
    parser.add_argument(
        '--engine',
        choices=['qwen3', 'whisper'],
        default=None,
        help='文字起こしエンジン。qwen3はApple SiliconのGPUで高速・句読点付き、'
             'whisperは従来のfaster-whisper。省略時はqwen3だが、'
             '--modelにWhisperのモデル名を指定した場合はwhisper (default: qwen3)'
    )

    parser.add_argument(
        '--model',
        default=None,
        help='使用するモデル。qwen3: Hugging FaceのモデルID '
             f'(default: {Qwen3ASRTranscriber.DEFAULT_MODEL})、'
             f'whisper: {", ".join(FasterWhisperTranscriber.MODEL_SIZES)} (default: large-v3)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='使用するデバイス (default: auto)'
    )
    
    parser.add_argument(
        '--compute_type',
        choices=FasterWhisperTranscriber.COMPUTE_TYPES,
        default='default',
        help='計算精度 (default: default)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default='ja',
        help='音声の言語コード (例: ja, en) (default: ja)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='出力ファイルパス。指定しない場合は入力ファイル名を基に自動生成'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'srt', 'vtt'],
        default='text',
        help='出力形式 (default: text)'
    )
    
    parser.add_argument(
        '--timestamps',
        action='store_true',
        help='タイムスタンプを含める（text, json形式で有効）'
    )
    
    parser.add_argument(
        '--word_timestamps',
        action='store_true',
        help='単語レベルのタイムスタンプを生成（処理時間が増加）'
    )
    
    parser.add_argument(
        '--confidence',
        action='store_true',
        help='信頼度情報を含める（text, json形式で有効）'
    )
    
    parser.add_argument(
        '--task',
        choices=['transcribe', 'translate'],
        default='transcribe',
        help='実行するタスク。translateを指定すると英語に翻訳 (default: transcribe)'
    )
    
    parser.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='ビームサーチのビーム幅 (default: 5)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        nargs='+',
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help='サンプリング温度。複数指定すると順にフォールバックし、'
             'リピート等のハルシネーションから脱出しやすくなる '
             '(default: 0.0 0.2 0.4 0.6 0.8 1.0)'
    )
    
    parser.add_argument(
        '--vad_filter',
        action='store_true',
        default=True,
        help='Voice Activity Detectionフィルタを使用 (default: True)'
    )

    parser.add_argument(
        '--no_vad_filter',
        dest='vad_filter',
        action='store_false',
        help='Voice Activity Detectionフィルタを無効化'
    )

    parser.add_argument(
        '--vad_threshold',
        type=float,
        help='VAD閾値（0-1）。低いほど敏感（小さい音声を検出）。'
             '指定しない場合は事前スキャンで自動選択'
    )

    parser.add_argument(
        '--no_vad_auto_threshold',
        dest='vad_auto_threshold',
        action='store_false',
        help='VAD閾値の自動選択（事前スキャン）を無効化し、0.20を使用'
    )

    parser.add_argument(
        '--vad_min_speech_duration',
        type=int,
        help='最小音声長(ms)。これより短い音声は無視'
    )

    parser.add_argument(
        '--vad_max_speech_duration',
        type=float,
        help='最大音声長(s)。これより長い場合は分割'
    )

    parser.add_argument(
        '--vad_min_silence_duration',
        type=int,
        help='最小無音長(ms)。これより短い無音では分割しない'
    )

    parser.add_argument(
        '--vad_speech_pad',
        type=int,
        help='音声の前後パディング(ms)。音声の切れを防ぐ'
    )

    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='音声をピークノーマライズ（-3.0 dBFSに調整）(default: True)'
    )

    parser.add_argument(
        '--no_normalize',
        dest='normalize',
        action='store_false',
        help='音声のピークノーマライズを無効化'
    )

    parser.add_argument(
        '--gain',
        type=float,
        default=0,
        help='ゲイン調整(dB)。正の値で音量アップ、負の値で音量ダウン (default: 0)'
    )

    parser.add_argument(
        '--denoise',
        action='store_true',
        default=None,
        help='ノイズ除去を強制的に有効化。指定しない場合は音声のSN比を解析して自動判定'
    )

    parser.add_argument(
        '--no_denoise',
        dest='denoise',
        action='store_false',
        help='ノイズ除去を強制的に無効化（自動判定もしない）'
    )

    parser.add_argument(
        '--noise_reduce_amount',
        type=float,
        default=0.8,
        help='ノイズ除去の強度（0.0-1.0）。1.0が最大 (default: 0.8)'
    )

    parser.add_argument(
        '--no_speech_threshold',
        type=float,
        default=0.9,
        help='Whisper側の無音判定閾値。1.0に近づけると無音除外を緩和 (default: 0.9)'
    )

    parser.add_argument(
        '--initial_prompt',
        type=str,
        help='文字起こしの初期プロンプト（文脈を与える）'
    )

    parser.add_argument(
        '--hotwords',
        type=str,
        help='認識させたい固有名詞・専門用語をスペース区切りで指定（例: "Kubernetes 岡田 議事録"）'
    )

    parser.add_argument(
        '--no_condition_on_previous_text',
        dest='condition_on_previous_text',
        action='store_false',
        help='直前の認識結果を文脈に使わない。同じ文が繰り返し出力される場合に有効'
    )

    parser.add_argument(
        '--cpu_threads',
        type=int,
        default=0,
        help='CPU使用時のスレッド数（0で自動）(default: 0)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='並列ワーカー数 (default: 1)'
    )
    
    args = parser.parse_args()

    try:
        # 処理開始時刻を記録
        start_time = time.time()
        model_load_start = time.time()

        # 出力ファイルパスを決定
        if args.output:
            output_path = args.output
        else:
            # 拡張子を適切に設定
            ext_map = {'text': '.txt', 'json': '.json', 'srt': '.srt', 'vtt': '.vtt'}
            ext = ext_map[args.format]
            output_path = args.audio_file.with_suffix(ext)
        
        # エンジンの決定
        engine = resolve_engine(args.engine, args.model)
        if engine == 'whisper' and args.engine is None:
            print(f"--model {args.model} が指定されたため、faster-whisperを使用します。")
        if engine == 'qwen3':
            ignored = find_changed_options(
                {name: getattr(args, name) for name in WHISPER_ONLY_OPTIONS},
                {name: parser.get_default(name) for name in WHISPER_ONLY_OPTIONS},
            )
            if ignored:
                print(f"警告: 次のオプションはQwen3-ASRでは無視されます: {', '.join('--' + n for n in ignored)}")
                print("      これらを使う場合は --engine whisper を指定してください。")

        # 文字起こし実行
        if engine == 'qwen3':
            args.model = resolve_model(engine, args.model)
            transcriber = Qwen3ASRTranscriber(model_name=args.model)
            # Qwen3-ASRは無音区間を自前で処理するため、Whisper用のVADは使わない
            args.vad_filter = False
        else:
            args.model = resolve_model(engine, args.model)
            transcriber = FasterWhisperTranscriber(
                model_name=args.model,
                device=args.device,
                compute_type=args.compute_type,
                cpu_threads=args.cpu_threads,
                num_workers=args.num_workers
            )
        model_load_time = time.time() - model_load_start

        # 温度パラメータの処理
        temperature = args.temperature[0] if len(args.temperature) == 1 else args.temperature

        # VADパラメータの構築
        vad_parameters = None
        if args.vad_filter:
            vad_parameters = {}
            if args.vad_threshold is not None:
                vad_parameters['threshold'] = args.vad_threshold
            if args.vad_min_speech_duration is not None:
                vad_parameters['min_speech_duration_ms'] = args.vad_min_speech_duration
            if args.vad_max_speech_duration is not None:
                vad_parameters['max_speech_duration_s'] = args.vad_max_speech_duration
            if args.vad_min_silence_duration is not None:
                vad_parameters['min_silence_duration_ms'] = args.vad_min_silence_duration
            if args.vad_speech_pad is not None:
                vad_parameters['speech_pad_ms'] = args.vad_speech_pad
            # パラメータが何も指定されていない場合はNoneに戻す（デフォルト値を使用）
            if not vad_parameters:
                vad_parameters = None

        # 文字起こし実行
        transcribe_start = time.time()
        result = transcriber.transcribe(
            args.audio_file,
            language=args.language,
            task=args.task,
            beam_size=args.beam_size,
            temperature=temperature,
            initial_prompt=args.initial_prompt,
            hotwords=args.hotwords,
            condition_on_previous_text=args.condition_on_previous_text,
            word_timestamps=args.word_timestamps,
            vad_filter=args.vad_filter,
            vad_parameters=vad_parameters,
            vad_auto_threshold=args.vad_auto_threshold,
            normalize=args.normalize,
            gain_db=args.gain,
            denoise=args.denoise,
            noise_reduce_amount=args.noise_reduce_amount,
            no_speech_threshold=args.no_speech_threshold
        )
        transcribe_time = time.time() - transcribe_start

        # 結果を保存
        transcriber.save_result(
            result,
            output_path,
            output_format=args.format,
            include_timestamps=args.timestamps or args.format in ['srt', 'vtt'],
            include_confidence=args.confidence
        )
        
        # 簡単な統計情報を表示
        print("\n--- 統計情報 ---")
        print(f"文字数: {len(result['text'])} 文字")
        print(f"セグメント数: {len(result['segments'])}")
        
        if result.get('duration'):
            print(f"音声の長さ: {transcriber._format_timestamp(result['duration'])}")
            if result.get('duration_after_vad'):
                vad_reduction = (1 - result['duration_after_vad'] / result['duration']) * 100
                print(f"VAD後の長さ: {transcriber._format_timestamp(result['duration_after_vad'])} "
                      f"({vad_reduction:.1f}%削減)")
                if args.vad_filter and vad_reduction >= 90:
                    print("\n警告: VADによる削減率が90%を超えています。")
                    print("閾値の自動選択が有効な場合、録音の大半が実際に無音である可能性が高いですが、")
                    print("取りこぼしが疑われる場合は以下を試してください:")
                    print(f"  uv run python {Path(__file__).name} {args.audio_file} --engine whisper --model {args.model} --language {args.language or 'ja'} --no_vad_filter")
                    print("必要に応じて:")
                    print("  --vad_threshold 0.05")
                    print("  --vad_speech_pad 1000")
                    print("  --no_speech_threshold 0.9")

        if result.get('language_probability'):
            print(f"言語検出の信頼度: {result['language_probability']:.2%}")

        # 処理時間を表示
        total_time = time.time() - start_time
        print("\n--- 処理時間 ---")
        print(f"モデル読み込み: {model_load_time:.1f}秒")
        print(f"文字起こし: {transcribe_time:.1f}秒")
        print(f"合計: {total_time:.1f}秒")

        # 実時間比率を計算（音声の長さに対する処理時間の比率）
        if result.get('duration'):
            audio_duration = result['duration']
            realtime_ratio = transcribe_time / audio_duration
            print(f"実時間比率: {realtime_ratio:.2f}x （1.0未満 = リアルタイムより高速）")

    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

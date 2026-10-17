"""transcribe_faster_whisper モジュールのテスト"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from pydub import AudioSegment

from transcribe_faster_whisper import (
    FasterWhisperTranscriber,
    analyze_audio_levels,
    build_qwen_context,
    choose_vad_threshold,
    chunks_to_segments,
    compute_normalize_gain,
    find_changed_options,
    normalize_peak,
    recommend_preprocessing,
    resolve_engine,
    resolve_model,
)


class TestChooseVadThreshold:
    """VAD閾値の自動選択ロジックのテスト"""

    def test_detections_equal_picks_highest_threshold(self):
        """全候補の検出量が同等なら、最も無音を削れる高い閾値を選ぶ"""
        scan = [(0.50, 1000.0), (0.30, 1010.0), (0.05, 1020.0)]
        assert choose_vad_threshold(scan) == 0.50

    def test_quiet_speech_requires_low_threshold(self):
        """高い閾値で検出が大きく減る場合、カバレッジを満たす低い閾値を選ぶ"""
        # 実測データに近い形: 閾値を下げるほど小さい発話を拾う
        scan = [(0.50, 636.0), (0.30, 750.0), (0.15, 930.0), (0.05, 1044.0)]
        # coverage=0.95 → 991.8秒以上を満たすのは 0.05 のみ
        assert choose_vad_threshold(scan) == 0.05

    def test_custom_coverage_allows_higher_threshold(self):
        """coverage を緩めれば、より高い閾値が許容される"""
        scan = [(0.50, 636.0), (0.30, 750.0), (0.15, 930.0), (0.05, 1044.0)]
        # coverage=0.85 → 887.4秒以上を満たす最高閾値は 0.15
        assert choose_vad_threshold(scan, coverage=0.85) == 0.15

    def test_unsorted_input(self):
        """候補の並び順に依存しない"""
        scan = [(0.05, 1044.0), (0.50, 636.0), (0.15, 930.0), (0.30, 750.0)]
        assert choose_vad_threshold(scan, coverage=0.85) == 0.15

    def test_no_speech_detected_picks_highest_threshold(self):
        """どの閾値でも検出ゼロなら、最も高い閾値を返す（ゼロ除算しない）"""
        scan = [(0.50, 0.0), (0.30, 0.0), (0.05, 0.0)]
        assert choose_vad_threshold(scan) == 0.50

    def test_empty_scan_raises(self):
        """空のスキャン結果は ValueError"""
        with pytest.raises(ValueError):
            choose_vad_threshold([])

    def test_invalid_coverage_raises(self):
        """coverage が (0, 1] の範囲外なら ValueError"""
        scan = [(0.50, 100.0), (0.05, 200.0)]
        with pytest.raises(ValueError):
            choose_vad_threshold(scan, coverage=0.0)
        with pytest.raises(ValueError):
            choose_vad_threshold(scan, coverage=1.5)


class TestComputeNormalizeGain:
    """ピークノーマライズのゲイン計算のテスト"""

    def test_gain_brings_peak_to_target(self):
        """ピーク -12.4 dBFS を -3 dBFS に揃えるゲインは +9.4 dB"""
        assert compute_normalize_gain(-12.4) == pytest.approx(9.4)

    def test_loud_audio_is_attenuated(self):
        """ピークが目標より大きければ負のゲイン（減衰）になる"""
        assert compute_normalize_gain(-1.0) == pytest.approx(-2.0)

    def test_silent_audio_gets_no_gain(self):
        """完全な無音（ピーク -inf）では無限大のゲインをかけない"""
        assert compute_normalize_gain(float('-inf')) == 0.0

    def test_custom_target(self):
        """目標ピークを変更できる"""
        assert compute_normalize_gain(-20.0, target_peak_dbfs=-6.0) == pytest.approx(14.0)


class TestNormalizePeak:
    """AudioSegment へのピークノーマライズ適用のテスト"""

    @staticmethod
    def _speech_like_segment():
        """音声に近いクレストファクタ（ピークとRMSの差が大きい）の信号"""
        rng = np.random.default_rng(0)
        x = np.clip(rng.laplace(scale=0.02, size=16000 * 2), -1, 1)
        return AudioSegment(
            (x * 32767).astype(np.int16).tobytes(),
            frame_rate=16000, sample_width=2, channels=1,
        )

    def test_peak_reaches_target_without_clipping(self):
        """平均音量ではなくピークを基準にするため、クリップしない"""
        normalized = normalize_peak(self._speech_like_segment())
        samples = np.array(normalized.get_array_of_samples())
        assert normalized.max_dBFS == pytest.approx(-3.0, abs=0.1)
        assert np.mean(np.abs(samples) >= 32767) == 0.0

    def test_silent_segment_is_unchanged(self):
        """無音の音声はエラーにならずそのまま返る"""
        silent = AudioSegment.silent(duration=500, frame_rate=16000)
        normalized = normalize_peak(silent)
        assert normalized.raw_data == silent.raw_data


def _tone_with_noise(speech_amp: float, noise_amp: float, seconds: int = 4) -> np.ndarray:
    """前半は雑音のみ、後半は雑音+発話（正弦波）の 16kHz 信号を作る"""
    sr = 16000
    rng = np.random.default_rng(1)
    noise = rng.normal(scale=noise_amp, size=sr * seconds)
    t = np.arange(sr * seconds // 2) / sr
    speech = np.concatenate([np.zeros(sr * seconds // 2), speech_amp * np.sin(2 * np.pi * 220 * t)])
    return np.clip(noise + speech, -1, 1).astype(np.float32)


class TestAnalyzeAudioLevels:
    """音声レベル解析のテスト"""

    def test_clean_recording_has_high_snr(self):
        """静かな環境の録音は SN 比が高い"""
        stats = analyze_audio_levels(_tone_with_noise(0.3, 0.001), 16000)
        assert stats['snr_db'] > 30
        assert stats['clipping_ratio'] == 0.0

    def test_noisy_recording_has_low_snr(self):
        """雑音が大きい録音は SN 比が低い"""
        stats = analyze_audio_levels(_tone_with_noise(0.1, 0.05), 16000)
        assert stats['snr_db'] < 10

    def test_clipping_is_detected(self):
        """振幅が上限に張り付いたサンプルをクリップとして数える"""
        audio = np.zeros(16000, dtype=np.float32)
        audio[:160] = 1.0
        stats = analyze_audio_levels(audio, 16000)
        assert stats['clipping_ratio'] == pytest.approx(0.01)

    def test_silent_audio_does_not_raise(self):
        """完全な無音でもエラーにならない"""
        stats = analyze_audio_levels(np.zeros(16000, dtype=np.float32), 16000)
        assert stats['peak_dbfs'] == float('-inf')
        assert stats['snr_db'] == 0.0

    def test_empty_audio_raises(self):
        """空の音声は ValueError"""
        with pytest.raises(ValueError):
            analyze_audio_levels(np.zeros(0, dtype=np.float32), 16000)


class TestRecommendPreprocessing:
    """解析結果から前処理パラメータを決めるロジックのテスト"""

    @staticmethod
    def _stats(**overrides):
        stats = {
            'peak_dbfs': -6.0, 'rms_dbfs': -25.0, 'noise_floor_dbfs': -60.0,
            'speech_level_dbfs': -20.0, 'snr_db': 40.0, 'clipping_ratio': 0.0,
        }
        stats.update(overrides)
        return stats

    def test_clean_audio_skips_denoise(self):
        """SN 比が十分なら、認識を劣化させうるノイズ除去はかけない"""
        rec = recommend_preprocessing(self._stats())
        assert rec['denoise'] is False
        assert rec['warnings'] == []

    def test_noisy_audio_enables_mild_denoise(self):
        """SN 比が低いときは弱めのノイズ除去をかける"""
        rec = recommend_preprocessing(self._stats(snr_db=8.0))
        assert rec['denoise'] is True
        assert 0 < rec['noise_reduce_amount'] <= 0.6

    def test_very_noisy_audio_uses_stronger_denoise(self):
        """SN 比が極端に低いときは強めにかける"""
        mild = recommend_preprocessing(self._stats(snr_db=8.0))
        strong = recommend_preprocessing(self._stats(snr_db=3.0))
        assert strong['noise_reduce_amount'] > mild['noise_reduce_amount']

    def test_clipped_source_warns(self):
        """元音声がクリップしている場合は警告する"""
        rec = recommend_preprocessing(self._stats(clipping_ratio=0.01))
        assert any('クリップ' in w for w in rec['warnings'])

    def test_quiet_recording_warns(self):
        """発話レベルが極端に小さい場合は警告する"""
        rec = recommend_preprocessing(self._stats(speech_level_dbfs=-50.0))
        assert any('小さ' in w for w in rec['warnings'])


class TestChunksToSegments:
    """Qwen3-ASR のチャンクを共通のセグメント形式に変換するテスト"""

    def test_converts_chunks_with_ids(self):
        chunks = [
            {'text': 'こんにちは。', 'start': 0.0, 'end': 29.5, 'chunk_index': 0},
            {'text': '今日は晴れ。', 'start': 29.5, 'end': 55.0, 'chunk_index': 1},
        ]
        segments = chunks_to_segments(chunks)
        assert [s['id'] for s in segments] == [1, 2]
        assert segments[1]['start'] == 29.5
        assert segments[1]['end'] == 55.0
        assert segments[0]['text'] == 'こんにちは。'

    def test_skips_empty_chunks(self):
        """無音チャンクの空文字は字幕に出さない"""
        chunks = [
            {'text': '  ', 'start': 0.0, 'end': 30.0},
            {'text': 'はい。', 'start': 30.0, 'end': 35.0},
        ]
        segments = chunks_to_segments(chunks)
        assert len(segments) == 1
        assert segments[0]['id'] == 1

    def test_none_chunks_returns_empty(self):
        assert chunks_to_segments(None) == []


class TestBuildQwenContext:
    """initial_prompt / hotwords を Qwen3-ASR の context に変換するテスト"""

    def test_joins_prompt_and_hotwords(self):
        assert build_qwen_context('会議の録音', 'Kubernetes 岡田') == '会議の録音 Kubernetes 岡田'

    def test_none_values_give_empty_context(self):
        assert build_qwen_context(None, None) == ''

    def test_only_hotwords(self):
        assert build_qwen_context(None, 'Kubernetes') == 'Kubernetes'


class TestSaveResultWithoutConfidence:
    """信頼度情報のないセグメント（Qwen3-ASR）でも保存できることのテスト"""

    def test_text_with_confidence_does_not_crash(self, tmp_path):
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model_name, transcriber.device = 'qwen', 'mlx'
        result = {
            'text': 'はい。',
            'segments': [{'id': 1, 'start': 0.0, 'end': 1.0, 'text': 'はい。'}],
        }
        out = tmp_path / 'out.txt'
        transcriber.save_result(result, out, include_timestamps=True, include_confidence=True)
        assert out.read_text(encoding='utf-8') == '[00:00:00.00 --> 00:00:01.00] はい。\n'

    def test_json_with_confidence_does_not_crash(self, tmp_path):
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model_name, transcriber.device = 'qwen', 'mlx'
        result = {
            'text': 'はい。',
            'segments': [{'id': 1, 'start': 0.0, 'end': 1.0, 'text': 'はい。'}],
        }
        out = tmp_path / 'out.json'
        transcriber.save_result(
            result, out, output_format='json', include_timestamps=True, include_confidence=True
        )
        assert 'no_speech_prob' not in out.read_text(encoding='utf-8')


class TestResolveEngine:
    """--engine 省略時のエンジン決定（後方互換）のテスト"""

    def test_default_is_qwen3(self):
        assert resolve_engine(None, None) == 'qwen3'

    def test_whisper_model_name_selects_whisper(self):
        """従来の --model large-v3 だけのコマンドは whisper で動く"""
        assert resolve_engine(None, 'large-v3') == 'whisper'

    def test_qwen_model_id_selects_qwen3(self):
        assert resolve_engine(None, 'Qwen/Qwen3-ASR-0.6B') == 'qwen3'

    def test_explicit_engine_wins(self):
        assert resolve_engine('whisper', None) == 'whisper'
        assert resolve_engine('qwen3', 'Qwen/Qwen3-ASR-1.7B') == 'qwen3'


class TestFindChangedOptions:
    """デフォルトから変更されたオプションの検出テスト"""

    def test_returns_only_changed_names(self):
        values = {'beam_size': 10, 'temperature': [0.0], 'device': 'auto'}
        defaults = {'beam_size': 5, 'temperature': [0.0], 'device': 'auto'}
        assert find_changed_options(values, defaults) == ['beam_size']

    def test_nothing_changed(self):
        assert find_changed_options({'beam_size': 5}, {'beam_size': 5}) == []


class TestResolveModel:
    """エンジンごとのデフォルトモデル決定のテスト"""

    def test_whisper_default_is_large_v3(self):
        assert resolve_model('whisper', None) == 'large-v3'

    def test_qwen3_default(self):
        assert resolve_model('qwen3', None) == 'Qwen/Qwen3-ASR-1.7B'

    def test_explicit_model_wins(self):
        assert resolve_model('whisper', 'medium') == 'medium'

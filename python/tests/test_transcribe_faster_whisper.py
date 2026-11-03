"""transcribe_faster_whisper モジュールのテスト"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from pydub import AudioSegment

from transcribe_faster_whisper import (
    FasterWhisperTranscriber,
    Qwen3ASRTranscriber,
    analyze_audio_levels,
    build_ffmpeg_extract_command,
    build_speech_enhance_command,
    build_qwen_context,
    choose_vad_threshold,
    chunks_to_segments,
    compute_normalize_gain,
    find_changed_options,
    normalize_peak,
    recommend_preprocessing,
    resolve_denoise_plan,
    resolve_engine,
    resolve_diarization,
    resolve_model,
    speaker_segments_to_segments,
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


class TestSpeakerSegmentsToSegments:
    """Qwen3-ASR の speaker_segments を共通のセグメント形式に変換するテスト"""

    def test_converts_with_speaker_and_ids(self):
        speaker_segments = [
            {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 2.5, 'text': 'こんにちは。'},
            {'speaker': 'SPEAKER_01', 'start': 2.5, 'end': 4.0, 'text': 'どうも。'},
        ]
        segments = speaker_segments_to_segments(speaker_segments)
        assert segments == [
            {'id': 1, 'start': 0.0, 'end': 2.5, 'text': 'こんにちは。', 'speaker': 'SPEAKER_00'},
            {'id': 2, 'start': 2.5, 'end': 4.0, 'text': 'どうも。', 'speaker': 'SPEAKER_01'},
        ]

    def test_skips_turns_without_text(self):
        """話者区間はあっても文字のない区間（咳・相づち等）は出力しない"""
        speaker_segments = [
            {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 1.0, 'text': ''},
            {'speaker': 'SPEAKER_01', 'start': 1.0, 'end': 2.0, 'text': 'はい。'},
        ]
        segments = speaker_segments_to_segments(speaker_segments)
        assert len(segments) == 1
        assert segments[0]['id'] == 1
        assert segments[0]['speaker'] == 'SPEAKER_01'

    def test_none_returns_empty(self):
        assert speaker_segments_to_segments(None) == []


def _speaker_result():
    return {
        'text': 'こんにちは。どうも。',
        'segments': [
            {'id': 1, 'start': 0.0, 'end': 1.0, 'text': 'こんにちは。', 'speaker': 'SPEAKER_00'},
            {'id': 2, 'start': 1.0, 'end': 2.0, 'text': 'どうも。', 'speaker': 'SPEAKER_01'},
        ],
    }


def _bare_transcriber():
    transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
    transcriber.model_name, transcriber.device = 'qwen', 'mlx'
    return transcriber


class TestSaveResultWithSpeaker:
    """話者ラベル付きセグメントの保存テスト"""

    def test_text_with_timestamps_has_speaker(self, tmp_path):
        out = tmp_path / 'out.txt'
        _bare_transcriber().save_result(_speaker_result(), out, include_timestamps=True)
        assert out.read_text(encoding='utf-8') == (
            '[00:00:00.00 --> 00:00:01.00] SPEAKER_00: こんにちは。\n'
            '[00:00:01.00 --> 00:00:02.00] SPEAKER_01: どうも。\n'
        )

    def test_text_without_timestamps_is_one_line_per_turn(self, tmp_path):
        """話者分離時はタイムスタンプなしでも発話ごとに改行し、話者を付ける"""
        out = tmp_path / 'out.txt'
        _bare_transcriber().save_result(_speaker_result(), out)
        assert out.read_text(encoding='utf-8') == (
            'SPEAKER_00: こんにちは。\n'
            'SPEAKER_01: どうも。\n'
        )

    def test_json_includes_speaker(self, tmp_path):
        import json
        out = tmp_path / 'out.json'
        _bare_transcriber().save_result(
            _speaker_result(), out, output_format='json', include_timestamps=True
        )
        data = json.loads(out.read_text(encoding='utf-8'))
        assert [s['speaker'] for s in data['segments']] == ['SPEAKER_00', 'SPEAKER_01']

    def test_srt_prefixes_speaker(self, tmp_path):
        out = tmp_path / 'out.srt'
        _bare_transcriber().save_result(
            _speaker_result(), out, output_format='srt', include_timestamps=True
        )
        assert '[SPEAKER_00] こんにちは。' in out.read_text(encoding='utf-8')

    def test_vtt_uses_voice_tag(self, tmp_path):
        out = tmp_path / 'out.vtt'
        _bare_transcriber().save_result(
            _speaker_result(), out, output_format='vtt', include_timestamps=True
        )
        assert '<v SPEAKER_01>どうも。' in out.read_text(encoding='utf-8')

    def test_without_speaker_output_is_unchanged(self, tmp_path):
        """話者分離なしの結果は従来どおりの出力（後方互換）"""
        out = tmp_path / 'out.txt'
        result = {'text': ' はい。 ', 'segments': [{'id': 1, 'start': 0.0, 'end': 1.0, 'text': 'はい。'}]}
        _bare_transcriber().save_result(result, out)
        assert out.read_text(encoding='utf-8') == 'はい。'


class TestResolveDiarization:
    """--diarize 関連オプションから話者分離設定を決めるテスト"""

    def test_disabled_returns_none(self):
        assert resolve_diarization('qwen3', diarize=False) is None

    def test_enabled_for_qwen3(self):
        assert resolve_diarization('qwen3', diarize=True, num_speakers=2) == {
            'num_speakers': 2, 'min_speakers': 1, 'max_speakers': 8,
        }

    def test_whisper_raises(self):
        with pytest.raises(ValueError, match='Qwen3-ASR'):
            resolve_diarization('whisper', diarize=True)

    def test_speaker_options_without_diarize_raise(self):
        """--diarize なしで話者数だけ指定されたら気づけるようにする"""
        with pytest.raises(ValueError, match='--diarize'):
            resolve_diarization('qwen3', diarize=False, num_speakers=2)

    @pytest.mark.parametrize('kwargs', [
        {'num_speakers': 0},
        {'min_speakers': 0},
        {'min_speakers': 3, 'max_speakers': 2},
    ])
    def test_invalid_counts_raise(self, kwargs):
        with pytest.raises(ValueError):
            resolve_diarization('qwen3', diarize=True, **kwargs)


class _FakeQwenResult:
    def __init__(self, speaker_segments=None):
        self.text = 'こんにちは。どうも。'
        self.language = 'Japanese'
        self.truncated = False
        self.finish_reason = 'stop'
        self.chunks = [{'text': 'こんにちは。どうも。', 'start': 0.0, 'end': 30.0}]
        self.speaker_segments = speaker_segments


class _FakeSession:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        return self.result


def _qwen_with(session, diarization=None):
    transcriber = Qwen3ASRTranscriber.__new__(Qwen3ASRTranscriber)
    transcriber.session = session
    transcriber.diarization = diarization
    transcriber.model_name, transcriber.device = 'qwen', 'mlx'
    return transcriber


class TestQwen3Diarization:
    """Qwen3ASRTranscriber の話者分離の受け渡しテスト"""

    def test_default_does_not_diarize(self, tmp_path):
        session = _FakeSession(_FakeQwenResult())
        result = _qwen_with(session)._run_model(tmp_path / 'a.wav')
        assert 'diarize' not in session.kwargs
        assert 'speaker' not in result['segments'][0]

    def test_diarize_passes_options_and_uses_speaker_segments(self, tmp_path):
        session = _FakeSession(_FakeQwenResult(speaker_segments=[
            {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 1.5, 'text': 'こんにちは。'},
            {'speaker': 'SPEAKER_01', 'start': 1.5, 'end': 3.0, 'text': 'どうも。'},
        ]))
        diarization = {'num_speakers': 2, 'min_speakers': 1, 'max_speakers': 8}
        result = _qwen_with(session, diarization)._run_model(tmp_path / 'a.wav')
        assert session.kwargs['diarize'] is True
        assert session.kwargs['diarization_num_speakers'] == 2
        assert session.kwargs['diarization_min_speakers'] == 1
        assert session.kwargs['diarization_max_speakers'] == 8
        assert [s['speaker'] for s in result['segments']] == ['SPEAKER_00', 'SPEAKER_01']
        assert result['duration'] == 30.0


class TestBuildFfmpegExtractCommand:
    """ffmpeg による音声抽出コマンド組み立てのテスト"""

    def test_default_resamples_to_16khz(self):
        cmd = build_ffmpeg_extract_command(Path('in.mp4'), Path('out.wav'))
        assert cmd[:3] == ['ffmpeg', '-i', 'in.mp4']
        assert cmd[cmd.index('-ar') + 1] == '16000'
        assert cmd[cmd.index('-ac') + 1] == '1'
        assert cmd[-1] == 'out.wav'

    def test_none_sample_rate_keeps_original_rate(self):
        cmd = build_ffmpeg_extract_command(Path('in.mp4'), Path('out.wav'), sample_rate=None)
        assert '-ar' not in cmd
        assert cmd[cmd.index('-ac') + 1] == '1'


class TestBuildSpeechEnhanceCommand:
    """speech.cpp の RE-USE 呼び出しコマンド組み立てのテスト"""

    def test_builds_re_use_command(self):
        cmd = build_speech_enhance_command(Path('noisy.wav'), Path('clean.wav'), 're-use')
        assert cmd == ['speech', 'enhance', 're-use', 'noisy.wav', '-o', 'clean.wav']

    def test_builds_laughter_model_command(self):
        cmd = build_speech_enhance_command(Path('a.wav'), Path('b.wav'), 're-use-laughter')
        assert cmd[2] == 're-use-laughter'


class TestResolveDenoisePlan:
    """ノイズ除去エンジンの選択ロジックのテスト（既定は既存挙動を維持）"""

    @pytest.mark.parametrize('denoise', [None, True, False])
    def test_noisereduce_keeps_existing_behavior(self, denoise):
        assert resolve_denoise_plan(denoise, 'noisereduce') == (False, denoise)

    @pytest.mark.parametrize('denoise', [None, True])
    def test_re_use_runs_dnn_and_disables_noisereduce(self, denoise):
        assert resolve_denoise_plan(denoise, 're-use') == (True, False)

    def test_no_denoise_wins_over_re_use(self):
        assert resolve_denoise_plan(False, 're-use') == (False, False)

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError):
            resolve_denoise_plan(None, 'unknown')


class TestEnhanceWithSpeechCpp:
    """speech.cpp による音声強調の実行テスト（subprocess はモック）"""

    def test_converts_then_enhances(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, '', '')

        monkeypatch.setattr(subprocess, 'run', fake_run)
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        out = transcriber.enhance_with_speech_cpp(tmp_path / 'in.m4a', tmp_path, 're-use')

        assert calls[0][0] == 'ffmpeg' and '-ar' not in calls[0]
        assert calls[1][:3] == ['speech', 'enhance', 're-use']
        assert calls[1][3] == calls[0][-1]
        assert calls[1][-1] == str(out)
        assert out.parent == tmp_path

    def test_missing_speech_command_raises_with_install_hint(self, tmp_path, monkeypatch):
        def fake_run(cmd, **kwargs):
            if cmd[0] == 'speech':
                raise FileNotFoundError(cmd[0])
            return subprocess.CompletedProcess(cmd, 0, '', '')

        monkeypatch.setattr(subprocess, 'run', fake_run)
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        with pytest.raises(RuntimeError, match='speech.cpp'):
            transcriber.enhance_with_speech_cpp(tmp_path / 'in.wav', tmp_path, 're-use')

    def test_speech_failure_raises_with_stderr(self, tmp_path, monkeypatch):
        def fake_run(cmd, **kwargs):
            if cmd[0] == 'speech':
                raise subprocess.CalledProcessError(1, cmd, stderr='model error')
            return subprocess.CompletedProcess(cmd, 0, '', '')

        monkeypatch.setattr(subprocess, 'run', fake_run)
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        with pytest.raises(RuntimeError, match='model error'):
            transcriber.enhance_with_speech_cpp(tmp_path / 'in.wav', tmp_path, 're-use')

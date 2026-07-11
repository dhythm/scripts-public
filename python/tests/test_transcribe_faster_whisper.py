"""transcribe_faster_whisper モジュールのテスト"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from transcribe_faster_whisper import choose_vad_threshold


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

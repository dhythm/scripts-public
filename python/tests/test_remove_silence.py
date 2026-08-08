from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from remove_silence import (
    default_output_path,
    find_speech_ranges,
    join_ranges,
    output_format,
)


def test_find_speech_ranges_removes_long_silence_and_keeps_padding() -> None:
    tone = Sine(440).to_audio_segment(duration=500).apply_gain(-10)
    audio = AudioSegment.silent(1000) + tone + AudioSegment.silent(2000) + tone

    ranges = find_speech_ranges(
        audio,
        min_silence_ms=800,
        silence_threshold_dbfs=-40,
        keep_silence_ms=200,
        min_speech_ms=100,
    )

    assert ranges == [(800, 1700), (3300, 4000)]
    result = join_ranges(audio, ranges, fade_ms=10)
    assert len(result) == 1590


def test_short_noise_is_ignored() -> None:
    short_noise = Sine(440).to_audio_segment(duration=80).apply_gain(-10)
    audio = AudioSegment.silent(500) + short_noise + AudioSegment.silent(500)

    ranges = find_speech_ranges(
        audio,
        min_silence_ms=200,
        silence_threshold_dbfs=-40,
        keep_silence_ms=100,
        min_speech_ms=150,
    )

    assert ranges == []


def test_overlapping_padding_is_merged() -> None:
    tone = Sine(440).to_audio_segment(duration=300).apply_gain(-10)
    audio = tone + AudioSegment.silent(400) + tone

    ranges = find_speech_ranges(
        audio,
        min_silence_ms=300,
        silence_threshold_dbfs=-40,
        keep_silence_ms=250,
        min_speech_ms=100,
    )

    assert ranges == [(0, 1000)]


def test_output_helpers() -> None:
    assert default_output_path(Path("voice.m4a")) == Path("voice_speech.m4a")
    assert output_format(Path("voice.m4a")) == "mp4"
    assert output_format(Path("voice.wav")) == "wav"
    with pytest.raises(ValueError, match="未対応"):
        output_format(Path("voice.txt"))

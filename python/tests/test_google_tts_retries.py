import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from google.api_core.exceptions import InvalidArgument, ServiceUnavailable, PermissionDenied
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import google_tts as tts


@pytest.mark.parametrize("attempts", [0, -1])
def test_invalid_attempt_count_never_calls_api(monkeypatch, attempts):
    api = Mock()
    monkeypatch.setattr(tts, "synthesize_speech", api)
    with pytest.raises(ValueError, match="1以上"):
        tts.synthesize_long_speech("text", Path("unused.mp3"), max_retries=attempts)
    api.assert_not_called()


@pytest.mark.parametrize("error", [InvalidArgument("invalid"), PermissionDenied("denied")])
def test_permanent_error_never_retried(monkeypatch, error):
    api = Mock(side_effect=error)
    delay = Mock()
    monkeypatch.setattr(tts, "synthesize_speech", api)
    monkeypatch.setattr(tts.time, "sleep", delay)
    with pytest.raises(type(error)):
        tts.synthesize_long_speech("short", Path("unused.mp3"))
    assert api.call_count == 1
    delay.assert_not_called()


def test_short_text_uses_retry_policy(monkeypatch):
    api = Mock(side_effect=[ServiceUnavailable("temporary"), Path("mock.mp3")])
    delay = Mock()
    monkeypatch.setattr(tts, "synthesize_speech", api)
    monkeypatch.setattr(tts.time, "sleep", delay)
    assert tts.synthesize_long_speech("short", Path("mock.mp3"), retry_delay=0) == Path("mock.mp3")
    assert api.call_count == 2
    delay.assert_called_once_with(0)


def test_multichunk_stops_before_merge_on_permanent_error(monkeypatch):
    api = Mock(side_effect=InvalidArgument("invalid"))
    merge = Mock()
    monkeypatch.setattr(tts, "split_text_for_tts", lambda *a, **kw: ["first", "second"])
    monkeypatch.setattr(tts, "synthesize_speech", api)
    monkeypatch.setattr(tts, "merge_audio_files", merge)
    with pytest.raises(InvalidArgument):
        tts.synthesize_long_speech("long", Path("unused.mp3"), show_progress=False)
    assert api.call_count == 1
    merge.assert_not_called()


def test_cli_rejects_zero_before_credentials(monkeypatch):
    credentials = Mock()
    monkeypatch.setattr(tts, "_ensure_credentials", credentials)
    result = CliRunner().invoke(tts.app, ["--text", "hello", "--max-retries", "0"])
    assert result.exit_code == 2
    credentials.assert_not_called()

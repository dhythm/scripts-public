"""open_voice_converter のテスト"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestValidation:
    def test_validate_audio_file_valid(self, tmp_path):
        from open_voice_converter import OpenVoiceConverter

        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

        converter = OpenVoiceConverter(openvoice_path=tmp_path, checkpoint_dir=tmp_path)
        converter.validate_audio_file(wav_file)

    def test_validate_audio_file_not_found(self, tmp_path):
        from open_voice_converter import OpenVoiceConverter

        converter = OpenVoiceConverter(openvoice_path=tmp_path, checkpoint_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="ファイルが見つかりません"):
            converter.validate_audio_file(tmp_path / "missing.wav")

    def test_validate_installation_success(self, tmp_path):
        from open_voice_converter import OpenVoiceConverter

        (tmp_path / "openvoice").mkdir()
        checkpoint_dir = tmp_path / "checkpoints_v2" / "converter"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "checkpoint.pth").write_bytes(b"dummy")

        converter = OpenVoiceConverter(openvoice_path=tmp_path, checkpoint_dir=checkpoint_dir)
        converter.validate_installation()

    def test_validate_installation_missing_checkpoint(self, tmp_path):
        from open_voice_converter import OpenVoiceConverter

        (tmp_path / "openvoice").mkdir()
        checkpoint_dir = tmp_path / "checkpoints_v2" / "converter"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "config.json").write_text("{}")

        converter = OpenVoiceConverter(openvoice_path=tmp_path, checkpoint_dir=checkpoint_dir)
        with pytest.raises(RuntimeError, match="checkpoint.pth"):
            converter.validate_installation()


class TestPrepareAudio:
    def test_prepare_audio_exports_wav(self, tmp_path):
        from pydub.generators import Sine

        from open_voice_converter import OpenVoiceConverter

        source = tmp_path / "input.mp3"
        Sine(440).to_audio_segment(duration=400).export(str(source), format="mp3")

        converter = OpenVoiceConverter(openvoice_path=tmp_path, checkpoint_dir=tmp_path)
        prepared = converter.prepare_audio(source, normalize=True, gain_db=3.0)

        assert prepared.exists()
        assert prepared.suffix == ".wav"


class TestConvertVoice:
    def test_build_conversion_payload(self, tmp_path):
        from open_voice_converter import OpenVoiceConverter

        converter = OpenVoiceConverter(
            openvoice_path=tmp_path / "openvoice",
            checkpoint_dir=tmp_path / "checkpoints",
            device="cpu",
            tau=0.45,
            enable_watermark=True,
            use_vad=False,
        )

        payload = converter.build_conversion_payload(
            tmp_path / "source.wav",
            tmp_path / "ref.wav",
            tmp_path / "out.wav",
        )

        assert payload["device"] == "cpu"
        assert payload["tau"] == 0.45
        assert payload["enable_watermark"] is True
        assert payload["use_vad"] is False

    def test_convert_voice_runs_subprocess(self, tmp_path):
        from pydub.generators import Sine

        from open_voice_converter import OpenVoiceConverter

        openvoice_path = tmp_path / "openvoice_repo"
        (openvoice_path / "openvoice").mkdir(parents=True)
        venv_bin = openvoice_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("#!/bin/sh\n")
        (venv_bin / "python").chmod(0o755)

        checkpoint_dir = tmp_path / "checkpoints_v2" / "converter"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "checkpoint.pth").write_bytes(b"dummy")

        source = tmp_path / "source.wav"
        reference = tmp_path / "reference.wav"
        output = tmp_path / "output.wav"
        audio = Sine(440).to_audio_segment(duration=500)
        audio.export(str(source), format="wav")
        audio.export(str(reference), format="wav")

        converter = OpenVoiceConverter(
            openvoice_path=openvoice_path,
            checkpoint_dir=checkpoint_dir,
            device="cpu",
        )

        with patch("subprocess.run") as mock_run:
            def side_effect(cmd, **kwargs):
                payload = Path(cmd[2])
                data = __import__("json").loads(payload.read_text(encoding="utf-8"))
                Path(data["output"]).write_bytes(b"RIFF" + b"\x00" * 100)
                return MagicMock(returncode=0)

            mock_run.side_effect = side_effect

            result = converter.convert_voice(source, reference, output)

            assert result == output
            assert output.exists()
            mock_run.assert_called_once()

    def test_convert_voice_subprocess_error(self, tmp_path):
        import subprocess

        from pydub.generators import Sine

        from open_voice_converter import OpenVoiceConverter

        openvoice_path = tmp_path / "openvoice_repo"
        (openvoice_path / "openvoice").mkdir(parents=True)
        venv_bin = openvoice_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("#!/bin/sh\n")
        (venv_bin / "python").chmod(0o755)

        checkpoint_dir = tmp_path / "checkpoints_v2" / "converter"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "checkpoint.pth").write_bytes(b"dummy")

        source = tmp_path / "source.wav"
        reference = tmp_path / "reference.wav"
        Sine(440).to_audio_segment(duration=500).export(str(source), format="wav")
        Sine(440).to_audio_segment(duration=500).export(str(reference), format="wav")

        converter = OpenVoiceConverter(
            openvoice_path=openvoice_path,
            checkpoint_dir=checkpoint_dir,
            device="cpu",
        )

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "python", stderr="boom")

            with pytest.raises(RuntimeError, match="OpenVoice の変換に失敗"):
                converter.convert_voice(source, reference, tmp_path / "out.wav")

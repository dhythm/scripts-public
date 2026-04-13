"""voice_convert モジュールのテスト"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestVoiceConverterValidation:
    """音声ファイルバリデーションのテスト"""

    def test_validate_audio_file_valid(self, tmp_path):
        """有効な音声ファイルは例外を投げない"""
        from voice_convert import VoiceConverter

        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"RIFF" + b"\x00" * 100)

        converter = VoiceConverter(seed_vc_path=tmp_path)
        converter.validate_audio_file(wav_file)

    def test_validate_audio_file_not_found(self, tmp_path):
        """存在しないファイルは FileNotFoundError"""
        from voice_convert import VoiceConverter

        converter = VoiceConverter(seed_vc_path=tmp_path)
        with pytest.raises(FileNotFoundError, match="ファイルが見つかりません"):
            converter.validate_audio_file(tmp_path / "nonexistent.wav")

    def test_validate_audio_file_not_a_file(self, tmp_path):
        """ディレクトリを渡すと ValueError"""
        from voice_convert import VoiceConverter

        converter = VoiceConverter(seed_vc_path=tmp_path)
        with pytest.raises(ValueError, match="ファイルではありません"):
            converter.validate_audio_file(tmp_path)

    def test_validate_audio_file_unsupported_format(self, tmp_path):
        """未対応形式は ValueError"""
        from voice_convert import VoiceConverter

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")

        converter = VoiceConverter(seed_vc_path=tmp_path)
        with pytest.raises(ValueError, match="サポートされていないファイル形式"):
            converter.validate_audio_file(txt_file)


class TestVoiceConverterInstallation:
    """Seed-VC インストール検証のテスト"""

    def test_validate_installation_missing(self, tmp_path):
        """Seed-VC が未インストールの場合は RuntimeError"""
        from voice_convert import VoiceConverter

        converter = VoiceConverter(seed_vc_path=tmp_path / "nonexistent")
        with pytest.raises(RuntimeError, match="Seed-VC"):
            converter.validate_installation()

    def test_validate_installation_no_inference_py(self, tmp_path):
        """inference.py が存在しない場合は RuntimeError"""
        from voice_convert import VoiceConverter

        converter = VoiceConverter(seed_vc_path=tmp_path)
        with pytest.raises(RuntimeError, match="inference.py"):
            converter.validate_installation()

    def test_validate_installation_success(self, tmp_path):
        """inference.py が存在すれば例外なし"""
        from voice_convert import VoiceConverter

        (tmp_path / "inference.py").write_text("# dummy")
        converter = VoiceConverter(seed_vc_path=tmp_path)
        converter.validate_installation()


class TestWavConversion:
    """WAV 変換処理のテスト"""

    def test_convert_to_wav_from_non_wav(self, tmp_path):
        """非 WAV ファイルが WAV に変換される"""
        from pydub.generators import Sine

        from voice_convert import VoiceConverter

        # テスト用 MP3 作成
        audio = Sine(440).to_audio_segment(duration=500)
        mp3_path = tmp_path / "test.mp3"
        audio.export(str(mp3_path), format="mp3")

        converter = VoiceConverter(seed_vc_path=tmp_path)
        wav_path = converter.convert_to_wav(mp3_path)

        assert wav_path.suffix == ".wav"
        assert wav_path.exists()

    def test_convert_to_wav_already_wav(self, tmp_path):
        """WAV ファイルは前処理不要ならそのまま返す"""
        from pydub.generators import Sine

        from voice_convert import VoiceConverter

        audio = Sine(440).to_audio_segment(duration=500)
        wav_path = tmp_path / "test.wav"
        audio.export(str(wav_path), format="wav")

        converter = VoiceConverter(seed_vc_path=tmp_path)
        result = converter.convert_to_wav(wav_path)

        assert result == wav_path

    def test_convert_to_wav_with_normalize(self, tmp_path):
        """ノーマライズ付き変換"""
        from pydub.generators import Sine

        from voice_convert import VoiceConverter

        audio = Sine(440).to_audio_segment(duration=500)
        wav_path = tmp_path / "test.wav"
        audio.export(str(wav_path), format="wav")

        converter = VoiceConverter(seed_vc_path=tmp_path)
        result = converter.convert_to_wav(wav_path, normalize=True)

        assert result.exists()
        assert result != wav_path  # 前処理ありなので別ファイル


class TestDeviceDetection:
    """デバイス自動検出のテスト"""

    def test_device_auto_cuda(self):
        """CUDA 利用可能時は cuda を選択"""
        from voice_convert import VoiceConverter

        with patch("voice_convert.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            converter = VoiceConverter(seed_vc_path=Path("/tmp"), device="auto")
            assert converter.device == "cuda"

    def test_device_auto_mps_falls_back_to_cpu(self):
        """MPS 利用可能時でも Seed-VC 非対応のため CPU にフォールバック"""
        from voice_convert import VoiceConverter

        with patch("voice_convert.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            converter = VoiceConverter(seed_vc_path=Path("/tmp"), device="auto")
            assert converter.device == "cpu"

    def test_device_auto_cpu(self):
        """GPU なしの場合は cpu を選択"""
        from voice_convert import VoiceConverter

        with patch("voice_convert.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            converter = VoiceConverter(seed_vc_path=Path("/tmp"), device="auto")
            assert converter.device == "cpu"

    def test_device_explicit(self):
        """明示指定時はそのまま使用"""
        from voice_convert import VoiceConverter

        converter = VoiceConverter(seed_vc_path=Path("/tmp"), device="cpu")
        assert converter.device == "cpu"


class TestConvertVoice:
    """コア変換ロジックのテスト"""

    def test_convert_voice_subprocess_called(self, tmp_path):
        """subprocess.run が正しい引数で呼ばれる"""
        from voice_convert import VoiceConverter

        seed_vc_path = tmp_path / "seed-vc"
        seed_vc_path.mkdir()
        (seed_vc_path / "inference.py").write_text("# dummy")
        # 専用 venv の Python をシミュレート
        venv_bin = seed_vc_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("# dummy")
        (venv_bin / "python").chmod(0o755)

        converter = VoiceConverter(seed_vc_path=seed_vc_path, device="cpu")

        source = tmp_path / "source.wav"
        reference = tmp_path / "reference.wav"
        output = tmp_path / "output.wav"

        # ダミーの音声ファイル作成
        from pydub.generators import Sine

        audio = Sine(440).to_audio_segment(duration=500)
        audio.export(str(source), format="wav")
        audio.export(str(reference), format="wav")

        with patch("subprocess.run") as mock_run:
            # Seed-VC が出力するファイルをシミュレート
            def side_effect(*args, **kwargs):
                # 出力ディレクトリにファイルを作成
                import re

                cmd = args[0]
                for i, arg in enumerate(cmd):
                    if arg == "--output":
                        out_dir = Path(cmd[i + 1])
                        out_dir.mkdir(parents=True, exist_ok=True)
                        dummy_out = out_dir / "source.wav"
                        audio.export(str(dummy_out), format="wav")
                        break
                return MagicMock(returncode=0)

            mock_run.side_effect = side_effect

            result = converter.convert_voice(source, reference, output)

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            # CPU モードではラッパースクリプト経由で実行されるため、
            # inference.py は直接引数に含まれない場合がある
            cmd_str = " ".join(call_args)
            assert str(source) in cmd_str
            assert str(reference) in cmd_str

    def test_convert_voice_subprocess_error(self, tmp_path):
        """subprocess エラー時に RuntimeError"""
        import subprocess

        from voice_convert import VoiceConverter

        seed_vc_path = tmp_path / "seed-vc"
        seed_vc_path.mkdir()
        (seed_vc_path / "inference.py").write_text("# dummy")
        venv_bin = seed_vc_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("# dummy")
        (venv_bin / "python").chmod(0o755)

        converter = VoiceConverter(seed_vc_path=seed_vc_path, device="cpu")

        source = tmp_path / "source.wav"
        reference = tmp_path / "reference.wav"
        output = tmp_path / "output.wav"

        from pydub.generators import Sine

        audio = Sine(440).to_audio_segment(duration=500)
        audio.export(str(source), format="wav")
        audio.export(str(reference), format="wav")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "inference.py", stderr="error message"
            )
            with pytest.raises(RuntimeError, match="Seed-VC"):
                converter.convert_voice(source, reference, output)


class TestReferenceDuration:
    """参照音声の長さチェックのテスト"""

    def test_check_reference_duration_warning_long(self, tmp_path, capsys):
        """30秒超の参照音声で警告"""
        from pydub.generators import Sine

        from voice_convert import VoiceConverter

        # 35秒の音声
        audio = Sine(440).to_audio_segment(duration=35000)
        ref_path = tmp_path / "long_ref.wav"
        audio.export(str(ref_path), format="wav")

        converter = VoiceConverter(seed_vc_path=tmp_path)
        converter.check_reference_duration(ref_path)

        captured = capsys.readouterr()
        assert "30秒" in captured.out or "警告" in captured.out

    def test_check_reference_duration_warning_short(self, tmp_path, capsys):
        """1秒未満の参照音声で警告"""
        from pydub.generators import Sine

        from voice_convert import VoiceConverter

        # 0.5秒の音声
        audio = Sine(440).to_audio_segment(duration=500)
        ref_path = tmp_path / "short_ref.wav"
        audio.export(str(ref_path), format="wav")

        converter = VoiceConverter(seed_vc_path=tmp_path)
        converter.check_reference_duration(ref_path)

        captured = capsys.readouterr()
        assert "1秒" in captured.out or "警告" in captured.out

    def test_check_reference_duration_ok(self, tmp_path, capsys):
        """適切な長さの参照音声では警告なし"""
        from pydub.generators import Sine

        from voice_convert import VoiceConverter

        # 10秒の音声
        audio = Sine(440).to_audio_segment(duration=10000)
        ref_path = tmp_path / "good_ref.wav"
        audio.export(str(ref_path), format="wav")

        converter = VoiceConverter(seed_vc_path=tmp_path)
        converter.check_reference_duration(ref_path)

        captured = capsys.readouterr()
        assert "警告" not in captured.out


class TestBatchProcessing:
    """バッチ処理のテスト"""

    def test_find_audio_files_in_directory(self, tmp_path):
        """ディレクトリ内の音声ファイルを検出"""
        from voice_convert import VoiceConverter

        # ダミーファイル作成
        (tmp_path / "audio1.wav").write_bytes(b"RIFF" + b"\x00" * 100)
        (tmp_path / "audio2.mp3").write_bytes(b"\x00" * 100)
        (tmp_path / "readme.txt").write_text("not audio")
        (tmp_path / "image.png").write_bytes(b"\x00" * 100)

        converter = VoiceConverter(seed_vc_path=tmp_path)
        files = converter.find_audio_files(tmp_path)

        assert len(files) == 2
        extensions = {f.suffix.lower() for f in files}
        assert extensions == {".wav", ".mp3"}

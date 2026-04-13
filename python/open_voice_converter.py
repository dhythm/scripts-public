#!/usr/bin/env python3
"""
OpenVoice V2 を使った音声 to 音声変換ツール。

既存の読み上げ音声の抑揚や間をできるだけ保ったまま、
参照音声の声質へ変換する用途を想定しています。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import typer
from pydub import AudioSegment

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

app = typer.Typer(
    add_completion=False,
    help="OpenVoice V2 を使った音声 to 音声変換 CLI",
)

DEFAULT_OPENVOICE_PATH = Path.home() / ".cache" / "openvoice"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OPENVOICE_PATH / "checkpoints_v2" / "converter"
OPENVOICE_REPO_URL = "https://github.com/myshell-ai/OpenVoice.git"
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}
OPENVOICE_MINIMAL_DEPENDENCIES = [
    "numpy==1.22.0",
    "librosa==0.9.1",
    "soundfile>=0.12.1",
    "pydub==0.25.1",
    "wavmark==0.0.3",
    "eng_to_ipa==0.0.2",
    "inflect==7.0.0",
    "unidecode==1.3.7",
    "pypinyin==0.50.0",
    "cn2an==0.5.22",
    "jieba==0.42.1",
    "langid==1.1.6",
]


def _get_openvoice_python(openvoice_path: Path) -> Path:
    venv_python = openvoice_path / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _detect_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class OpenVoiceConverter:
    def __init__(
        self,
        openvoice_path: Path = DEFAULT_OPENVOICE_PATH,
        checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
        device: str = "auto",
        tau: float = 0.3,
        enable_watermark: bool = False,
        use_vad: bool = True,
    ) -> None:
        self.openvoice_path = Path(openvoice_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = _detect_device(device)
        self.tau = tau
        self.enable_watermark = enable_watermark
        self.use_vad = use_vad

    @property
    def config_path(self) -> Path:
        return self.checkpoint_dir / "config.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "checkpoint.pth"

    def validate_installation(self) -> None:
        if not self.openvoice_path.exists():
            raise RuntimeError(
                f"OpenVoice が見つかりません: {self.openvoice_path}\n"
                "先に setup を実行してください。"
            )
        if not (self.openvoice_path / "openvoice").exists():
            raise RuntimeError(
                f"OpenVoice のコードが見つかりません: {self.openvoice_path / 'openvoice'}\n"
                "setup を再実行してください。"
            )
        if not self.config_path.exists():
            raise RuntimeError(
                f"OpenVoice の config.json が見つかりません: {self.config_path}\n"
                "checkpoints_v2 を配置してください。"
            )
        if not self.checkpoint_path.exists():
            raise RuntimeError(
                f"OpenVoice の checkpoint.pth が見つかりません: {self.checkpoint_path}\n"
                "checkpoints_v2 を配置してください。"
            )

    def validate_audio_file(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"パスがファイルではありません: {file_path}")
        if file_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"サポートされていないファイル形式: {file_path.suffix}\n"
                f"サポート形式: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
            )

    def prepare_audio(
        self,
        file_path: Path,
        sample_rate: int = 16000,
        normalize: bool = False,
        gain_db: float = 0.0,
    ) -> Path:
        self.validate_audio_file(file_path)

        audio = AudioSegment.from_file(str(file_path))
        audio = audio.set_channels(1).set_frame_rate(sample_rate)

        if gain_db != 0:
            audio = audio.apply_gain(gain_db)

        if normalize and audio.max_dBFS != float("-inf"):
            target_dbfs = -3.0
            audio = audio.apply_gain(target_dbfs - audio.max_dBFS)

        output_path = Path(tempfile.mktemp(prefix="openvoice_", suffix=".wav"))
        audio.export(str(output_path), format="wav")
        return output_path

    def build_conversion_payload(
        self,
        source: Path,
        reference: Path,
        output: Path,
    ) -> dict[str, Any]:
        output = output.resolve()
        return {
            "source": str(source.resolve()),
            "reference": str(reference.resolve()),
            "output": str(output),
            "openvoice_path": str(self.openvoice_path),
            "config_path": str(self.config_path),
            "checkpoint_path": str(self.checkpoint_path),
            "device": self.device,
            "tau": self.tau,
            "enable_watermark": self.enable_watermark,
            "use_vad": self.use_vad,
        }

    def convert_voice(
        self,
        source: Path,
        reference: Path,
        output: Path,
        normalize: bool = False,
        gain_db: float = 0.0,
    ) -> Path:
        self.validate_installation()

        temp_paths: list[Path] = []
        prepared_source = self.prepare_audio(source, normalize=normalize, gain_db=gain_db)
        prepared_reference = self.prepare_audio(reference, normalize=normalize)
        temp_paths.extend([prepared_source, prepared_reference])

        payload = self.build_conversion_payload(prepared_source, prepared_reference, output)
        wrapper_path = Path(tempfile.mktemp(prefix="openvoice_driver_", suffix=".py"))
        payload_path = Path(tempfile.mktemp(prefix="openvoice_payload_", suffix=".json"))
        temp_paths.extend([wrapper_path, payload_path])

        wrapper_path.write_text(_conversion_driver_script(), encoding="utf-8")
        payload_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

        openvoice_python = _get_openvoice_python(self.openvoice_path)
        env = dict(os.environ)
        if self.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

        try:
            subprocess.run(
                [str(openvoice_python), str(wrapper_path), str(payload_path)],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(self.openvoice_path),
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else str(exc)
            raise RuntimeError(f"OpenVoice の変換に失敗しました:\n{stderr}") from exc
        finally:
            for path in temp_paths:
                if path.exists():
                    path.unlink()

        if not output.exists():
            raise RuntimeError(f"OpenVoice の出力ファイルが生成されませんでした: {output}")

        return output


def _conversion_driver_script() -> str:
    return """\
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    repo_root = Path(payload["openvoice_path"])
    sys.path.insert(0, str(repo_root))

    from openvoice.api import ToneColorConverter

    converter = ToneColorConverter(
        payload["config_path"],
        device=payload["device"],
    )
    converter.load_ckpt(payload["checkpoint_path"])

    source_se = converter.extract_se(payload["source"])
    target_se = converter.extract_se(payload["reference"])

    output_path = Path(payload["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "audio_src_path": payload["source"],
        "src_se": source_se,
        "tgt_se": target_se,
        "output_path": str(output_path),
        "tau": payload["tau"],
    }
    if payload["enable_watermark"]:
        kwargs["message"] = "@MyShell"

    converter.convert(**kwargs)


if __name__ == "__main__":
    main()
"""


@app.command()
def setup(
    path: Path = typer.Option(
        DEFAULT_OPENVOICE_PATH,
        "--path",
        help="OpenVoice の配置先",
    ),
    update: bool = typer.Option(
        False,
        "--update",
        help="既存 clone を更新",
    ),
) -> None:
    """OpenVoice 本体をセットアップします。"""
    if path.exists() and not update:
        print(f"OpenVoice は既に存在します: {path}")
    elif path.exists() and update:
        print(f"OpenVoice を更新中: {path}")
        subprocess.run(["git", "pull"], cwd=str(path), check=True)
    else:
        print(f"OpenVoice を clone 中: {OPENVOICE_REPO_URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", OPENVOICE_REPO_URL, str(path)], check=True)

    uv_path = shutil.which("uv")
    venv_path = path / ".venv"
    if not venv_path.exists() or update:
        print(f"仮想環境を作成中: {venv_path}")
        if uv_path:
            subprocess.run([uv_path, "venv", str(venv_path), "--python", "3.10"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

    openvoice_python = _get_openvoice_python(path)
    print("OpenVoice の依存関係をインストール中...")
    if uv_path:
        editable_install = [
            uv_path,
            "pip",
            "install",
            "--python",
            str(openvoice_python),
            "--no-deps",
            "-e",
            str(path),
        ]
        subprocess.run(editable_install, check=True)
        minimal_install = [
            uv_path,
            "pip",
            "install",
            "--python",
            str(openvoice_python),
            *OPENVOICE_MINIMAL_DEPENDENCIES,
        ]
        subprocess.run(minimal_install, check=True)
    else:
        subprocess.run([str(openvoice_python), "-m", "ensurepip", "--upgrade"], check=True)
        editable_install = [str(openvoice_python), "-m", "pip", "install", "--no-deps", "-e", str(path)]
        subprocess.run(editable_install, check=True)
        minimal_install = [str(openvoice_python), "-m", "pip", "install", *OPENVOICE_MINIMAL_DEPENDENCIES]
        subprocess.run(minimal_install, check=True)

    print()
    print("セットアップは完了しました。")
    print("次に OpenVoice V2 の checkpoints を配置してください。")
    print(f"推奨配置先: {path / 'checkpoints_v2' / 'converter'}")
    print("必要ファイル: config.json, checkpoint.pth")


@app.command()
def convert(
    source: Path = typer.Argument(..., help="変換元の読み上げ音声"),
    reference: Path = typer.Argument(..., help="目標声質の参照音声"),
    output: Path = typer.Option(..., "--output", "-o", help="出力ファイル"),
    openvoice_path: Path = typer.Option(
        DEFAULT_OPENVOICE_PATH,
        "--openvoice-path",
        envvar="OPENVOICE_PATH",
        help="OpenVoice リポジトリのパス",
    ),
    checkpoint_dir: Path = typer.Option(
        DEFAULT_CHECKPOINT_DIR,
        "--checkpoint-dir",
        envvar="OPENVOICE_CHECKPOINT_DIR",
        help="config.json / checkpoint.pth を含むディレクトリ",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="使用デバイス (auto/cuda/cpu)",
    ),
    tau: float = typer.Option(
        0.3,
        "--tau",
        min=0.0,
        max=1.0,
        help="声質変換の強さ。低いほど原音寄り",
    ),
    normalize: bool = typer.Option(
        True,
        "--normalize/--no-normalize",
        help="入出力前に音量を軽く整える",
    ),
    gain: float = typer.Option(
        0.0,
        "--gain",
        help="変換元音声へのゲイン調整(dB)",
    ),
    use_vad: bool = typer.Option(
        True,
        "--use-vad/--no-vad",
        help="話者埋め込み抽出時に VAD 分割を使う",
    ),
    enable_watermark: bool = typer.Option(
        False,
        "--enable-watermark",
        help="OpenVoice の watermark を有効化",
    ),
) -> None:
    """読み上げ済み音声の抑揚を保ちながら、声質だけを変換します。"""
    converter = OpenVoiceConverter(
        openvoice_path=openvoice_path,
        checkpoint_dir=checkpoint_dir,
        device=device,
        tau=tau,
        enable_watermark=enable_watermark,
        use_vad=use_vad,
    )

    result = converter.convert_voice(
        source=source,
        reference=reference,
        output=output,
        normalize=normalize,
        gain_db=gain,
    )

    print(f"変換完了: {result}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

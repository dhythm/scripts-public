#!/usr/bin/env python3
"""
Seed-VCを使用した音声変換（Voice Conversion）ツール

短い参照音声（1〜30秒）から声質を借りて、入力音声を別の声に変換します。
学習不要の zero-shot voice conversion に対応しています。

使用方法:
    # Seed-VC のセットアップ
    python voice_convert.py setup

    # 音声変換（単一ファイル）
    python voice_convert.py convert input.wav reference.wav -o output.wav

    # 音声変換（バッチ処理）
    python voice_convert.py convert ./inputs/ reference.wav -o ./outputs/

    # 前処理オプション付き
    python voice_convert.py convert input.wav reference.wav --denoise --normalize
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import typer
from pydub import AudioSegment

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import noisereduce as nr

    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

app = typer.Typer(
    add_completion=False,
    help="Seed-VCを使用した音声変換ツール。短い参照音声から声質を変換します。",
)

DEFAULT_SEED_VC_PATH = Path.home() / ".cache" / "seed-vc"
SEED_VC_REPO_URL = "https://github.com/Plachtaa/seed-vc.git"

SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}


def _filter_requirements(req_file: Path) -> Path:
    """requirements.txt から pip 専用フラグを含む行を除外した一時ファイルを作成

    Seed-VC の requirements.txt には ``--pre --index-url ...`` のような
    uv pip が解釈できないフラグ付き行が含まれるため、それらを除外する。

    Returns:
        フィルタリング済みの requirements ファイルのパス。
        変更がなければ元のパスをそのまま返す。
    """
    lines = req_file.read_text().splitlines()
    filtered = [
        line for line in lines
        if not any(flag in line for flag in ("--pre", "--index-url", "--extra-index-url", "--find-links"))
    ]
    if len(filtered) == len(lines):
        return req_file
    filtered_path = Path(tempfile.mktemp(suffix=".txt", prefix="req_filtered_"))
    filtered_path.write_text("\n".join(filtered) + "\n")
    return filtered_path


def _get_seed_vc_python(seed_vc_path: Path) -> Path:
    """Seed-VC 専用 venv の Python パスを返す"""
    venv_python = seed_vc_path / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    # venv がない場合はシステムの Python にフォールバック
    return Path(sys.executable)


class VoiceConverter:
    """Seed-VCを使用した音声変換クラス"""

    def __init__(
        self,
        seed_vc_path: Path = DEFAULT_SEED_VC_PATH,
        device: str = "auto",
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
        fp16: bool = False,
    ):
        self.seed_vc_path = Path(seed_vc_path)
        self.diffusion_steps = diffusion_steps
        self.length_adjust = length_adjust
        self.inference_cfg_rate = inference_cfg_rate
        self.fp16 = fp16

        # デバイスの決定
        if device == "auto":
            if torch is not None:
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    # MPS は Seed-VC (PyTorch 2.4) の torch.autocast で未サポートのため CPU を使用
                    device = "cpu"
            else:
                device = "cpu"
        self.device = device

        # MPS 指定時の警告とフォールバック
        if self.device == "mps":
            print(
                "警告: MPS (Apple Silicon) は Seed-VC の torch.autocast で未サポートです。"
                " CPU にフォールバックします。"
            )
            self.device = "cpu"

        # CPU では FP16 が非効率
        if self.device == "cpu" and self.fp16:
            print("警告: CPU では FP16 が非効率なため無効化します。")
            self.fp16 = False

    def validate_installation(self) -> None:
        """Seed-VC のインストール状態を検証"""
        if not self.seed_vc_path.exists():
            raise RuntimeError(
                f"Seed-VC が見つかりません: {self.seed_vc_path}\n"
                "セットアップを実行してください: python voice_convert.py setup"
            )
        inference_py = self.seed_vc_path / "inference.py"
        if not inference_py.exists():
            raise RuntimeError(
                f"Seed-VC の inference.py が見つかりません: {inference_py}\n"
                "セットアップを再実行してください: python voice_convert.py setup --update"
            )

    def validate_audio_file(self, file_path: Path) -> None:
        """音声ファイルの検証"""
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"パスがファイルではありません: {file_path}")
        if file_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"サポートされていないファイル形式: {file_path.suffix}\n"
                f"サポートされている形式: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
            )

    def convert_to_wav(
        self,
        file_path: Path,
        normalize: bool = False,
        gain_db: float = 0,
        denoise: bool = False,
    ) -> Path:
        """音声ファイルを WAV 形式に変換し、オプションで前処理を適用

        Args:
            file_path: 音声ファイルのパス
            normalize: ピークノーマライズを適用するか
            gain_db: ゲイン調整(dB)
            denoise: ノイズ除去を適用するか

        Returns:
            変換後の WAV ファイルのパス
        """
        needs_processing = normalize or gain_db != 0 or denoise
        if file_path.suffix.lower() == ".wav" and not needs_processing:
            return file_path

        print(f"音声ファイルを WAV 形式に変換中: {file_path.name}")
        audio = AudioSegment.from_file(str(file_path))

        # ノイズ除去
        if denoise:
            audio = self._apply_noise_reduction(audio)

        # ゲイン調整
        if gain_db != 0:
            print(f"ゲイン調整: {gain_db:+.1f} dB")
            audio = audio + gain_db

        # ノーマライズ（ピークを -3.0 dBFS に調整）
        if normalize:
            target_dbfs = -3.0
            change = target_dbfs - audio.dBFS
            print(f"ノーマライズ: {change:+.1f} dB (ピーク: {audio.dBFS:.1f} → {target_dbfs:.1f} dBFS)")
            audio = audio.apply_gain(change)

        # 一時 WAV ファイルに出力
        temp_wav = Path(tempfile.mktemp(suffix=".wav", prefix="vc_"))
        audio.export(str(temp_wav), format="wav")
        print("変換が完了しました。")
        return temp_wav

    def _apply_noise_reduction(self, audio: AudioSegment) -> AudioSegment:
        """ノイズ除去を適用"""
        if not NOISEREDUCE_AVAILABLE:
            print("警告: noisereduce がインストールされていません。ノイズ除去をスキップします。")
            return audio

        import numpy as np

        print("ノイズ除去を適用中...")
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        sample_rate = audio.frame_rate

        reduced = nr.reduce_noise(
            y=samples,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.8,
        )

        denoised = AudioSegment(
            reduced.astype(np.int16).tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=audio.channels,
        )
        print("ノイズ除去が完了しました。")
        return denoised

    def check_reference_duration(self, reference_path: Path) -> None:
        """参照音声の長さをチェックし、必要に応じて警告"""
        audio = AudioSegment.from_file(str(reference_path))
        duration_sec = len(audio) / 1000.0

        if duration_sec < 1.0:
            print(f"警告: 参照音声が非常に短いです（{duration_sec:.1f}秒）。1秒以上を推奨します。")
        elif duration_sec > 30.0:
            print(
                f"警告: 参照音声が長いです（{duration_sec:.1f}秒）。"
                "Seed-VC は 1〜30秒の参照音声で最適に動作します。"
            )
        else:
            print(f"参照音声の長さ: {duration_sec:.1f}秒")

    def find_audio_files(self, directory: Path) -> list[Path]:
        """ディレクトリ内の音声ファイルを検出"""
        files = []
        for f in sorted(directory.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                files.append(f)
        return files

    def convert_voice(
        self,
        source: Path,
        reference: Path,
        output: Path,
    ) -> Path:
        """音声変換を実行

        Args:
            source: 変換元音声ファイル（WAV）
            reference: 参照音声ファイル（WAV）
            output: 出力ファイルパス

        Returns:
            出力ファイルのパス
        """
        self.validate_installation()

        # 一時出力ディレクトリ
        with tempfile.TemporaryDirectory(prefix="vc_out_") as temp_dir:
            temp_out = Path(temp_dir)

            seed_python = _get_seed_vc_python(self.seed_vc_path)
            cmd = [
                str(seed_python),
                str(self.seed_vc_path / "inference.py"),
                "--source",
                str(source),
                "--target",
                str(reference),
                "--output",
                str(temp_out),
                "--diffusion-steps",
                str(self.diffusion_steps),
                "--length-adjust",
                str(self.length_adjust),
                "--inference-cfg-rate",
                str(self.inference_cfg_rate),
            ]

            if self.fp16 and self.device == "cuda":
                cmd.extend(["--fp16", "True"])
            else:
                cmd.extend(["--fp16", "False"])

            print("Seed-VC で音声変換を実行中...")
            print(f"  デバイス: {self.device}")
            print(f"  拡散ステップ: {self.diffusion_steps}")

            # Seed-VC の inference.py はグローバルでデバイスを自動検出するため、
            # CPU を強制する場合は MPS を無効化するラッパースクリプト経由で実行する
            env = dict(os.environ)
            wrapper_script: Optional[Path] = None
            if self.device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
                # MPS を無効化するラッパースクリプトを一時ファイルに書き出す
                wrapper_script = Path(tempfile.mktemp(suffix=".py", prefix="vc_wrapper_"))
                inference_path = str(self.seed_vc_path / "inference.py").replace("\\", "\\\\")
                seed_vc_dir = str(self.seed_vc_path).replace("\\", "\\\\")
                wrapper_script.write_text(
                    "import sys\n"
                    f'sys.path.insert(0, "{seed_vc_dir}")\n'
                    "import torch\n"
                    "torch.backends.mps.is_available = lambda: False\n"
                    "import runpy\n"
                    f'sys.argv[0] = "{inference_path}"\n'
                    f'runpy.run_path("{inference_path}", run_name="__main__")\n'
                )
                cmd[1] = str(wrapper_script)  # inference.py をラッパーに置換

            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=str(self.seed_vc_path),
                    env=env,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Seed-VC の実行に失敗しました:\n{e.stderr}"
                ) from e
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Seed-VC の inference.py を実行できません: {e}"
                ) from e
            finally:
                if wrapper_script and wrapper_script.exists():
                    wrapper_script.unlink()

            # 出力ファイルを検出してコピー
            output_files = list(temp_out.glob("*.wav"))
            if not output_files:
                output_files = list(temp_out.glob("*.*"))

            if not output_files:
                raise RuntimeError("Seed-VC の出力ファイルが見つかりません。")

            # 最新のファイルを使用
            result_file = max(output_files, key=lambda f: f.stat().st_mtime)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(result_file), str(output))

        print(f"変換完了: {output}")
        return output


@app.command()
def setup(
    path: Path = typer.Option(
        DEFAULT_SEED_VC_PATH,
        "--path",
        help="Seed-VC のインストール先パス",
    ),
    update: bool = typer.Option(
        False,
        "--update",
        help="既存のインストールを更新",
    ),
) -> None:
    """Seed-VC をセットアップ（git clone + 依存関係インストール）"""
    if path.exists() and not update:
        print(f"Seed-VC は既にインストールされています: {path}")
        print("更新するには --update オプションを使用してください。")
        return

    if path.exists() and update:
        print(f"Seed-VC を更新中: {path}")
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=str(path),
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"git pull に失敗しました: {e}") from e
    else:
        print(f"Seed-VC をクローン中: {SEED_VC_REPO_URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", SEED_VC_REPO_URL, str(path)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"git clone に失敗しました: {e}") from e

    # Seed-VC 専用の venv を作成（プロジェクトの venv と依存関係が競合するため分離）
    venv_path = path / ".venv"
    uv_path = shutil.which("uv")

    if not venv_path.exists() or update:
        print(f"Seed-VC 専用の仮想環境を作成中: {venv_path}")
        if uv_path:
            subprocess.run([uv_path, "venv", str(venv_path), "--python", "3.10"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

    # 依存関係のインストール
    req_file = path / "requirements.txt"
    if not req_file.exists():
        req_file = path / "requirements-mac.txt"

    if req_file.exists():
        print(f"依存関係をインストール中: {req_file.name}")
        # requirements.txt から pip 専用フラグ（--pre, --index-url 等）を含む行を除外
        # Seed-VC の requirements.txt には CUDA nightly ビルド用の行が含まれるため
        filtered_req = _filter_requirements(req_file)
        seed_python = _get_seed_vc_python(path)
        if uv_path:
            install_cmd = [uv_path, "pip", "install", "--python", str(seed_python), "-r", str(filtered_req)]
        else:
            install_cmd = [str(seed_python), "-m", "pip", "install", "-r", str(filtered_req)]
        try:
            subprocess.run(install_cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"依存関係のインストールに失敗しました: {e}") from e
        finally:
            if filtered_req != req_file and filtered_req.exists():
                filtered_req.unlink()

    print(f"Seed-VC のセットアップが完了しました: {path}")


@app.command()
def convert(
    source: Path = typer.Argument(
        ...,
        help="変換元の音声ファイル（またはディレクトリ）",
    ),
    reference: Path = typer.Argument(
        ...,
        help="参照音声ファイル（1〜30秒推奨）",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="出力ファイルパス（またはディレクトリ）",
    ),
    seed_vc_path: Path = typer.Option(
        DEFAULT_SEED_VC_PATH,
        "--seed-vc-path",
        envvar="SEED_VC_PATH",
        help="Seed-VC のインストールパス",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="使用デバイス (auto/cuda/mps/cpu)",
    ),
    diffusion_steps: int = typer.Option(
        25,
        "--diffusion-steps",
        help="拡散ステップ数（大きいほど高品質、遅い）",
    ),
    length_adjust: float = typer.Option(
        1.0,
        "--length-adjust",
        help="長さ調整係数（<1.0: 早口、>1.0: ゆっくり）",
    ),
    inference_cfg_rate: float = typer.Option(
        0.7,
        "--inference-cfg-rate",
        help="推論 CFG レート",
    ),
    fp16: bool = typer.Option(
        False,
        "--fp16",
        help="FP16 精度を使用（CUDA 推奨）",
    ),
    normalize: bool = typer.Option(
        False,
        "--normalize",
        help="音声をノーマライズ",
    ),
    denoise: bool = typer.Option(
        False,
        "--denoise",
        help="ノイズ除去を適用",
    ),
    gain: float = typer.Option(
        0.0,
        "--gain",
        help="ゲイン調整 (dB)",
    ),
) -> None:
    """音声変換を実行（入力音声の声質を参照音声に変換）"""
    print("=" * 60)
    print("音声変換 (Voice Conversion) - Seed-VC")
    print("=" * 60)
    print()
    print("注意: 音声の権利・利用規約にご注意ください。")
    print("      他者の声を無断で使用・公開することは法的リスクがあります。")
    print()

    start_time = time.time()

    converter = VoiceConverter(
        seed_vc_path=seed_vc_path,
        device=device,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        fp16=fp16,
    )

    converter.validate_installation()

    # 参照音声のバリデーションと長さチェック
    converter.validate_audio_file(reference)
    converter.check_reference_duration(reference)

    # 参照音声を WAV に変換
    reference_wav = converter.convert_to_wav(reference, normalize=normalize, denoise=denoise, gain_db=gain)

    temp_files: list[Path] = []
    if reference_wav != reference:
        temp_files.append(reference_wav)

    try:
        if source.is_dir():
            # バッチ処理
            audio_files = converter.find_audio_files(source)
            if not audio_files:
                print(f"エラー: ディレクトリ内に音声ファイルが見つかりません: {source}")
                raise typer.Exit(1)

            output_dir = output or source / "converted"
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nバッチ処理: {len(audio_files)} ファイル")
            from tqdm import tqdm

            for audio_file in tqdm(audio_files, desc="変換中"):
                source_wav = converter.convert_to_wav(audio_file, normalize=normalize, denoise=denoise, gain_db=gain)
                if source_wav != audio_file:
                    temp_files.append(source_wav)

                out_path = output_dir / f"{audio_file.stem}_converted.wav"
                converter.convert_voice(source_wav, reference_wav, out_path)

            print(f"\n全ファイルの変換が完了しました: {output_dir}")
        else:
            # 単一ファイル処理
            converter.validate_audio_file(source)

            source_wav = converter.convert_to_wav(source, normalize=normalize, denoise=denoise, gain_db=gain)
            if source_wav != source:
                temp_files.append(source_wav)

            output_path = output or source.parent / f"{source.stem}_converted.wav"
            converter.convert_voice(source_wav, reference_wav, output_path)
    finally:
        # 一時ファイルの削除
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()

    elapsed = time.time() - start_time
    print(f"\n処理時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    app()

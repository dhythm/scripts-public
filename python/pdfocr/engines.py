"""ローカルOCRエンジン（NDLOCR-Lite / YomiToku / Indra OCR）の呼び出し

いずれも依存パッケージのバージョンがこのプロジェクトと衝突するため、
~/.cache 配下の専用 venv にインストールしてサブプロセスとして実行する。
"""

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Union

NDL_REPO_URL = "https://github.com/ndl-lab/ndlocr-lite.git"
# 動作確認したコミット（v1.3.0）。インストール済みCLIは上流のパッケージ定義の不備で
# 起動しないため、src/ocr.py を直接実行する
NDL_COMMIT = "d001dad2013ceabb5736ff8d637474e443eb73c2"
DEFAULT_NDL_PATH = Path.home() / ".cache" / "ndlocr-lite"

YOMITOKU_VERSION = "0.15.0"
DEFAULT_YOMITOKU_PATH = Path.home() / ".cache" / "yomitoku"

INDRA_VERSION = "1.1.2"
DEFAULT_INDRA_PATH = Path.home() / ".cache" / "indra-ocr"
# 柱とノンブル。本文には含めない
INDRA_FURNITURE_LABELS = ("Page-Header", "Page-Footer")

ENGINES = ["ndl", "yomitoku", "indra"]


def parse_ndl_json(path: Union[str, Path]) -> str:
    """NDLOCR-Lite のページJSONから、行を読み順に改行でつないだ本文を返す"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lines = []
    for block in data.get("contents", []):
        for line in block:
            text = line.get("text", "")
            if text:
                lines.append(text)
    return "\n".join(lines)


def parse_yomitoku_json(path: Union[str, Path]) -> str:
    """YomiToku のページJSONから、柱とノンブルを除いた段落を読み順に返す"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    paragraphs = sorted(data.get("paragraphs", []), key=lambda p: p.get("order", 0))
    return "\n\n".join(
        p["contents"] for p in paragraphs
        if p.get("contents") and p.get("role") not in ("page_header", "page_footer")
    )


def parse_indra_jsonl(path: Union[str, Path]) -> str:
    """Indra OCR のJSON Lines（画像1枚＝1行）から、柱とノンブルを除いたブロックを読み順に返す"""
    pages = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        if record.get("source") == "error":
            print(f"警告: Indra OCR がページを読めませんでした: {record.get('note', '')}", file=sys.stderr)
            continue
        blocks = record.get("blocks") or []
        if blocks:
            text = "\n\n".join(
                b["text"] for b in blocks
                if b.get("text") and b.get("label") not in INDRA_FURNITURE_LABELS
            )
        else:
            text = "\n".join(line["text"] for line in record.get("lines") or [] if line.get("text"))
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _require_uv() -> str:
    uv_path = shutil.which("uv")
    if not uv_path:
        raise RuntimeError("uv が見つかりません。https://docs.astral.sh/uv/ からインストールしてください。")
    return uv_path


def _venv_python(path: Path) -> Path:
    return path / ".venv" / "bin" / "python"


def setup_ndl(path: Path = DEFAULT_NDL_PATH, update: bool = False) -> None:
    """NDLOCR-Lite をクローンし、専用 venv に依存をインストールする"""
    uv_path = _require_uv()
    if not (path / ".git").exists():
        print(f"NDLOCR-Lite をクローン中: {NDL_REPO_URL}")
        subprocess.run(["git", "clone", NDL_REPO_URL, str(path)], check=True)
    elif update:
        subprocess.run(["git", "-C", str(path), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", NDL_COMMIT], check=True)

    if not _venv_python(path).exists() or update:
        print(f"NDLOCR-Lite 専用の仮想環境を作成中: {path / '.venv'}")
        subprocess.run([uv_path, "venv", str(path / ".venv"), "--python", "3.12"], check=True)
        subprocess.run(
            [uv_path, "pip", "install", "-p", str(_venv_python(path)),
             "-r", str(path / "requirements.txt")],
            check=True,
        )
    print("NDLOCR-Lite のセットアップが完了しました。")


def setup_yomitoku(path: Path = DEFAULT_YOMITOKU_PATH, update: bool = False) -> None:
    """YomiToku を専用 venv にインストールする（モデルは初回実行時にダウンロードされる）"""
    uv_path = _require_uv()
    if not _venv_python(path).exists() or update:
        print(f"YomiToku 専用の仮想環境を作成中: {path / '.venv'}")
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run([uv_path, "venv", str(path / ".venv"), "--python", "3.12"], check=True)
        subprocess.run(
            [uv_path, "pip", "install", "-p", str(_venv_python(path)), f"yomitoku=={YOMITOKU_VERSION}"],
            check=True,
        )
    print("YomiToku のセットアップが完了しました。")


def setup_indra(path: Path = DEFAULT_INDRA_PATH, update: bool = False) -> None:
    """Indra OCR を専用 venv にインストールする（モデル約76MBは初回実行時にダウンロードされる）"""
    uv_path = _require_uv()
    if not _venv_python(path).exists() or update:
        print(f"Indra OCR 専用の仮想環境を作成中: {path / '.venv'}")
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run([uv_path, "venv", str(path / ".venv"), "--python", "3.12"], check=True)
        subprocess.run(
            [uv_path, "pip", "install", "-p", str(_venv_python(path)), f"indra-ocr=={INDRA_VERSION}"],
            check=True,
        )
    print("Indra OCR のセットアップが完了しました。")


def _ensure_installed(engine: str, path: Path) -> Path:
    python = _venv_python(path)
    if not python.exists():
        raise RuntimeError(
            f"{engine} がセットアップされていません。次を実行してください:\n"
            f"  uv run python -m pdfocr setup {engine}"
        )
    return python


def run_ndl(image_paths: List[Path], work_dir: Path, path: Path = DEFAULT_NDL_PATH) -> Dict[Path, str]:
    """NDLOCR-Lite で画像群をOCRし、画像パスごとの本文を返す"""
    python = _ensure_installed("ndl", path)
    image_dir = image_paths[0].parent
    out_dir = work_dir / "ndl_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(python), str(path / "src" / "ocr.py"),
         "--sourcedir", str(image_dir), "--output", str(out_dir), "--json-only"],
        check=True,
        cwd=str(path / "src"),
    )
    return {image: parse_ndl_json(out_dir / f"{image.stem}.json") for image in image_paths}


def default_yomitoku_device() -> str:
    """Apple Silicon では GPU（mps）、それ以外は CPU を使う"""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def run_yomitoku(
    image_paths: List[Path],
    work_dir: Path,
    path: Path = DEFAULT_YOMITOKU_PATH,
    device: str = "",
) -> Dict[Path, str]:
    """YomiToku で画像群をOCR（ルビは除去）し、画像パスごとの本文を返す"""
    _ensure_installed("yomitoku", path)
    image_dir = image_paths[0].parent
    out_dir = work_dir / "yomitoku_out"
    subprocess.run(
        [str(path / ".venv" / "bin" / "yomitoku"), str(image_dir),
         "-f", "json", "-o", str(out_dir), "-d", device or default_yomitoku_device(),
         "--ignore_ruby"],
        check=True,
    )
    results = {}
    for image in image_paths:
        # 出力名は「<ディレクトリ名>_<画像名>_p1.json」
        matches = sorted(out_dir.glob(f"*_{image.stem}_p1.json"))
        if not matches:
            raise RuntimeError(f"YomiToku の出力が見つかりません: {image.name}")
        results[image] = parse_yomitoku_json(matches[0])
    return results


def run_indra(image_paths: List[Path], work_dir: Path, path: Path = DEFAULT_INDRA_PATH) -> Dict[Path, str]:
    """Indra OCR で画像群をOCR（全コアで並列）し、画像パスごとの本文を返す"""
    _ensure_installed("indra", path)
    out_dir = work_dir / "indra_out"
    subprocess.run(
        [str(path / ".venv" / "bin" / "indra-ocr"), *[str(p) for p in image_paths],
         "--mode", "ocr", "--format", "json", "--out", str(out_dir)],
        check=True,
    )
    results = {}
    for image in image_paths:
        # 出力名は「<画像名>.jsonl」
        output = out_dir / f"{image.stem}.jsonl"
        if not output.exists():
            raise RuntimeError(f"Indra OCR の出力が見つかりません: {image.name}")
        results[image] = parse_indra_jsonl(output)
    return results

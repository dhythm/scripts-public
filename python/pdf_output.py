"""Atomic text publication shared by the local PDF command-line tools."""
import os
import stat
import tempfile
from pathlib import Path


def validate_output(source, target, *, overwrite=False):
    source, target = Path(source), Path(target)
    if target.resolve() == source.resolve() or (target.exists() and source.samefile(target)):
        raise ValueError("入力PDFを出力先には指定できません")
    if target.exists() and not overwrite:
        raise FileExistsError(f"出力先が既に存在します: {target}（上書きは --force）")


def _output_mode(target):
    """上書き時は既存ファイルの権限を引き継ぎ、新規作成時は umask に従う"""
    if target.exists():
        return stat.S_IMODE(target.stat().st_mode)
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def save_text(source, target, text, *, overwrite=False):
    source, target = Path(source), Path(target)
    validate_output(source, target, overwrite=overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        # mkstemp は 0600 で作成するため、通常のファイル作成と同じ権限に揃える
        os.chmod(temporary, _output_mode(target))
        validate_output(source, target, overwrite=overwrite)
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)

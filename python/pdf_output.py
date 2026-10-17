"""Atomic text publication shared by the local PDF command-line tools."""
import os
import tempfile
from pathlib import Path


def validate_output(source, target, *, overwrite=False):
    source, target = Path(source), Path(target)
    if target.resolve() == source.resolve() or (target.exists() and source.samefile(target)):
        raise ValueError("入力PDFを出力先には指定できません")
    if target.exists() and not overwrite:
        raise FileExistsError(f"出力先が既に存在します: {target}（上書きは --force）")


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
        validate_output(source, target, overwrite=overwrite)
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)

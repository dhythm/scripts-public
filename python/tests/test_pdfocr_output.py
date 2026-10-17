"""pdfocr の出力の安全性のテスト（旧 test_pdf_output_safety.py から移植）"""

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdfocr import cli, output
from pdfocr.output import save_text


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "input.pdf"
    path.write_bytes(b"original")
    return path


def test_cannot_overwrite_input_even_with_force(source, monkeypatch):
    monkeypatch.setattr(cli, "convert", lambda *a, **k: ["本文です。"])
    assert cli.main([str(source), "-o", str(source), "--force"]) == 1
    assert source.read_bytes() == b"original"


def test_force_does_not_follow_hard_link_to_input(source, tmp_path, monkeypatch):
    target = tmp_path / "output.txt"
    target.hardlink_to(source)
    monkeypatch.setattr(cli, "convert", lambda *a, **k: ["本文です。"])
    assert cli.main([str(source), "-o", str(target), "--force"]) == 1
    assert source.read_bytes() == b"original"


def test_ocr_failure_does_not_publish_partial_result(source, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated OCR failure")
    monkeypatch.setattr(cli, "convert", fail)
    assert cli.main([str(source)]) == 1
    assert not source.with_suffix(".txt").exists()


def test_failed_atomic_replace_preserves_previous_output(source, tmp_path, monkeypatch):
    target = tmp_path / "output.txt"
    target.write_text("previous")

    def fail(*args):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(output.os, "replace", fail)
    with pytest.raises(OSError):
        save_text(source, target, "new", overwrite=True)
    assert target.read_text() == "previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.pdf", "output.txt"]


def test_new_output_follows_umask(source, tmp_path):
    target = tmp_path / "out.txt"
    previous = os.umask(0o022)
    try:
        save_text(source, target, "text")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_forced_output_keeps_existing_mode(source, tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    save_text(source, target, "new", overwrite=True)
    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640

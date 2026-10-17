from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import os
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pdf_to_text
import pdf_ocr
import pdf_to_markdown


@pytest.mark.parametrize("module, function, suffix", [
    (pdf_to_text, "extract_text_from_pdf", ".txt"),
    (pdf_ocr, "extract_text_with_ocr", ".ocr.txt"),
    (pdf_to_markdown, "convert_pdf_to_markdown", ".md"),
])
def test_existing_output_is_untouched_before_extraction(tmp_path, monkeypatch, module, function, suffix):
    source = tmp_path / "input.pdf"
    source.write_bytes(b"test fixture")
    target = source.with_suffix(suffix)
    target.write_text("existing result")
    def unexpected(*args, **kwargs):
        pytest.fail("extraction must not run for an existing output")
    monkeypatch.setattr(module, "extract_text_from_pdf" if module is pdf_to_markdown else
                        "pdf_to_images" if module is pdf_ocr else "open", unexpected,
                        raising=False)
    assert not getattr(module, function)(str(source))
    assert target.read_text() == "existing result"


def test_force_allows_text_replacement(tmp_path, monkeypatch):
    source = tmp_path / "input.pdf"
    source.write_bytes(b"test fixture")
    source.with_suffix(".txt").write_text("old")
    monkeypatch.setattr(pdf_to_text.pdfplumber, "open", lambda *_: nullcontext(
        SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "new text")])) )
    assert pdf_to_text.extract_text_from_pdf(source, overwrite=True)
    assert "new text" in source.with_suffix(".txt").read_text()


def test_ocr_page_failure_does_not_publish_partial_result(tmp_path, monkeypatch):
    source = tmp_path / "input.pdf"
    source.write_bytes(b"test fixture")
    monkeypatch.setattr(pdf_ocr, "pdf_to_images", lambda *_: [1, 2])
    def ocr(page):
        if page == 2:
            raise RuntimeError("simulated OCR failure")
        return "first page"
    monkeypatch.setattr(pdf_ocr, "perform_ocr", ocr)
    assert not pdf_ocr.extract_text_with_ocr(source)
    assert not source.with_suffix(".ocr.txt").exists()


def test_markdown_cannot_overwrite_input_even_with_force(tmp_path):
    source = tmp_path / "input.pdf"
    source.write_bytes(b"original")
    assert not pdf_to_markdown.convert_pdf_to_markdown(source, source, overwrite=True)
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize("module, function", [(pdf_to_text, "extract_text_from_pdf"), (pdf_ocr, "extract_text_with_ocr")])
def test_cli_returns_failure(monkeypatch, module, function):
    monkeypatch.setattr(sys, "argv", [module.__name__, "missing.pdf"])
    monkeypatch.setattr(module, function, lambda *a, **kw: False)
    if module is pdf_ocr:
        monkeypatch.setattr(module, "check_tesseract", lambda: True)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1


def test_force_does_not_follow_hard_link_to_input(tmp_path):
    source = tmp_path / "input.pdf"
    source.write_bytes(b"original")
    target = tmp_path / "output.md"
    target.hardlink_to(source)
    assert not pdf_to_markdown.convert_pdf_to_markdown(source, target, overwrite=True)
    assert source.read_bytes() == b"original"


def test_failed_atomic_replace_preserves_previous_output(tmp_path, monkeypatch):
    import pdf_output
    source, target = tmp_path / "input.pdf", tmp_path / "output.txt"
    source.write_bytes(b"source")
    target.write_text("previous")
    def fail(*args):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(pdf_output.os, "replace", fail)
    with pytest.raises(OSError):
        pdf_output.save_text(source, target, "new", overwrite=True)
    assert target.read_text() == "previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.pdf", "output.txt"]


def test_new_output_follows_umask(tmp_path):
    from pdf_output import save_text
    source = tmp_path / "in.pdf"
    source.write_bytes(b"%PDF")
    target = tmp_path / "out.txt"
    previous = os.umask(0o022)
    try:
        save_text(source, target, "text")
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_forced_output_keeps_existing_mode(tmp_path):
    from pdf_output import save_text
    source = tmp_path / "in.pdf"
    source.write_bytes(b"%PDF")
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    save_text(source, target, "new", overwrite=True)
    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640

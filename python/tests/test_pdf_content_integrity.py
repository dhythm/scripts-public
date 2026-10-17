import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from google_pdf_pipeline import PdfTextExtractor
from pdf_to_markdown import format_text_to_markdown


def test_trailing_copyright_is_retained_after_body():
    result = format_text_to_markdown("本文\nCopyright 2026 Example", "sample.pdf")
    assert "Copyright 2026 Example" in result
    assert result.index("本文") < result.index("Copyright")


def test_normal_press_and_agency_body_is_not_publication_metadata():
    result = format_text_to_markdown("Press release\nAgency research findings", "sample.pdf")
    assert "Press release" in result
    assert "Agency research findings" in result
    assert "原書情報" not in result


def test_publication_lines_are_separated_and_body_order_is_preserved():
    result = format_text_to_markdown("First\nCopyright A\nAll rights reserved\nLast", "sample.pdf")
    assert "> Copyright A\n> All rights reserved" in result
    assert result.index("First") < result.index("Copyright") < result.index("Last")


def test_mixed_pdf_ocr_only_missing_pages_in_order(monkeypatch):
    extractor = PdfTextExtractor()
    monkeypatch.setattr(extractor, "_extract_pages_with_pdfplumber", lambda _: ["first", "", "third"])
    ocr = Mock(return_value="second")
    monkeypatch.setattr(extractor, "_extract_text_with_ocr", ocr)
    assert asyncio.run(extractor.extract_text(b"fixture")) == "first\n\nsecond\n\nthird"
    ocr.assert_called_once_with(b"fixture", 2)


def test_disable_ocr_never_invokes_ocr(monkeypatch):
    extractor = PdfTextExtractor(use_ocr_fallback=False)
    monkeypatch.setattr(extractor, "_extract_pages_with_pdfplumber", lambda _: ["first", ""])
    ocr = Mock(side_effect=AssertionError("OCR disabled"))
    monkeypatch.setattr(extractor, "_extract_text_with_ocr", ocr)
    assert asyncio.run(extractor.extract_text(b"fixture")) == "first"
    ocr.assert_not_called()


def test_ocr_failure_is_not_silent_partial_success(monkeypatch):
    extractor = PdfTextExtractor()
    monkeypatch.setattr(extractor, "_extract_pages_with_pdfplumber", lambda _: ["first", ""])
    monkeypatch.setattr(extractor, "_extract_text_with_ocr", Mock(side_effect=RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="ページ 2"):
        asyncio.run(extractor.extract_text(b"fixture"))

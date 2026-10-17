"""ページごとにテキスト層とOCRを振り分けてPDFからテキストを取り出す"""

import tempfile
from pathlib import Path
from typing import Callable, Dict, List

from pdfocr.textlayer import extract_text_pages, needs_ocr

OCR_MODES = ["auto", "always", "never"]
DEFAULT_DPI = 300

OcrRunner = Callable[[List[Path], Path], Dict[Path, str]]


def render_pages(pdf_path: Path, page_indexes: List[int], dpi: int, out_dir: Path) -> List[Path]:
    """指定ページをPNG画像にする（pdfium を使うので Poppler は不要）"""
    import pypdfium2 as pdfium

    image_dir = out_dir / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        paths = []
        for index in page_indexes:
            image = pdf[index].render(scale=dpi / 72).to_pil()
            path = image_dir / f"page_{index + 1:04d}.png"
            image.save(path)
            paths.append(path)
        return paths
    finally:
        pdf.close()


def convert(
    pdf_path: Path,
    ocr_mode: str,
    run_ocr: OcrRunner,
    dpi: int = DEFAULT_DPI,
    extract: Callable[[Path], List[str]] = extract_text_pages,
    render: Callable[[Path, List[int], int, Path], List[Path]] = render_pages,
) -> List[str]:
    """
    PDFの各ページのテキストを返す

    Args:
        ocr_mode: auto（テキスト層が使えないページだけOCR）/ always（全ページOCR）/
            never（テキスト層のみ）
        run_ocr: 画像パスのリストと作業ディレクトリを受け取り、画像ごとの本文を返す関数
    """
    if ocr_mode not in OCR_MODES:
        raise ValueError(f"ocr_mode は {OCR_MODES} のいずれかです: {ocr_mode}")

    pages = list(extract(pdf_path))
    if ocr_mode == "never":
        return pages
    if ocr_mode == "always":
        targets = list(range(len(pages)))
    else:
        targets = [i for i, text in enumerate(pages) if needs_ocr(text)]
    if not targets:
        return pages

    print(f"{len(pages)} ページ中 {len(targets)} ページをOCRします...")
    with tempfile.TemporaryDirectory(prefix="pdfocr_") as tmp:
        work_dir = Path(tmp)
        images = render(pdf_path, targets, dpi, work_dir)
        results = run_ocr(images, work_dir)
        for index, image in zip(targets, images):
            pages[index] = results[image]
    return pages

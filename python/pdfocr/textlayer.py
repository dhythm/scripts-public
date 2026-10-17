"""PDFのテキスト層の抽出と、OCRが必要かどうかの判定"""

import re
from pathlib import Path
from typing import Iterable, List, Tuple, Union

# ページ上部・下部のこの比率の領域を、柱やページ番号とみなして除外する
DEFAULT_HEADER_RATIO = 0.12
DEFAULT_FOOTER_RATIO = 0.13

# OCRが必要と判定する基準
MIN_TEXT_CHARS = 20          # これより短いページはテキスト層がないとみなす
MAX_BROKEN_RATIO = 0.05      # (cid:NNN) や U+FFFD の比率の上限
MIN_EXPECTED_RATIO = 0.8     # 日本語・英数字・記号として想定される文字の比率の下限
# アクセント付きラテン文字の比率の上限。UTF-8 を Latin-1 として読んだ文字化け
# （Ã や Â の連続）を検出するため。通常の英文ではごく一部にしか現れない
MAX_ACCENTED_LATIN_RATIO = 0.3
ACCENTED_LATIN_RANGE = (0x00A0, 0x024F)

CID_PATTERN = re.compile(r'\(cid:\d+\)')
EXPECTED_CHAR_RANGES = [
    (0x0021, 0x007E),  # ASCII
    (0x00A0, 0x024F),  # ラテン文字（アクセント付き）
    (0x2000, 0x206F),  # 一般句読点
    (0x2100, 0x22FF),  # 記号・矢印・数学記号
    (0x2460, 0x24FF),  # 丸数字
    (0x2500, 0x27BF),  # 罫線・図形・記号
    (0x3000, 0x30FF),  # CJK記号・ひらがな・カタカナ
    (0x3190, 0x31FF),  # 漢文・カタカナ拡張
    (0x3200, 0x33FF),  # 囲み文字・単位
    (0x3400, 0x4DBF),  # CJK統合漢字拡張A
    (0x4E00, 0x9FFF),  # CJK統合漢字
    (0xF900, 0xFAFF),  # CJK互換漢字
    (0xFE10, 0xFE1F),  # 縦書き形
    (0xFE30, 0xFE4F),  # CJK互換形（縦書き約物）
    (0xFF00, 0xFFEF),  # 全角英数・半角カナ
    (0x20000, 0x3134F),  # CJK統合漢字拡張B以降
]

Box = Tuple[float, float, float, float, str]


def _is_expected_char(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in EXPECTED_CHAR_RANGES)


def needs_ocr(text: str) -> bool:
    """
    テキスト層がそのまま使えず、OCRが必要かどうかを判定する

    次のいずれかに当てはまればOCRが必要とみなす。
    - 文字がほとんどない（スキャン画像だけのページ）
    - フォントの対応表がなく (cid:NNN) や U+FFFD になっている
    - 想定外の文字（文字化け）が多い
    """
    compact = re.sub(r'\s+', '', text or '')
    without_cid = CID_PATTERN.sub('', compact)
    if len(without_cid) < MIN_TEXT_CHARS:
        return True

    broken = len(CID_PATTERN.findall(compact)) + compact.count('�')
    if broken / len(without_cid) > MAX_BROKEN_RATIO:
        return True

    start, end = ACCENTED_LATIN_RANGE
    accented = sum(1 for char in without_cid if start <= ord(char) <= end)
    if accented / len(without_cid) > MAX_ACCENTED_LATIN_RATIO:
        return True

    expected = sum(1 for char in without_cid if _is_expected_char(char))
    return expected / len(without_cid) < MIN_EXPECTED_RATIO


def select_body_boxes(
    boxes: Iterable[Box],
    page_height: float,
    header_ratio: float = DEFAULT_HEADER_RATIO,
    footer_ratio: float = DEFAULT_FOOTER_RATIO,
) -> List[str]:
    """
    ヘッダー/フッター領域にかからないテキストボックスの文字列を返す

    Args:
        boxes: (x0, y0, x1, y1, text)。PDF座標系なので y は下が0
        page_height: ページの高さ
    """
    header_boundary = page_height * (1 - header_ratio)
    footer_boundary = page_height * footer_ratio
    return [
        text for _x0, y0, _x1, y1, text in boxes
        if y1 <= header_boundary and y0 >= footer_boundary
    ]


def extract_text_pages(
    pdf_path: Union[str, Path],
    header_ratio: float = DEFAULT_HEADER_RATIO,
    footer_ratio: float = DEFAULT_FOOTER_RATIO,
) -> List[str]:
    """縦書きを考慮してPDFのテキスト層をページごとに抽出する"""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextBox

    laparams = LAParams(detect_vertical=True, all_texts=True)
    pages = []
    for page_layout in extract_pages(str(pdf_path), laparams=laparams):
        boxes = [
            (*element.bbox, element.get_text())
            for element in page_layout
            if isinstance(element, LTTextBox)
        ]
        body = select_body_boxes(boxes, page_layout.height, header_ratio, footer_ratio)
        pages.append('\n'.join(body))
    return pages

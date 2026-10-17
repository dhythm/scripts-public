"""抽出・OCR後のテキストに共通で適用する後処理"""

import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple

# 縦書き用の約物を横書きの文字に変換する
VERTICAL_FORMS = {
    '︑': '、',
    '︒': '。',
    '︵': '（',
    '︶': '）',
    '︽': '《',
    '︾': '》',
    '﹁': '「',
    '﹂': '」',
    '︱': '｜',
    '︳': '｜',
    '︴': '｜',
    '�': '',  # 不明な文字を削除
}

CHAPTER_PATTERN = re.compile(
    r'^(序章|第[一二三四五六七八九十百]+章|結論|謝\s*辞|目\s*次|訳者解説|原注|参考文献)'
)
SECTION_PATTERN = re.compile(r'^[一二三四五六七八九十百]+[　．、]')
COPYRIGHT_PATTERN = re.compile(
    r"^(?:Copyright\b|©|Published in English\b|published by arrangement\b)|All rights reserved",
    re.IGNORECASE,
)
PAGE_NUMBER_PATTERN = re.compile(r'^\d+\s*$')
# ひらがな・カタカナ・漢字・全角記号に挟まれた半角スペース（全角スペースは対象外）
_JAPANESE_CHAR = r'[\u3001-\u303F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uFF01-\uFF60]'
SPACES_BETWEEN_JAPANESE = re.compile(rf'(?<={_JAPANESE_CHAR})[ \t]+(?={_JAPANESE_CHAR})')


def dedupe_consecutive_pages(pages: List[str]) -> Tuple[List[str], int]:
    """
    連続する同一内容のページをスキップする

    Returns:
        重複を除去したページリストと、スキップしたページ数
    """
    deduped: List[str] = []
    last_hash: Optional[str] = None
    skipped = 0

    for page_text in pages:
        normalized = re.sub(r"\s+", " ", (page_text or "")).strip()
        page_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

        if page_hash == last_hash:
            skipped += 1
            continue

        last_hash = page_hash
        deduped.append(page_text)

    return deduped, skipped


def normalize_special_chars(text: str) -> str:
    """縦書き用の約物を横書きの文字に変換し、不明な文字を削除する"""
    for old, new in VERTICAL_FORMS.items():
        text = text.replace(old, new)
    return text


def remove_spaces_between_japanese(text: str) -> str:
    """
    日本語の文字の間に挟まった半角スペースを除去する

    1文字ずつ配置されたテキスト層（縦書きに多い）から抽出すると
    「吾 輩 は」のように文字間に空白が入るため。英単語の前後の空白と全角スペースは残す。
    """
    return SPACES_BETWEEN_JAPANESE.sub('', text)


def join_paragraph_lines(lines: List[str]) -> str:
    """段落内の行をつなぐ。日本語どうしの境目には空白を入れない"""
    joined = ''
    for line in lines:
        if joined and not (re.match(_JAPANESE_CHAR, joined[-1]) and re.match(_JAPANESE_CHAR, line[0])):
            joined += ' '
        joined += line
    return joined


def format_text_to_markdown(text: str, pdf_filename: str) -> str:
    """
    抽出したテキストをMarkdown形式に整形する

    章（第一章など）は ##、節（一　など）は ###、原書の著作権表記は引用ブロックにまとめる。

    Args:
        text: 抽出されたテキスト
        pdf_filename: PDFファイル名（タイトル用）
    """
    result_lines = [
        f"# {Path(pdf_filename).stem}\n",
        f"（PDF: {Path(pdf_filename).name}）\n",
        "\n---\n\n",
    ]
    current_section: List[str] = []
    seen_sections = set()
    copyright_section: List[str] = []
    in_copyright = False

    def flush_paragraph():
        if current_section:
            result_lines.append(join_paragraph_lines(current_section))
            result_lines.append("\n")
            current_section.clear()

    for line in text.split('\n'):
        if not line.strip():
            flush_paragraph()
            continue

        line = normalize_special_chars(line)
        stripped = line.strip()

        # 原書の著作権表記（本文の Press や Agency は対象外）
        if COPYRIGHT_PATTERN.search(stripped):
            if not in_copyright:
                flush_paragraph()
                in_copyright = True
            copyright_section.append(f"> {stripped}")
            continue
        if in_copyright and copyright_section:
            result_lines.append("\n**原書情報:**\n\n")
            result_lines.append("\n".join(copyright_section))
            result_lines.append("\n\n---\n\n")
            copyright_section = []
            in_copyright = False

        for pattern, kind, mark in ((CHAPTER_PATTERN, "chapter", "##"), (SECTION_PATTERN, "section", "###")):
            if pattern.match(stripped):
                section_key = f"{kind}:{stripped}"
                if section_key not in seen_sections:
                    seen_sections.add(section_key)
                    flush_paragraph()
                    result_lines.append(f"\n{mark} {stripped}\n\n")
                break
        else:
            if not PAGE_NUMBER_PATTERN.match(stripped) and stripped:
                current_section.append(stripped)

    # 文末の著作権表記と段落も出力する
    if copyright_section:
        result_lines.append("\n**原書情報:**\n\n")
        result_lines.append("\n".join(copyright_section))
        result_lines.append("\n")
    if current_section:
        result_lines.append(join_paragraph_lines(current_section))

    return ''.join(result_lines)

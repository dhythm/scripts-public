#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
縦書きPDFから読みやすいMarkdownファイルを生成する統合スクリプト

使い方:
    python3 pdf_to_markdown.py <PDFファイルのパス> [出力ファイルのパス]

例:
    python3 pdf_to_markdown.py book.pdf
    python3 pdf_to_markdown.py book.pdf output.md
"""

import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Any


def get_margin_texts(
    pdf_path: str,
    header_ratio: float = 0.12,
    footer_ratio: float = 0.13
) -> set:
    """
    座標情報を使用してヘッダー/フッター領域にあるテキストを特定する

    pdfminer.sixのextract_pages()を使用して、各テキストボックスの座標を取得し、
    ページ上部・下部のマージン領域にあるテキストを収集する。

    Args:
        pdf_path: PDFファイルのパス
        header_ratio: ページ上部の除外比率（デフォルト5%）
        footer_ratio: ページ下部の除外比率（デフォルト5%）

    Returns:
        ヘッダー/フッター領域にあるテキストのセット
    """
    margin_texts: set = set()

    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LAParams, LTTextBox
    except ImportError:
        return margin_texts

    laparams = LAParams(
        detect_vertical=True,
        all_texts=True
    )

    try:
        for page_layout in extract_pages(pdf_path, laparams=laparams):
            page_height = page_layout.height
            header_boundary = page_height * (1 - header_ratio)
            footer_boundary = page_height * footer_ratio

            for element in page_layout:
                if isinstance(element, LTTextBox):
                    x0, y0, x1, y1 = element.bbox

                    # ヘッダー/フッター領域のテキストを収集
                    if y1 > header_boundary or y0 < footer_boundary:
                        text = element.get_text().strip()
                        if text:
                            # 改行で分割して各行を追加
                            for line in text.split('\n'):
                                line = line.strip()
                                if line:
                                    margin_texts.add(line)

    except Exception as e:
        print(f"⚠ マージンテキスト取得でエラー: {e}")

    return margin_texts


def remove_margin_texts(text: str, margin_texts: set) -> str:
    """
    抽出されたテキストからヘッダー/フッターのテキストを除去する

    Args:
        text: 抽出されたテキスト
        margin_texts: 除去するテキストのセット

    Returns:
        ヘッダー/フッターを除去したテキスト
    """
    if not margin_texts:
        return text

    lines = text.split('\n')
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        # 完全一致でマージンテキストを除去
        if stripped and stripped not in margin_texts:
            filtered_lines.append(line)
        elif not stripped:
            # 空行は保持
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def extract_text_from_pdf(pdf_path: str, exclude_margins: bool = True) -> Optional[str]:
    """
    PDFからテキストを抽出する

    Args:
        pdf_path: PDFファイルのパス
        exclude_margins: ヘッダー/フッター領域を除外するか（デフォルトTrue）

    Returns:
        抽出されたテキスト、失敗時はNone
    """
    # ヘッダー/フッターのテキストを事前に収集
    margin_texts: set = set()
    if exclude_margins:
        print("ヘッダー/フッター領域のテキストを特定中...")
        margin_texts = get_margin_texts(pdf_path)
        if margin_texts:
            print(f"✓ {len(margin_texts)}個のマージンテキストを特定")

    # pdfminer.sixのextract_text()を試す（レイアウト保持）
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams

        print("pdfminer.sixを使用してテキストを抽出中...")

        # LAParams: レイアウト解析のパラメータ
        laparams = LAParams(
            detect_vertical=True,  # 縦書きテキストを検出
            all_texts=True
        )

        text = extract_text(
            pdf_path,
            laparams=laparams
        )

        if text and text.strip():
            # ヘッダー/フッターを除去
            if margin_texts:
                print("ヘッダー/フッターを除去中...")
                text = remove_margin_texts(text, margin_texts)
            print("✓ pdfminer.sixでの抽出に成功")
            return text

    except ImportError:
        print("⚠ pdfminer.sixがインストールされていません")
    except Exception as e:
        print(f"⚠ pdfminer.sixでのエラー: {e}")

    # 次にpdfplumberを試す
    try:
        import pdfplumber

        print("pdfplumberを使用してテキストを抽出中...")

        with pdfplumber.open(pdf_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text(
                    x_tolerance=3,
                    y_tolerance=3,
                    layout=True
                )
                if page_text:
                    text_parts.append(page_text)

            text = '\n\n'.join(text_parts)

            if text and text.strip():
                print("✓ pdfplumberでの抽出に成功")
                return text

    except ImportError:
        print("⚠ pdfplumberがインストールされていません")
    except Exception as e:
        print(f"⚠ pdfplumberでのエラー: {e}")

    # 最後にPyPDF2を試す
    try:
        import PyPDF2

        print("PyPDF2を使用してテキストを抽出中...")

        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            text = '\n\n'.join(text_parts)

            if text and text.strip():
                print("✓ PyPDF2での抽出に成功")
                return text

    except ImportError:
        print("⚠ PyPDF2がインストールされていません")
    except Exception as e:
        print(f"⚠ PyPDF2でのエラー: {e}")

    return None


def normalize_special_chars(text: str) -> str:
    """
    特殊文字を正規化する（縦書き記号を横書きに変換）

    Args:
        text: 入力テキスト

    Returns:
        正規化されたテキスト
    """
    replacements = {
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

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def format_text_to_markdown(text: str, pdf_filename: str) -> str:
    """
    抽出したテキストをMarkdown形式に整形する

    Args:
        text: 抽出されたテキスト
        pdf_filename: PDFファイル名（タイトル用）

    Returns:
        整形されたMarkdownテキスト
    """
    lines = text.split('\n')
    result_lines = []

    # PDFファイル名からタイトルを生成
    title = Path(pdf_filename).stem

    # ヘッダーを追加
    result_lines.append(f"# {title}\n")
    result_lines.append(f"（PDF: {Path(pdf_filename).name}）\n")
    result_lines.append("\n---\n\n")

    current_section = []
    seen_sections = set()  # 重複チェック用
    copyright_section = []
    in_copyright = False

    for line in lines:
        # 空行の処理
        if not line.strip():
            if current_section:
                # セクションの終わりで段落を追加
                result_lines.append(' '.join(current_section))
                result_lines.append("\n")
                current_section = []
            continue

        # 特殊文字を正規化
        line = normalize_special_chars(line)

        # コピーライト情報の検出
        if any(keyword in line for keyword in [
            "Published in English", "Copyright", "All rights reserved",
            "published by arrangement", "Agency", "Press"
        ]):
            if not in_copyright:
                in_copyright = True
            copyright_section.append(f"> {line.strip()}")
            continue
        else:
            # コピーライトセクションの終了
            if in_copyright and copyright_section:
                result_lines.append("\n**原書情報:**\n\n")
                result_lines.extend(copyright_section)
                result_lines.append("\n\n---\n\n")
                copyright_section = []
                in_copyright = False

        # 章タイトルの検出（重複チェック付き）
        chapter_match = re.match(r'^(序章|第[一二三四五六七八九十百]+章|結論|謝\s*辞|目\s*次|訳者解説|原注|参考文献)', line.strip())
        if chapter_match:
            section_key = f"chapter:{line.strip()}"
            if section_key in seen_sections:
                continue  # 重複をスキップ
            seen_sections.add(section_key)

            if current_section:
                result_lines.append(' '.join(current_section))
                result_lines.append("\n")
                current_section = []
            result_lines.append(f"\n## {line.strip()}\n\n")
            continue

        # 節タイトルの検出（一、二、三...）
        section_match = re.match(r'^[一二三四五六七八九十百]+[��、]', line.strip())
        if section_match:
            section_key = f"section:{line.strip()}"
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)

            if current_section:
                result_lines.append(' '.join(current_section))
                result_lines.append("\n")
                current_section = []
            result_lines.append(f"\n### {line.strip()}\n\n")
            continue

        # ページ番号（単独の数字）をスキップ
        if re.match(r'^\d+\s*$', line.strip()):
            continue

        # 通常のテキスト行
        cleaned_line = line.strip()
        if cleaned_line:
            current_section.append(cleaned_line)

    # 最後のセクションを追加
    if current_section:
        result_lines.append(' '.join(current_section))

    return ''.join(result_lines)


def convert_pdf_to_markdown(pdf_path: str, output_path: Optional[str] = None) -> bool:
    """
    PDFファイルをMarkdownファイルに変換する

    Args:
        pdf_path: PDFファイルのパス
        output_path: 出力ファイルのパス（省略時は自動生成）

    Returns:
        成功時True、失敗時False
    """
    # パスの検証
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"✗ エラー: PDFファイルが見つかりません: {pdf_path}")
        return False

    # 出力パスの決定
    if output_path is None:
        output_path = pdf_file.parent / f"{pdf_file.stem}.md"
    else:
        output_path = Path(output_path)

    print(f"\n{'='*60}")
    print(f"PDF→Markdown変換")
    print(f"{'='*60}")
    print(f"入力: {pdf_file}")
    print(f"出力: {output_path}")
    print(f"{'='*60}\n")

    # ステップ1: PDFからテキストを抽出
    print("[1/2] PDFからテキストを抽出中...")
    extracted_text = extract_text_from_pdf(str(pdf_file))

    if not extracted_text:
        print("\n✗ エラー: テキストの抽出に失敗しました")
        print("\n以下のライブラリをインストールしてください:")
        print("  pip install pdfminer.six pdfplumber PyPDF2")
        return False

    print(f"✓ テキスト抽出完了（{len(extracted_text)}文字）\n")

    # ステップ2: Markdownに整形
    print("[2/2] Markdownに整形中...")
    markdown_text = format_text_to_markdown(extracted_text, pdf_file.name)

    # ファイルに保存
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        print(f"✓ Markdown整形完了\n")
    except Exception as e:
        print(f"\n✗ エラー: ファイルの保存に失敗しました: {e}")
        return False

    # 完了メッセージ
    print(f"{'='*60}")
    print(f"✓ 変換完了!")
    print(f"{'='*60}")
    print(f"出力ファイル: {output_path}")
    print(f"ファイルサイズ: {output_path.stat().st_size:,} bytes")
    print(f"{'='*60}\n")

    return True


def main():
    """メイン処理"""
    # コマンドライン引数の処理
    if len(sys.argv) < 2:
        print("使い方: python3 pdf_to_markdown.py <PDFファイル> [出力ファイル]")
        print("\n例:")
        print("  python3 pdf_to_markdown.py book.pdf")
        print("  python3 pdf_to_markdown.py book.pdf output.md")
        print("\n必要なライブラリ:")
        print("  pip install pdfminer.six pdfplumber PyPDF2")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # 変換実行
    success = convert_pdf_to_markdown(pdf_path, output_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

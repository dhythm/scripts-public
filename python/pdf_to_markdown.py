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
from typing import Optional


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    PDFからテキストを抽出する

    Args:
        pdf_path: PDFファイルのパス

    Returns:
        抽出されたテキスト、失敗時はNone
    """
    # まずpdfminer.sixを試す（縦書きに最適）
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

import argparse
import pdfplumber
import sys
from pathlib import Path
import warnings
import logging
from contextlib import redirect_stderr
import io
import hashlib
import re

def extract_text_from_pdf(pdf_path, *, overwrite=False):
    """PDFファイルからテキストを抽出して同じディレクトリに.txtファイルを保存する"""
    
    # pdfplumberの警告を抑制
    warnings.filterwarnings('ignore', message='.*FontBBox.*')
    logging.getLogger('pdfplumber').setLevel(logging.ERROR)
    
    try:
        # 入力ファイルのパスを解析
        input_path = Path(pdf_path)
        if not input_path.exists():
            print(f"エラー: ファイル '{pdf_path}' が見つかりません。")
            return False
        
        if not input_path.suffix.lower() == '.pdf':
            print(f"エラー: '{pdf_path}' はPDFファイルではありません。")
            return False
        
        # 出力ファイルのパスを生成（同じディレクトリに.txt拡張子で）
        output_path = input_path.with_suffix('.txt')
        if output_path.exists() and not overwrite:
            print(f'エラー: 出力ファイルが既に存在します: {output_path}（上書きは --force）')
            return False
        
        # PDFからテキストを抽出（警告を抑制）
        all_text = ""
        last_page_hash = None
        skipped_pages = 0
        with io.StringIO() as buf, redirect_stderr(buf):
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        # 同一ページ内容の連続重複をスキップ（空白差分を吸収）
                        normalized = re.sub(r"\s+", " ", text).strip()
                        page_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
                        if page_hash == last_page_hash:
                            skipped_pages += 1
                            continue
                        last_page_hash = page_hash

                        all_text += f"--- Page {page_number + 1} ---\n"
                        all_text += text + "\n\n"
        
        if not all_text.strip():
            print("エラー: テキストを抽出できませんでした。出力は保存しません。")
            return False

        # テキストをファイルに保存
        with open(output_path, mode='w' if overwrite else 'x', encoding='utf-8') as output_file:
            output_file.write(all_text)
        
        print(f"テキスト抽出完了！結果は「{output_path}」に保存されました。")
        if skipped_pages:
            print(f"重複ページを {skipped_pages} ページスキップしました。")
        
        return True
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='pdf_to_text')
    parser.add_argument('pdf_path')
    parser.add_argument('--force', action='store_true', help='既存の出力ファイルを上書きする')
    args = parser.parse_args()

    success = extract_text_from_pdf(args.pdf_path, overwrite=args.force)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

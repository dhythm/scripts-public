import pdfplumber
import sys
import os
from pathlib import Path
import warnings
import logging
from contextlib import redirect_stderr
import io
import hashlib
import re

def extract_text_from_pdf(pdf_path):
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
        
        # テキストをファイルに保存
        with open(output_path, mode='w', encoding='utf-8') as output_file:
            output_file.write(all_text)
        
        print(f"テキスト抽出完了！結果は「{output_path}」に保存されました。")
        if skipped_pages:
            print(f"重複ページを {skipped_pages} ページスキップしました。")
        
        # 抽出されたテキストが空の場合の警告
        if not all_text.strip():
            print("警告: PDFからテキストが抽出されませんでした。画像ベースのPDFの可能性があります。")
        
        return True
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return False

def main():
    # コマンドライン引数をチェック
    if len(sys.argv) != 2:
        print("使用方法: python pdf_to_text.py <PDFファイルパス>")
        print("例: python pdf_to_text.py document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    extract_text_from_pdf(pdf_path)

if __name__ == "__main__":
    main()

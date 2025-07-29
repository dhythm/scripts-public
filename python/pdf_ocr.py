import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import sys
from pathlib import Path
import warnings
import logging
import cv2
import numpy as np

def pdf_to_images(pdf_path):
    """PDFファイルを画像のリストに変換する"""
    try:
        # PDFを画像に変換（解像度を300dpiに設定してOCRの精度を上げる）
        images = convert_from_path(pdf_path, dpi=300, fmt='png')
        return images
    except Exception as e:
        raise Exception(f"PDFを画像に変換中にエラーが発生しました: {str(e)}")

def preprocess_image(image):
    """OCRの精度を向上させるための画像前処理"""
    # PILイメージをOpenCV形式に変換
    opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # グレースケール変換
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    
    # ノイズ除去
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # コントラスト向上（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 二値化（大津の方法）
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OpenCV形式からPIL形式に戻す
    processed_image = Image.fromarray(binary)
    
    return processed_image

def perform_ocr(image, lang='jpn'):
    """画像に対してOCRを実行する"""
    try:
        # 前処理を適用
        processed_image = preprocess_image(image)
        
        # OCR設定
        custom_config = r'--psm 6 -c preserve_interword_spaces=1'
        
        # 日本語OCRを実行（日本語データがない場合は英語で実行）
        try:
            text = pytesseract.image_to_string(processed_image, lang='jpn', config=custom_config)
        except Exception:
            # 日本語データがない場合は英語で実行し、警告を表示
            print("警告: 日本語OCRデータが見つかりません。英語モードで実行します。")
            print("日本語OCRをインストールするには:")
            print("  macOS: brew install tesseract-lang")
            print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr-jpn")
            text = pytesseract.image_to_string(processed_image, lang='eng', config=custom_config)
        
        return text
    except Exception as e:
        # Tesseractがインストールされていない場合のエラーメッセージ
        if "tesseract is not installed" in str(e).lower():
            raise Exception(
                "Tesseractがインストールされていません。\n"
                "インストール方法:\n"
                "  macOS: brew install tesseract\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-jpn\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki からダウンロード"
            )
        else:
            raise Exception(f"OCR処理中にエラーが発生しました: {str(e)}")

def extract_text_with_ocr(pdf_path):
    """PDFファイルからOCRを使用してテキストを抽出し、.txtファイルに保存する"""
    
    # 警告を抑制
    warnings.filterwarnings('ignore')
    logging.getLogger('pdf2image').setLevel(logging.ERROR)
    
    try:
        # 入力ファイルのパスを解析
        input_path = Path(pdf_path)
        if not input_path.exists():
            print(f"エラー: ファイル '{pdf_path}' が見つかりません。")
            return False
        
        if not input_path.suffix.lower() == '.pdf':
            print(f"エラー: '{pdf_path}' はPDFファイルではありません。")
            return False
        
        # 出力ファイルのパスを生成（同じディレクトリに_ocr.txt拡張子で）
        output_path = input_path.with_suffix('.ocr.txt')
        
        print("PDFを画像に変換中...")
        images = pdf_to_images(pdf_path)
        
        # 各ページに対してOCRを実行
        all_text = ""
        total_pages = len(images)
        
        for page_number, image in enumerate(images):
            print(f"ページ {page_number + 1}/{total_pages} を処理中...")
            
            try:
                text = perform_ocr(image)
                if text.strip():
                    all_text += f"--- Page {page_number + 1} ---\n"
                    all_text += text + "\n\n"
            except Exception as e:
                print(f"ページ {page_number + 1} の処理中にエラーが発生しました: {str(e)}")
                continue
        
        # テキストをファイルに保存
        with open(output_path, mode='w', encoding='utf-8') as output_file:
            output_file.write(all_text)
        
        print(f"\nOCR処理完了！結果は「{output_path}」に保存されました。")
        
        # 抽出されたテキストが空の場合の警告
        if not all_text.strip():
            print("警告: PDFからテキストが抽出されませんでした。画像の品質を確認してください。")
        
        return True
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return False

def check_tesseract():
    """Tesseractがインストールされているかチェックする"""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def main():
    # コマンドライン引数をチェック
    if len(sys.argv) != 2:
        print("使用方法: python pdf_ocr.py <PDFファイルパス>")
        print("例: python pdf_ocr.py document.pdf")
        sys.exit(1)
    
    # Tesseractのインストールをチェック
    if not check_tesseract():
        print("エラー: Tesseractがインストールされていません。")
        print("インストール方法:")
        print("  macOS: brew install tesseract")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-jpn")
        print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki からダウンロード")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    extract_text_with_ocr(pdf_path)

if __name__ == "__main__":
    main()
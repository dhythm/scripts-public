import pytesseract
from pytesseract import TesseractError, TesseractNotFoundError
from pdf2image import convert_from_path
from PIL import Image
import sys
from pathlib import Path
import warnings
import logging
import cv2
import numpy as np


LANG_WARNING_SHOWN = set()


def warn_missing_language(lang):
    """OCR言語データが不足している場合の警告を1度だけ表示"""
    if lang in LANG_WARNING_SHOWN:
        return
    LANG_WARNING_SHOWN.add(lang)

    if lang == 'jpn':
        print("警告: 日本語OCRデータ (jpn) が見つかりません。英語モードで実行します。")
        print("日本語OCRをインストールするには:")
        print("  macOS: brew install tesseract-lang")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr-jpn")
    elif lang == 'jpn_vert':
        print("警告: 縦書き日本語OCRデータ (jpn_vert) が見つかりません。横書きモードで処理を継続します。")
        print("縦書き対応データをインストールするには:")
        print("  macOS: brew install tesseract-lang")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr-jpn")


def _is_japanese_character(char):
    code = ord(char)
    return (
        0x3000 <= code <= 0x303F or  # 記号
        0x3040 <= code <= 0x30FF or  # ひらがな・カタカナ
        0x3400 <= code <= 0x4DBF or  # CJK拡張A
        0x4E00 <= code <= 0x9FFF or  # 漢字
        0xFF01 <= code <= 0xFF60 or  # 全角記号
        0xFF61 <= code <= 0xFF9F     # 半角カナ
    )


def _score_text(text):
    if text is None:
        return (0, 0.0, 0)

    stripped = ''.join(ch for ch in text if not ch.isspace())
    if not stripped:
        return (0, 0.0, 0)

    jp_count = sum(1 for ch in stripped if _is_japanese_character(ch))
    ratio = jp_count / len(stripped)
    return (jp_count, ratio, len(stripped))


def _try_ocr_variant(image, lang, config, warn_missing=True):
    try:
        return pytesseract.image_to_string(image, lang=lang, config=config)
    except TesseractNotFoundError:
        raise
    except (TesseractError, RuntimeError, OSError) as e:
        message = str(e)
        missing_keywords = (
            "Error opening data file",
            "Failed loading language",
            "Missing Tesseract OCR language",
            "training data for language"
        )
        if any(keyword in message for keyword in missing_keywords):
            if warn_missing:
                warn_missing_language(lang)
            return None
        raise Exception(str(e))

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
        processed_image = preprocess_image(image)

        base_config = r'--psm 6 -c preserve_interword_spaces=1'
        vertical_config = r'--psm 5 -c preserve_interword_spaces=1 -c textord_vertical_text=1'

        results = []

        # 横書き（標準）の日本語OCR
        horizontal_text = _try_ocr_variant(processed_image, 'jpn', base_config)
        horizontal_lang = 'jpn'
        if horizontal_text is None:
            horizontal_lang = 'eng'
            horizontal_text = _try_ocr_variant(processed_image, 'eng', base_config, warn_missing=False)
        if horizontal_text is not None:
            results.append({'label': f'{horizontal_lang}_horizontal', 'text': horizontal_text})

        # 縦書き用の日本語OCR（jpn_vert）
        vertical_text = _try_ocr_variant(processed_image, 'jpn_vert', vertical_config)
        if vertical_text is not None:
            results.append({'label': 'jpn_vert_vertical', 'text': vertical_text})

        # ページが回転しているケースにも対応
        for angle in (90, 270):
            rotated_image = processed_image.rotate(angle, expand=True)
            rotated_horizontal = _try_ocr_variant(rotated_image, 'jpn', base_config, warn_missing=False)
            if rotated_horizontal is not None:
                results.append({'label': f'jpn_rot{angle}', 'text': rotated_horizontal})

            rotated_vertical = _try_ocr_variant(rotated_image, 'jpn_vert', vertical_config, warn_missing=False)
            if rotated_vertical is not None:
                results.append({'label': f'jpn_vert_rot{angle}', 'text': rotated_vertical})

        if not results:
            return ""

        best_result = max(results, key=lambda item: _score_text(item['text']))
        return best_result['text']
    except Exception as e:
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

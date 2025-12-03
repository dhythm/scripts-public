
import argparse
from pathlib import Path
from typing import List, Tuple

from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def get_text_from_layout(layout: documentai.Document.Page.Layout, text: str) -> str:
    """Document AIのLayout情報からテキストを抽出する"""
    text_segments = layout.text_anchor.text_segments
    if not text_segments:
        return ""
    
    # 全文から、レイアウトが指すセグメントのテキストを連結して返す
    return "".join(
        text[int(segment.start_index) : int(segment.end_index)]
        for segment in text_segments
    )

def get_bounding_box(bounding_poly: documentai.Document.BoundingPoly) -> Tuple[float, float, float, float]:
    """BoundingPolyから正規化された座標の最小/最大値を取得する"""
    if not bounding_poly or not bounding_poly.normalized_vertices:
        return 0.0, 0.0, 0.0, 0.0
    
    vertices = bounding_poly.normalized_vertices
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    
    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

def process_document_with_bounds(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str,
    y_min: float = 0.0,
    y_max: float = 1.0,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> str:
    """
    PDFドキュメントを処理し、指定された境界内のテキストを抽出する。
    
    Args:
        project_id: Google CloudプロジェクトID
        location: プロセッサのリージョン
        processor_id: Document AIプロセッサID
        file_path: 処理するPDFファイルのパス
        mime_type: ファイルのMIMEタイプ (例: "application/pdf")
        y_min: 処理範囲の上端Y座標 (0.0-1.0)
        y_max: 処理範囲の下端Y座標 (0.0-1.0)
        x_min: 処理範囲の左端X座標 (0.0-1.0)
        x_max: 処理範囲の右端X座標 (0.0-1.0)
        
    Returns:
        抽出されたテキスト
    """
    client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)
    name = client.processor_path(project_id, location, processor_id)

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    image_content = path.read_bytes()
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    document = result.document
    full_text = document.text

    extracted_text = []

    print(f"ドキュメント全体からテキストを抽出中 (ページ数: {len(document.pages)})...")

    for page_num, page in enumerate(document.pages):
        page_text = ""
        print(f"ページ {page_num + 1} を処理中...")
        
        # 縦書きを考慮し、ブロック単位で処理する
        # Document AIは通常、自然な読み順にブロックをソートしてくれる
        for block in page.blocks:
            block_x_min, block_x_max, block_y_min, block_y_max = get_bounding_box(block.layout.bounding_poly)
            
            # ブロックが指定されたY座標範囲内に完全に含まれているかチェック
            # かつ、X座標範囲にも含まれているかチェック
            if (block_y_min >= y_min and block_y_max <= y_max and
                block_x_min >= x_min and block_x_max <= x_max):
                block_text = get_text_from_layout(block.layout, full_text)
                page_text += block_text

        if page_text.strip():
            extracted_text.append(f"--- Page {page_num + 1} (bounds: x=[{x_min:.2f}-{x_max:.2f}], y=[{y_min:.2f}-{y_max:.2f}]) ---\n")
            extracted_text.append(page_text.strip())
            extracted_text.append("\n")

    return "\n".join(extracted_text)


def main():
    parser = argparse.ArgumentParser(
        description="Google Document AIを使用してPDFの指定範囲からテキストを抽出します。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""
実行例:
  python %(prog)s YOUR_PROJECT_ID us YOUR_PROCESSOR_ID document.pdf
  python %(prog)s YOUR_PROJECT_ID jp YOUR_PROCESSOR_ID document.pdf --y-min 0.1 --y-max 0.9 --x-min 0.05 --x-max 0.95

座標は正規化された値（0.0〜1.0）で指定します。
(0, 0)が左上隅、(1, 1)が右下隅です。
- ヘッダーをスキップするには `--y-min` を 0 より大きい値に設定します (例: 0.1)。
- フッターをスキップするには `--y-max` を 1 より小さい値に設定します (例: 0.9)。
"""
    )
    parser.add_argument("project_id", help="Google Cloud プロジェクトID")
    parser.add_argument("location", help="Document AI プロセッサのリージョン (例: 'us' or 'jp')")
    parser.add_argument("processor_id", help="Document AI プロセッサID")
    parser.add_argument("file_path", help="処理対象のPDFファイルパス")
    parser.add_argument("--mime-type", default="application/pdf", help="ファイルのMIMEタイプ (デフォルト: application/pdf)")
    parser.add_argument("--y-min", type=float, default=0.0, help="処理範囲の上端Y座標 (0.0-1.0)")
    parser.add_argument("--y-max", type=float, default=1.0, help="処理範囲の下端Y座標 (0.0-1.0)")
    parser.add_argument("--x-min", type=float, default=0.0, help="処理範囲の左端X座標 (0.0-1.0)")
    parser.add_argument("--x-max", type=float, default=1.0, help="処理範囲の右端X座標 (0.0-1.0)")
    
    args = parser.parse_args()

    # 座標のバリデーション
    if not (0.0 <= args.y_min < args.y_max <= 1.0):
        raise ValueError(f"不正なY座標の範囲です: y_min={args.y_min}, y_max={args.y_max}")
    if not (0.0 <= args.x_min < args.x_max <= 1.0):
        raise ValueError(f"不正なX座標の範囲です: x_min={args.x_min}, x_max={args.x_max}")


    try:
        text = process_document_with_bounds(
            project_id=args.project_id,
            location=args.location,
            processor_id=args.processor_id,
            file_path=args.file_path,
            mime_type=args.mime_type,
            y_min=args.y_min,
            y_max=args.y_max,
            x_min=args.x_min,
            x_max=args.x_max,
        )
        
        if text.strip():
            # 結果を標準出力に表示
            print("\n--- 抽出されたテキスト ---")
            print(text)
            
            # 結果をファイルに保存
            output_path = Path(args.file_path).with_suffix(".layout-ocr.txt")
            output_path.write_text(text, encoding="utf-8")
            print(f"\n結果はファイルにも保存されました: {output_path}")
        else:
            print("\n指定された範囲内にテキストが見つかりませんでした。")

    except FileNotFoundError as e:
        print(f"エラー: {e}")
    except ValueError as e:
        print(f"エラー: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        print("Google Cloudの認証が正しく設定されているか確認してください。")
        print("  - gcloud auth application-default login")


if __name__ == "__main__":
    main()

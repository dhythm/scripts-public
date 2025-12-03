import argparse
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import documentai
from google.cloud import storage


def get_text_from_layout(layout: documentai.Document.Page.Layout, text: str) -> str:
    """Document AIのLayout情報からテキストを抽出する"""
    text_segments = layout.text_anchor.text_segments
    if not text_segments:
        return ""
    return "".join(
        text[int(segment.start_index) : int(segment.end_index)]
        for segment in text_segments
    )

def get_bounding_box(bounding_poly: Optional[documentai.BoundingPoly]) -> Tuple[float, float, float, float]:
    """BoundingPolyから正規化された座標の最小/最大値を取得する"""
    if not bounding_poly or not bounding_poly.normalized_vertices:
        return 0.0, 0.0, 0.0, 0.0
    
    vertices = bounding_poly.normalized_vertices
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    
    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

def create_or_get_bucket(storage_client: storage.Client, bucket_name: str, location: str) -> storage.Bucket:
    """GCSバケットが存在しない場合は作成し、バケットオブジェクトを返す"""
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"GCSバケット '{bucket_name}' は既に存在します。")
    except NotFound:
        print(f"GCSバケット '{bucket_name}' が見つかりません。新規に作成します...")
        try:
            bucket = storage_client.create_bucket(bucket_or_name=bucket_name, location=location)
            print(f"GCSバケット '{bucket_name}' を '{location}' リージョンに作成しました。")
        except GoogleAPICallError as e:
            raise Exception(f"GCSバケットの作成に失敗しました: {e}. リージョン '{location}' がGCSで有効か確認してください。")
    return bucket

def cleanup_gcs_folder(bucket: storage.Bucket, prefix: str):
    """GCSフォルダ内のファイルをすべて削除する"""
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        return
        
    print(f"GCSフォルダ '{prefix}' 内のファイルをクリーンアップ中...")
    for blob in blobs:
        blob.delete()
    print(f"'{prefix}' 内の {len(blobs)} 個のファイルを削除しました。")


def batch_process_document_with_bounds(
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
    PDFドキュメントを非同期で処理し、指定された境界内のテキストを抽出する。
    """
    storage_client = storage.Client()
    docai_client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    docai_client = documentai.DocumentProcessorServiceClient(client_options=docai_client_options)

    gcs_location = location if location not in ['jp'] else 'US'
    bucket_name = f"{project_id}-docai-batch-processing"
    bucket = create_or_get_bucket(storage_client, bucket_name, gcs_location)

    gcs_input_prefix = "input/"
    gcs_output_prefix = "output/"
    
    extracted_text = []

    try:
        # ---- GCSの準備とアップロード ----
        cleanup_gcs_folder(bucket, gcs_input_prefix)
        cleanup_gcs_folder(bucket, gcs_output_prefix)

        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
        gcs_input_uri = f"gs://{bucket_name}/{gcs_input_prefix}{path.name}"
        blob = bucket.blob(f"{gcs_input_prefix}{path.name}")
        print(f"ファイルをGCSにアップロード中: {gcs_input_uri}")
        blob.upload_from_filename(file_path)

        # ---- 非同期処理の実行 ----
        gcs_document = documentai.GcsDocument(gcs_uri=gcs_input_uri, mime_type=mime_type)
        batch_documents_input_config = documentai.BatchDocumentsInputConfig(
            gcs_documents=documentai.GcsDocuments(documents=[gcs_document])
        )

        gcs_output_uri = f"gs://{bucket_name}/{gcs_output_prefix}"
        gcs_output_config = documentai.DocumentOutputConfig(
            gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(gcs_uri=gcs_output_uri)
        )
        
        name = docai_client.processor_path(project_id, location, processor_id)
        request = documentai.BatchProcessRequest(
            name=name,
            input_documents=batch_documents_input_config,
            document_output_config=gcs_output_config,
        )

        print("Document AIによる非同期処理を開始します。完了まで数分かかることがあります...")
        operation = docai_client.batch_process_documents(request)
        
        try:
            operation.result(timeout=1800)
            print("非同期処理が完了しました。")
        except Exception as e:
            # 非同期処理が失敗した場合、詳細なエラー情報を取得する
            metadata = documentai.BatchProcessMetadata.deserialize(operation.metadata.value)
            error_message = f"Document AIの非同期処理に失敗しました: {e}\n"
            
            for process in metadata.individual_process_statuses:
                # 0 (OK) でない場合、エラー詳細を出力
                if process.status.code != 0:
                    error_message += f"  -失敗したドキュメント: {process.input_gcs_source}\n"
                    error_message += f"  -エラーコード: {process.status.code}\n"
                    error_message += f"  -エラー詳細: {process.status.message}\n"
            raise Exception(error_message)

        # ---- 結果のダウンロードと解析 ----
        print("GCSから処理結果をダウンロード中...")
        output_blobs = list(bucket.list_blobs(prefix=gcs_output_prefix))
        if not output_blobs:
            raise Exception("処理結果が見つかりませんでした。")

        for output_blob in output_blobs:
            if ".json" not in output_blob.name:
                continue
            
            json_string = output_blob.download_as_text()
            document = documentai.Document.from_json(json_string)
            full_text = document.text

            print(f"結果を解析中: {output_blob.name} (ページ数: {len(document.pages)})")

            for page in document.pages:
                page_text = ""
                for block in page.blocks:
                    block_x_min, block_x_max, block_y_min, block_y_max = get_bounding_box(block.layout.bounding_poly)
                    if (block_y_min >= y_min and block_y_max <= y_max and
                        block_x_min >= x_min and block_x_max <= x_max):
                        block_text = get_text_from_layout(block.layout, full_text)
                        page_text += block_text

                if page_text.strip():
                    extracted_text.append(f"--- Page {page.page_number} (bounds: x=[{x_min:.2f}-{x_max:.2f}], y=[{y_min:.2f}-{y_max:.2f}]) ---\n")
                    extracted_text.append(page_text.strip())
                    extracted_text.append("\n")
    
    finally:
        # ---- クリーンアップ ----
        print("クリーンアップ処理を開始します...")
        cleanup_gcs_folder(bucket, gcs_input_prefix)
        cleanup_gcs_folder(bucket, gcs_output_prefix)
        print("クリーンアップが完了しました。")

    return "\n".join(extracted_text)


def main():
    parser = argparse.ArgumentParser(
        description="Google Document AIを使用してPDFの指定範囲からテキストを抽出します（非同期処理）。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
実行例:
  python %(prog)s YOUR_PROJECT_ID us YOUR_PROCESSOR_ID document.pdf
  python %(prog)s YOUR_PROJECT_ID jp YOUR_PROCESSOR_ID document.pdf --y-min 0.1 --y-max 0.9

このスクリプトは、大きなPDFファイルを処理するために非同期APIを使用します。
処理中、ファイルは一時的にGoogle Cloud Storageバケットにアップロードされます。
バケットは `{PROJECT_ID}-docai-batch-processing` という名前で自動的に作成されます。
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

    if not (0.0 <= args.y_min < args.y_max <= 1.0):
        raise ValueError(f"不正なY座標の範囲です: y_min={args.y_min}, y_max={args.y_max}")
    if not (0.0 <= args.x_min < args.x_max <= 1.0):
        raise ValueError(f"不正なX座標の範囲です: x_min={args.x_min}, x_max={args.x_max}")

    try:
        text = batch_process_document_with_bounds(
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
            print("\n--- 抽出されたテキスト ---")
            print(text)
            
            output_path = Path(args.file_path).with_suffix(".batch-layout-ocr.txt")
            output_path.write_text(text, encoding="utf-8")
            print(f"\n結果はファイルにも保存されました: {output_path}")
        else:
            print("\n指定された範囲内にテキストが見つかりませんでした。")

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"エラーが発生しました: {e}")
        print("Google Cloudの認証情報やプロジェクトID、ファイルパスが正しいか確認してください。")

if __name__ == "__main__":
    main()
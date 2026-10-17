import argparse
import os
import tempfile
from contextlib import contextmanager
import uuid
from pathlib import Path
from typing import Tuple, Optional

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import documentai
from google.cloud import storage


def get_text_from_layout(
    layout: documentai.Document.Page.Layout, text: str, text_offset: int = 0
) -> str:
    """Resolve shard-local anchors against the reconstructed document text."""
    parts = []
    for segment in layout.text_anchor.text_segments:
        start = int(segment.start_index) + text_offset
        end = int(segment.end_index) + text_offset
        if not 0 <= start <= end <= len(text):
            raise ValueError(f"テキスト参照が範囲外です: [{start}, {end}), length={len(text)}")
        parts.append(text[start:end])
    return "".join(parts)


def prepare_document_shards(document_shards: list[documentai.Document]):
    """Validate a complete shard set and retain each page's local text origin.

    TextSegment indices are local to a shard. ShardInfo.text_offset maps them
    to global text, including anchors that extend across a shard boundary.
    """
    if not document_shards:
        raise ValueError("Document AIのシャードがありません。")
    ordered = sorted(document_shards, key=lambda doc: int(doc.shard_info.shard_index))
    text_parts = []
    pages_with_offsets = []
    expected_offset = 0
    for index, document in enumerate(ordered):
        info = document.shard_info
        if len(ordered) > 1 or info.shard_count:
            if int(info.shard_count) != len(ordered) or int(info.shard_index) != index:
                raise ValueError("Document AIのシャードが不足または重複しています。")
        offset = int(info.text_offset)
        if offset != expected_offset:
            raise ValueError(f"シャードのテキスト位置が不連続です: expected={expected_offset}, actual={offset}")
        if document.error.code:
            raise ValueError(f"Document AIのシャード処理エラー: {document.error.message}")
        text_parts.append(document.text)
        pages_with_offsets.extend((page, offset) for page in document.pages)
        expected_offset += len(document.text)
    pages_with_offsets.sort(key=lambda item: item[0].page_number)
    return "".join(text_parts), pages_with_offsets


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

def cleanup_run_blobs(blobs):
    """Delete only the exact object generations observed during this run."""
    for blob in blobs:
        try:
            if blob.generation is None:
                print(f"世代番号が不明なため保持します: gs://{blob.bucket.name}/{blob.name}")
                continue
            blob.delete(if_generation_match=blob.generation)
        except Exception as error:
            # Cleanup must not hide a processing error or discard extracted text.
            print(f"一時ファイルを削除できませんでした ({blob.name}): {error}")


@contextmanager
def atomic_text_output(output_path: Optional[Path]):
    """Reserve a private sibling temp file and publish without replacing a file.

    The hard link publishes fully written bytes atomically and fails if another
    process creates the destination after preflight. Only our temp is removed.
    """
    if output_path is None:
        yield None
        return
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"出力先が既に存在します（上書きしません）: {output_path}")
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent,
        prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
    )
    temp_path = Path(handle.name)
    try:
        def save_text(text):
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temp_path, output_path)
        yield save_text
    finally:
        try:
            handle.close()
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as error:
                print(f"一時ファイルを削除できませんでした ({temp_path}): {error}")


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
    output_path: Optional[Path] = None,
) -> str:
    """Preflight local output before cloud work; clean GCS only after saving."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    with atomic_text_output(output_path) as save_output:
        return _batch_process_document_with_bounds(
            project_id, location, processor_id, file_path, mime_type,
            y_min, y_max, x_min, x_max, save_output=save_output,
        )


def _batch_process_document_with_bounds(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str,
    y_min: float = 0.0,
    y_max: float = 1.0,
    x_min: float = 0.0,
    x_max: float = 1.0,
    save_output=None,
) -> str:
    """
    PDFドキュメントを非同期で処理し、指定された境界内のテキストを抽出する。
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    storage_client = storage.Client()
    docai_client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    docai_client = documentai.DocumentProcessorServiceClient(client_options=docai_client_options)

    gcs_location = location if location not in ['jp'] else 'US'
    bucket_name = f"{project_id}-docai-batch-processing"
    bucket = create_or_get_bucket(storage_client, bucket_name, gcs_location)

    run_prefix = f"runs/{uuid.uuid4().hex}/"
    gcs_input_prefix = f"{run_prefix}input/"
    gcs_output_prefix = f"{run_prefix}output/"
    owned_blobs = []
    processing_succeeded = False
    
    extracted_text = []

    try:
        # ---- GCSの準備とアップロード ----
        gcs_input_uri = f"gs://{bucket_name}/{gcs_input_prefix}{path.name}"
        blob = bucket.blob(f"{gcs_input_prefix}{path.name}")
        print(f"ファイルをGCSにアップロード中: {gcs_input_uri}")
        blob.upload_from_filename(file_path, if_generation_match=0)
        owned_blobs.append(blob)

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
            # GAPIC operations expose typed metadata; do not mask the original
            # exception if metadata is absent or cannot be decoded.
            metadata = operation.metadata
            if not isinstance(metadata, documentai.BatchProcessMetadata):
                try:
                    metadata = documentai.BatchProcessMetadata.deserialize(metadata.value)
                except Exception:
                    metadata = documentai.BatchProcessMetadata()
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
        output_blobs = sorted(list(bucket.list_blobs(prefix=gcs_output_prefix)), key=lambda b: b.name)
        if not output_blobs:
            raise Exception("処理結果が見つかりませんでした。")

        owned_blobs.extend(output_blobs)
        json_blobs = [b for b in output_blobs if b.name.endswith(".json")]
        if not json_blobs:
            raise Exception("JSON形式の処理結果が見つかりませんでした。")
        
        # 1. 全てのシャードをパースしてメモリに読み込む
        document_shards = [documentai.Document.from_json(blob.download_as_text()) for blob in json_blobs]

        # Shard-local anchors must be translated by their own text offset.
        full_text, pages_with_offsets = prepare_document_shards(document_shards)

        print(f"結果を解析中...（合計{len(pages_with_offsets)}ページ）")

        # 5. ソート済みのページリストを処理する
        for page, text_offset in pages_with_offsets:
            print(f"ページ {page.page_number} を処理中...")
            page_text = ""
            for block in page.blocks:
                block_x_min, block_x_max, block_y_min, block_y_max = get_bounding_box(block.layout.bounding_poly)
                if (block_y_min >= y_min and block_y_max <= y_max and
                    block_x_min >= x_min and block_x_max <= x_max):
                    block_text = get_text_from_layout(block.layout, full_text, text_offset)
                    page_text += block_text

            if page_text.strip():
                extracted_text.append(f"--- Page {page.page_number} (bounds: x=[{x_min:.2f}-{x_max:.2f}], y=[{y_min:.2f}-{y_max:.2f}]) ---\n")
                extracted_text.append(page_text.strip())
                extracted_text.append("\n")
    
        result = "\n".join(extracted_text)
        if result.strip() and save_output is not None:
            save_output(result)
        processing_succeeded = True
    finally:
        if processing_succeeded:
            cleanup_run_blobs(owned_blobs)
        else:
            # Preserve input and partial output for diagnosis or recovery. A failed
            # or timed-out operation may still be writing to its own prefix.
            print(f"処理に失敗したため一時ファイルを保持します: gs://{bucket_name}/{run_prefix}")

    return result


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
実行ごとに固有の runs/<ID>/ を使用し、成功時は当該実行で確認した世代だけ削除します。
失敗時は診断・復旧用に入力と部分出力を保持し、保存先を表示します。
出力先が既に存在する場合は処理前に終了します。保存成功後にGCSをクリーンアップします。
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

    output_path = Path(args.file_path).with_suffix(".batch-layout-ocr.txt")
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
            output_path=output_path,
        )
        
        if text.strip():
            print("\n--- 抽出されたテキスト ---")
            print(text)
            
            print(f"\n結果はファイルにも保存されました: {output_path}")
        else:
            print("\n指定された範囲内にテキストが見つかりませんでした。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("Google Cloudの認証情報やプロジェクトID、ファイルパスが正しいか確認してください。")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""Offline safety regression tests. No Google client can make a network call."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import pdf_ocr_with_layout as ocr


@pytest.fixture
def run_env(monkeypatch, tmp_path):
    source = tmp_path / "input.pdf"
    source.write_bytes(b"test fixture, not uploaded")
    bucket = Mock(name="bucket")
    bucket.name = "project-docai-batch-processing"
    uploaded = Mock(generation=11)
    uploaded.name = "uploaded"
    uploaded.bucket = bucket
    bucket.blob.return_value = uploaded
    output = Mock(generation=22)
    output.name = "result.json"
    output.bucket = bucket
    output.download_as_text.return_value = "{}"
    bucket.list_blobs.return_value = [output]
    storage_ctor = Mock()
    docai_ctor = Mock()
    docai_ctor.return_value.processor_path.return_value = "projects/project/locations/us/processors/processor"
    operation = docai_ctor.return_value.batch_process_documents.return_value
    monkeypatch.setattr(ocr.storage, "Client", storage_ctor)
    monkeypatch.setattr(ocr.documentai, "DocumentProcessorServiceClient", docai_ctor)
    monkeypatch.setattr(ocr, "create_or_get_bucket", Mock(return_value=bucket))
    monkeypatch.setattr(ocr.documentai.Document, "from_json", Mock(return_value=ocr.documentai.Document(text="", pages=[])))
    return SimpleNamespace(source=source, bucket=bucket, uploaded=uploaded, output=output,
                           storage=storage_ctor, docai=docai_ctor, operation=operation)


def process(env):
    return ocr.batch_process_document_with_bounds("project", "us", "processor", str(env.source), "application/pdf")


def test_missing_input_does_not_construct_cloud_clients(run_env):
    run_env.source.unlink()
    with pytest.raises(FileNotFoundError):
        process(run_env)
    run_env.storage.assert_not_called()
    run_env.docai.assert_not_called()


def test_success_only_deletes_observed_generations_and_uses_unique_prefixes(run_env):
    process(run_env)
    process(run_env)
    names = [call.args[0] for call in run_env.bucket.blob.call_args_list]
    assert names[0] != names[1]
    assert all(name.startswith("runs/") and name.endswith("/input/input.pdf") for name in names)
    for call in run_env.bucket.list_blobs.call_args_list:
        assert call.kwargs["prefix"].startswith("runs/")
        assert call.kwargs["prefix"].endswith("/output/")
    run_env.uploaded.upload_from_filename.assert_called_with(str(run_env.source), if_generation_match=0)
    run_env.uploaded.delete.assert_called_with(if_generation_match=11)
    run_env.output.delete.assert_called_with(if_generation_match=22)
    assert run_env.uploaded.delete.call_count == 2
    assert run_env.output.delete.call_count == 2


@pytest.mark.parametrize("failure", ["upload", "parse", "no-output", "operation"])
def test_failure_keeps_input_and_partial_output(run_env, monkeypatch, capsys, failure):
    if failure == "upload":
        run_env.uploaded.upload_from_filename.side_effect = RuntimeError("upload failed")
    elif failure == "parse":
        run_env.output.download_as_text.side_effect = RuntimeError("download failed")
    elif failure == "no-output":
        run_env.bucket.list_blobs.return_value = []
    else:
        run_env.operation.result.side_effect = TimeoutError("pending")
        monkeypatch.setattr(ocr.documentai.BatchProcessMetadata, "deserialize", Mock(return_value=SimpleNamespace(individual_process_statuses=[])))
    expected = {"upload": "upload failed", "parse": "download failed", "no-output": "処理結果が見つかりません", "operation": "pending"}[failure]
    with pytest.raises(Exception, match=expected):
        process(run_env)
    run_env.uploaded.delete.assert_not_called()
    run_env.output.delete.assert_not_called()
    assert "gs://project-docai-batch-processing/runs/" in capsys.readouterr().out


def test_cleanup_failure_does_not_hide_extracted_result(run_env, capsys):
    run_env.output.delete.side_effect = RuntimeError("generation changed")
    assert process(run_env) == ""
    assert "generation changed" in capsys.readouterr().out


def test_unknown_generation_is_never_deleted(run_env):
    run_env.output.generation = None
    process(run_env)
    run_env.output.delete.assert_not_called()

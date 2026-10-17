"""Local-save regression tests; all cloud clients are mocked."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import pdf_ocr_with_layout as ocr
from test_pdf_ocr_gcs_safety import run_env as cloud_environment

# Reuse the cloud-isolation fixture without sharing any real credentials.
run_env = cloud_environment


def nonempty_document(env):
    ocr.documentai.Document.from_json.return_value = ocr.documentai.Document(
        text="保存したテキスト", pages=[{"page_number": 1, "blocks": [{"layout": {
            "text_anchor": {"text_segments": [{"end_index": len("保存したテキスト")}]}
        }}]}],
    )


def process(env, output):
    return ocr.batch_process_document_with_bounds(
        "project", "us", "processor", str(env.source), "application/pdf", output_path=output,
    )


def test_existing_output_fails_before_cloud_and_keeps_bytes(run_env):
    output = run_env.source.with_suffix(".txt")
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        process(run_env, output)
    assert output.read_text() == "keep me"
    run_env.storage.assert_not_called()
    run_env.docai.assert_not_called()


def test_dangling_output_symlink_is_not_replaced(run_env):
    output = run_env.source.with_suffix(".txt")
    output.symlink_to(output.parent / "missing-target")
    with pytest.raises(FileExistsError):
        process(run_env, output)
    assert output.is_symlink()
    run_env.storage.assert_not_called()


def test_invalid_parent_is_rejected_before_cloud(run_env):
    with pytest.raises(FileNotFoundError):
        process(run_env, run_env.source.parent / "missing" / "output.txt")
    run_env.storage.assert_not_called()


def test_complete_output_is_published_before_any_gcs_cleanup(run_env):
    nonempty_document(run_env)
    output = run_env.source.with_suffix(".txt")
    def check_saved(**kwargs):
        assert "保存したテキスト" in output.read_text(encoding="utf-8")
    run_env.uploaded.delete.side_effect = check_saved
    run_env.output.delete.side_effect = check_saved
    result = process(run_env, output)
    assert output.read_text(encoding="utf-8") == result
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []
    run_env.uploaded.delete.assert_called_once_with(if_generation_match=11)
    run_env.output.delete.assert_called_once_with(if_generation_match=22)


def test_disk_write_failure_keeps_gcs_and_removes_owned_temp(run_env, monkeypatch, capsys):
    nonempty_document(run_env)
    output = run_env.source.with_suffix(".txt")
    monkeypatch.setattr(ocr.os, "fsync", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        process(run_env, output)
    assert not output.exists()
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []
    run_env.uploaded.delete.assert_not_called()
    run_env.output.delete.assert_not_called()
    assert "gs://project-docai-batch-processing/runs/" in capsys.readouterr().out


def test_concurrent_destination_creation_is_not_overwritten(run_env, monkeypatch):
    nonempty_document(run_env)
    output = run_env.source.with_suffix(".txt")
    original_link = ocr.os.link
    def competing_save(source, destination):
        Path(destination).write_text("other process", encoding="utf-8")
        original_link(source, destination)
    monkeypatch.setattr(ocr.os, "link", competing_save)
    with pytest.raises(FileExistsError):
        process(run_env, output)
    assert output.read_text() == "other process"
    run_env.uploaded.delete.assert_not_called()
    run_env.output.delete.assert_not_called()
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_cloud_constructor_failure_removes_local_temp(run_env):
    output = run_env.source.with_suffix(".txt")
    run_env.storage.side_effect = RuntimeError("credentials unavailable")
    with pytest.raises(RuntimeError, match="credentials unavailable"):
        process(run_env, output)
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []
    assert not output.exists()


def test_cli_reports_failure_with_nonzero_status(run_env, monkeypatch):
    output = run_env.source.with_suffix(".batch-layout-ocr.txt")
    output.write_text("existing")
    monkeypatch.setattr(sys, "argv", ["ocr", "project", "us", "processor", str(run_env.source)])
    assert ocr.main() == 1
    assert output.read_text() == "existing"
    run_env.storage.assert_not_called()


def test_cli_saves_successfully_and_returns_zero(run_env, monkeypatch):
    nonempty_document(run_env)
    output = run_env.source.with_suffix(".batch-layout-ocr.txt")
    monkeypatch.setattr(sys, "argv", ["ocr", "project", "us", "processor", str(run_env.source)])
    assert ocr.main() == 0
    assert "保存したテキスト" in output.read_text(encoding="utf-8")


def test_empty_bounded_result_is_success_without_output_file(run_env):
    output = run_env.source.with_suffix(".txt")
    assert process(run_env, output) == ""
    assert not output.exists()
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []
    run_env.output.delete.assert_called_once_with(if_generation_match=22)


def test_cli_process_exit_code_is_one_for_missing_input(tmp_path):
    import subprocess
    result = subprocess.run(
        [sys.executable, "-B", str(Path(ocr.__file__)), "project", "us", "processor", str(tmp_path / "missing.pdf")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "ファイルが見つかりません" in result.stdout

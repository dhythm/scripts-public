"""Document AI protobuf fixtures; no clients, models or network are used."""
import sys
from pathlib import Path

import pytest
from google.cloud import documentai

sys.path.insert(0, str(Path(__file__).parent.parent))
from pdf_ocr_with_layout import get_text_from_layout, prepare_document_shards


def layout(*segments):
    return documentai.Document.Page.Layout(text_anchor={
        "text_segments": [{"start_index": start, "end_index": end} for start, end in segments]
    })


def shard(text, index, count, offset, page_number, segments):
    return documentai.Document(
        text=text,
        shard_info={"shard_index": index, "shard_count": count, "text_offset": offset},
        pages=[{"page_number": page_number, "blocks": [{"layout": layout(*segments)}]}],
    )


def extract(shards):
    text, pages = prepare_document_shards(shards)
    return [(page.page_number, get_text_from_layout(page.blocks[0].layout, text, offset))
            for page, offset in pages]


def test_later_shards_use_their_own_text_and_local_anchor_origin():
    first = shard("first", 0, 2, 0, 1, [(0, 5)])
    second = shard("second", 1, 2, 5, 2, [(0, 6)])
    assert extract([second, first]) == [(1, "first"), (2, "second")]


def test_unicode_offsets_count_characters_and_multiple_segments():
    first = shard("日本語\n", 0, 2, 0, 1, [(0, 3)])
    second = shard("後半😀です", 1, 2, 4, 2, [(0, 2), (3, 5)])
    assert extract([first, second]) == [(1, "日本語"), (2, "後半です")]


def test_anchor_can_cross_shard_boundary():
    first = shard("abc", 0, 2, 0, 1, [(1, 5)])
    second = shard("def", 1, 2, 3, 2, [(0, 3)])
    assert extract([first, second]) == [(1, "bcde"), (2, "def")]


def test_pages_are_sorted_without_losing_their_shard_origin():
    first = shard("one", 0, 2, 0, 2, [(0, 3)])
    second = shard("two", 1, 2, 3, 1, [(0, 3)])
    assert extract([first, second]) == [(1, "two"), (2, "one")]


def test_unsharded_document_and_omitted_zero_start_index():
    doc = documentai.Document(text="plain", pages=[{"page_number": 1, "blocks": [{
        "layout": {"text_anchor": {"text_segments": [{"end_index": 5}]}}
    }]}])
    assert extract([doc]) == [(1, "plain")]
    assert get_text_from_layout(layout(), "plain") == ""


@pytest.mark.parametrize("shards", [
    [],
    [shard("a", 0, 2, 0, 1, [(0, 1)])],
    [shard("a", 0, 2, 0, 1, [(0, 1)]), shard("b", 0, 2, 1, 2, [(0, 1)])],
    [shard("a", 0, 2, 0, 1, [(0, 1)]), shard("b", 1, 2, 2, 2, [(0, 1)])],
])
def test_incomplete_duplicate_and_discontinuous_shards_fail_explicitly(shards):
    with pytest.raises(ValueError):
        prepare_document_shards(shards)


def test_anchor_outside_complete_document_is_not_silently_truncated():
    with pytest.raises(ValueError, match="範囲外"):
        extract([shard("abc", 0, 1, 0, 1, [(0, 4)])])


def test_shard_processing_error_is_not_successful_text():
    doc = shard("abc", 0, 1, 0, 1, [(0, 3)])
    doc.error.code = 13
    doc.error.message = "internal processing error"
    with pytest.raises(ValueError, match="internal processing error"):
        prepare_document_shards([doc])

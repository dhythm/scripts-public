"""fetch_viral_english_posts のユーティリティテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fetch_viral_english_posts import (
    PRESETS,
    _normalize_posts,
    _parse_json_payload,
    _to_int,
)


def test_parse_json_payload_strips_markdown_fence() -> None:
    raw = '```json\n{"posts": []}\n```'
    assert _parse_json_payload(raw) == {"posts": []}


def test_parse_json_payload_extracts_embedded_object() -> None:
    raw = 'Here is the result:\n{"posts": [{"rank": 1}]}'
    parsed = _parse_json_payload(raw)
    assert len(parsed["posts"]) == 1


def test_normalize_posts_fills_url_from_citations() -> None:
    raw = {
        "posts": [
            {
                "rank": 1,
                "author_handle": "elonmusk",
                "post_text": "Hello",
                "estimated_likes": 50000,
                "topic": "AI",
                "why_viral": "Big news",
            }
        ]
    }
    citations = ["https://x.com/i/status/1234567890"]
    posts = _normalize_posts(raw, citations)
    assert posts[0].post_url == citations[0]
    assert posts[0].author_handle == "elonmusk"
    assert posts[0].estimated_likes == 50000


def test_to_int_handles_strings() -> None:
    assert _to_int("1200") == 1200
    assert _to_int(None) is None
    assert _to_int("n/a") is None


def test_presets_are_non_empty() -> None:
    assert "general" in PRESETS
    assert "ai" in PRESETS


def test_build_user_prompt_includes_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    from fetch_viral_english_posts import _build_user_prompt

    prompt = _build_user_prompt(
        preset="ai",
        topic="Claude",
        count=5,
        min_likes=10000,
        from_date="2026-05-01",
        to_date="2026-05-29",
    )
    assert "Claude" in prompt
    assert "5" in prompt

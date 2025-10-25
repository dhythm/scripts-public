from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .models import CrawlSummary, PageData


def export_json(pages: Iterable[PageData], path: Path) -> None:
    serializable = [page.model_dump(mode="json") for page in pages]
    with path.open("w", encoding="utf-8") as fp:
        json.dump(serializable, fp, ensure_ascii=False, indent=2)


def export_csv(pages: Iterable[PageData], path: Path) -> None:
    fieldnames = [
        "url",
        "status_code",
        "title",
        "description",
        "word_count",
        "heading_count",
        "image_count",
        "fetched_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for page in pages:
            writer.writerow(
                {
                    "url": str(page.url),
                    "status_code": page.status_code,
                    "title": page.title or "",
                    "description": page.description or "",
                    "word_count": page.word_count,
                    "heading_count": len(page.headings),
                    "image_count": len(page.images),
                    "fetched_at": page.fetched_at.isoformat(),
                }
            )


def export_markdown(pages: List[PageData], summary: CrawlSummary, path: Path) -> None:
    structure = build_site_structure(pages)

    lines = [
        "# Webサイト分析レポート",
        "",
        f"- クロール開始: {summary.start_time.isoformat()}",
        f"- クロール終了: {summary.end_time.isoformat()}",
        f"- 所要時間: {summary.duration_seconds:.1f} 秒",
        f"- 取得ページ数: {summary.total_pages}",
        f"- 取得画像数: {summary.total_images}",
        "",
        "## サイト構造",
        "",
    ]

    lines.extend(render_structure_markdown(structure))

    lines.extend(
        [
            "",
            "## ページ一覧",
            "",
        ]
    )

    for page in pages:
        lines.extend(
            [
                f"### {page.title or '(タイトル不明)'}",
                f"- URL: {page.url}",
                f"- ステータス: {page.status_code}",
                f"- 見出し数: {len(page.headings)}",
                f"- 画像数: {len(page.images)}",
                f"- 要約: {page.text_preview or ''}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_structure_json(pages: List[PageData], path: Path) -> None:
    structure = build_site_structure(pages)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(structure, fp, ensure_ascii=False, indent=2)


def build_site_structure(pages: List[PageData]) -> List[dict[str, Any]]:
    page_lookup = {str(page.url): page for page in pages}
    children_map: dict[Optional[str], List[PageData]] = defaultdict(list)

    for page in pages:
        parent = page.parent_url
        if parent is not None and parent not in page_lookup:
            parent = None
        children_map[parent].append(page)

    def sort_key(page: PageData) -> tuple[int, str]:
        return (page.depth, str(page.url))

    def build_nodes(parent_key: Optional[str]) -> List[dict[str, Any]]:
        nodes: List[dict[str, Any]] = []
        for child in sorted(children_map.get(parent_key, []), key=sort_key):
            child_url = str(child.url)
            node = {
                "url": child_url,
                "title": child.title,
                "status_code": child.status_code,
                "depth": child.depth,
                "children": build_nodes(child_url),
            }
            nodes.append(node)
        return nodes

    return build_nodes(None)


def render_structure_markdown(structure: List[dict[str, Any]]) -> List[str]:
    lines: List[str] = []

    def _render(nodes: List[dict[str, Any]], depth: int) -> None:
        indent = "  " * depth
        for node in nodes:
            title = node.get("title") or "(タイトル不明)"
            url = node.get("url")
            lines.append(f"{indent}- {title} ({url})")
            children = node.get("children") or []
            if children:
                _render(children, depth + 1)

    _render(structure, 0)
    if not lines:
        lines.append("(取得されたページがありません)")
    return lines

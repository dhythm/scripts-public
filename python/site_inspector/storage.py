from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

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
    lines = [
        "# Webサイト分析レポート",
        "",
        f"- クロール開始: {summary.start_time.isoformat()}",
        f"- クロール終了: {summary.end_time.isoformat()}",
        f"- 所要時間: {summary.duration_seconds:.1f} 秒",
        f"- 取得ページ数: {summary.total_pages}",
        f"- 取得画像数: {summary.total_images}",
        "",
        "## ページ一覧",
        "",
    ]

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

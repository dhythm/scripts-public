from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .models import CrawlSummary, PageData


def build_summary(pages: Iterable[PageData], *, start_time: datetime, end_time: datetime) -> CrawlSummary:
    pages_list = list(pages)
    total_images = sum(len(page.images) for page in pages_list)
    duration = (end_time - start_time).total_seconds()
    return CrawlSummary(
        total_pages=len(pages_list),
        total_images=total_images,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
    )

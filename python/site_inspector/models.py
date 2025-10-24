from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, HttpUrl


class ImageData(BaseModel):
    src: str
    absolute_url: Optional[HttpUrl] = None
    alt_text: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    downloaded_path: Optional[str] = None


class PageMetadata(BaseModel):
    name: str
    value: str


class PageData(BaseModel):
    url: HttpUrl
    depth: int
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    fetched_at: datetime
    title: Optional[str] = None
    description: Optional[str] = None
    canonical_url: Optional[str] = None
    headings: List[str] = Field(default_factory=list)
    meta: List[PageMetadata] = Field(default_factory=list)
    text_preview: Optional[str] = None
    word_count: int = 0
    links: List[str] = Field(default_factory=list)
    images: List[ImageData] = Field(default_factory=list)
    error: Optional[str] = None

    def hostname(self) -> Optional[str]:
        return urlparse(str(self.url)).hostname


class CrawlSummary(BaseModel):
    total_pages: int
    total_images: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float

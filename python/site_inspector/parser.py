from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .models import ImageData, PageData, PageMetadata


def sanitize_text(text: str, *, max_length: int = 600) -> str:
    clean = " ".join(text.split())
    if len(clean) > max_length:
        return clean[: max_length - 3] + "..."
    return clean


class HTMLParser:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def parse(
        self,
        *,
        url: str,
        html: str,
        depth: int,
        status_code: int | None,
        content_type: str | None,
        fetched_at: datetime,
    ) -> Tuple[PageData, List[str]]:
        soup = BeautifulSoup(html, "lxml")

        title_node = soup.find("title")
        title = title_node.get_text(strip=True) if title_node else None

        description = None
        description_meta = soup.find("meta", attrs={"name": "description"})
        if description_meta and description_meta.get("content"):
            description = description_meta["content"].strip()

        canonical = None
        canonical_link = soup.find("link", attrs={"rel": ["canonical", "alternate"]})
        if canonical_link and canonical_link.get("href"):
            canonical = urljoin(url, canonical_link["href"])

        headings = [
            sanitize_text(node.get_text(separator=" ", strip=True))
            for node in soup.find_all(["h1", "h2", "h3"])
        ]

        meta_entries = []
        for node in soup.find_all("meta"):
            key = node.get("name") or node.get("property") or node.get("http-equiv")
            content = node.get("content")
            if key and content:
                meta_entries.append(PageMetadata(name=key, value=content))

        text_content = sanitize_text(soup.get_text(separator=" ", strip=True), max_length=1200)
        word_count = len(text_content.split())

        image_entries = self._extract_images(soup, url)
        links = self._extract_links(soup, url)

        page = PageData(
            url=url,
            depth=depth,
            status_code=status_code,
            content_type=content_type,
            fetched_at=fetched_at,
            title=title,
            description=description,
            canonical_url=canonical,
            headings=headings,
            meta=meta_entries,
            text_preview=text_content,
            word_count=word_count,
            links=links,
            images=image_entries,
        )
        return page, links

    def _extract_images(self, soup: BeautifulSoup, page_url: str) -> List[ImageData]:
        images: List[ImageData] = []
        for node in soup.find_all("img"):
            src = node.get("src")
            if not src:
                continue
            absolute = urljoin(page_url, src)
            alt_text = node.get("alt")
            width = self._parse_int(node.get("width"))
            height = self._parse_int(node.get("height"))

            images.append(
                ImageData(
                    src=src,
                    absolute_url=absolute,
                    alt_text=alt_text,
                    width=width,
                    height=height,
                )
            )
        return images

    def _extract_links(self, soup: BeautifulSoup, page_url: str) -> List[str]:
        links: List[str] = []
        for node in soup.find_all("a"):
            href = node.get("href")
            if not href:
                continue
            absolute = urljoin(page_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                links.append(absolute)
        return links

    @staticmethod
    def _parse_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

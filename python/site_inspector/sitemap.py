from __future__ import annotations

import asyncio
from collections import deque
from typing import Iterable, List, Set
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx


class SitemapFetcher:
    def __init__(self, client: httpx.AsyncClient, *, timeout: int = 10) -> None:
        self._client = client
        self._timeout = timeout

    async def fetch(self, urls: Iterable[str], *, limit: int = 2000) -> List[str]:
        discovered: List[str] = []
        visited: Set[str] = set()
        queue: deque[str] = deque(urls)

        while queue and len(discovered) < limit:
            sitemap_url = queue.popleft()
            if sitemap_url in visited:
                continue
            visited.add(sitemap_url)
            try:
                response = await self._client.get(sitemap_url, timeout=self._timeout)
                response.raise_for_status()
            except httpx.HTTPError:
                continue

            content_type = response.headers.get("content-type", "")
            if "xml" not in content_type:
                continue
            try:
                root = ElementTree.fromstring(response.content)
            except ElementTree.ParseError:
                continue

            namespace = self._detect_namespace(root)
            if root.tag.endswith("sitemapindex"):
                for child in root.findall(f".//{{{namespace}}}sitemap") if namespace else root.findall(".//sitemap"):
                    loc = child.find(f"{{{namespace}}}loc") if namespace else child.find("loc")
                    if loc is not None and loc.text:
                        queue.append(loc.text.strip())
            else:
                for child in root.findall(f".//{{{namespace}}}url") if namespace else root.findall(".//url"):
                    loc = child.find(f"{{{namespace}}}loc") if namespace else child.find("loc")
                    if loc is not None and loc.text:
                        url = loc.text.strip()
                        if url and self._is_http_url(url):
                            discovered.append(url)
                            if len(discovered) >= limit:
                                break
        return discovered

    @staticmethod
    def _detect_namespace(root: ElementTree.Element) -> str | None:
        if "}" in root.tag:
            return root.tag.split("}")[0].strip("{")
        return None

    @staticmethod
    def _is_http_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


async def fetch_sitemap_urls(client: httpx.AsyncClient, urls: Iterable[str], *, limit: int = 2000) -> List[str]:
    fetcher = SitemapFetcher(client)
    return await fetcher.fetch(urls, limit=limit)

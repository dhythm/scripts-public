from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import List, Optional, Set, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import CrawlConfig
from .images import ImageDownloader
from .models import CrawlSummary, PageData
from .parser import HTMLParser
from .reporting import build_summary
from .sitemap import fetch_sitemap_urls
from .storage import export_csv, export_json, export_markdown

console = Console()


def normalize_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


class RobotsHandler:
    def __init__(self, config: CrawlConfig) -> None:
        self._config = config
        self._parser: Optional[RobotFileParser] = None
        self._sitemaps: List[str] = []

    @property
    def sitemaps(self) -> List[str]:
        return self._sitemaps

    def can_fetch(self, url: str) -> bool:
        if not self._config.respect_robots:
            return True
        if self._parser is None:
            return True
        return self._parser.can_fetch(self._config.user_agent, url)

    async def prepare(self, client: httpx.AsyncClient) -> None:
        if not self._config.respect_robots:
            return

        robots_url = urljoin(str(self._config.base_url), "/robots.txt")
        parser = RobotFileParser()
        try:
            response = await client.get(robots_url, timeout=self._config.request_timeout)
            if response.status_code >= 400:
                return
            content = response.text
        except httpx.HTTPError:
            return

        parser.parse(content.splitlines())
        self._parser = parser
        site_maps = parser.site_maps()
        if site_maps:
            self._sitemaps = list(site_maps)


class Crawler:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self._allowed_domains = config.normalized_domains()
        self._blacklist: List[re.Pattern[str]] = [re.compile(pat) for pat in config.blacklist_patterns]
        self._visited: Set[str] = set()
        self._results: List[PageData] = []
        self._pages_crawled = 0
        self._parser = HTMLParser(str(config.base_url))
        self._client: Optional[httpx.AsyncClient] = None
        self._robots = RobotsHandler(config)
        self._image_downloader: Optional[ImageDownloader] = None

    async def run(self) -> Tuple[List[PageData], dict]:
        self.config.ensure_output_dirs()

        headers = {"User-Agent": self.config.user_agent}

        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            self._client = client
            await self._robots.prepare(client)
            if self.config.output.download_images and self.config.output.image_directory:
                self._image_downloader = ImageDownloader(
                    client,
                    self.config.output.image_directory,
                    timeout=self.config.request_timeout,
                    concurrency=max(1, self.config.concurrency // 2),
                )

            start_time = datetime.utcnow()
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]クロール中...[/] {task.completed}/{task.total} ページ"),
                console=console,
                transient=True,
            )
            with progress:
                task_id = progress.add_task("crawl", total=self.config.max_pages)
                await self._crawl(progress, task_id)
            end_time = datetime.utcnow()

        summary = build_summary(self._results, start_time=start_time, end_time=end_time)
        self._export_results(summary)
        return self._results, summary.model_dump()

    async def _crawl(self, progress: Progress, task_id: int) -> None:
        assert self._client is not None

        queue: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
        seeds = await self._initial_urls()
        for seed in seeds:
            await queue.put((seed, 0))

        workers = [
            asyncio.create_task(self._worker(queue, progress, task_id))
            for _ in range(self.config.concurrency)
        ]

        await queue.join()
        for worker in workers:
            worker.cancel()

    async def _initial_urls(self) -> Set[str]:
        seeds: Set[str] = {normalize_url(str(self.config.base_url))}
        if self.config.follow_sitemap and self._robots.sitemaps:
            sitemap_urls = await fetch_sitemap_urls(
                self._client,
                self._robots.sitemaps,
                limit=self.config.max_pages * 2,
            )
            for url in sitemap_urls:
                seeds.add(normalize_url(url))
        return seeds

    async def _worker(self, queue: asyncio.Queue[Tuple[str, int]], progress: Progress, task_id: int) -> None:
        while True:
            url, depth = await queue.get()
            try:
                if self._pages_crawled >= self.config.max_pages:
                    continue
                if not self._should_visit(url, depth):
                    continue
                await self._process_url(url, depth, queue, progress, task_id)
            finally:
                queue.task_done()

    def _should_visit(self, url: str, depth: int) -> bool:
        if depth > self.config.max_depth:
            return False
        normalized = normalize_url(url)
        if normalized in self._visited:
            return False
        if not self._is_allowed_domain(normalized):
            return False
        if self._is_blacklisted(normalized):
            return False
        if not self._robots.can_fetch(normalized):
            return False
        return True

    def _is_allowed_domain(self, url: str) -> bool:
        hostname = urlsplit(url).hostname or ""
        hostname = hostname.lower()
        for domain in self._allowed_domains:
            if hostname == domain or hostname.endswith(f".{domain}"):
                return True
        return False

    def _is_blacklisted(self, url: str) -> bool:
        return any(pattern.search(url) for pattern in self._blacklist)

    async def _process_url(
        self,
        url: str,
        depth: int,
        queue: asyncio.Queue[Tuple[str, int]],
        progress: Progress,
        task_id: int,
    ) -> None:
        assert self._client is not None

        normalized = normalize_url(url)
        self._visited.add(normalized)

        fetched_at = datetime.utcnow()
        try:
            response = await self._client.get(
                normalized,
                timeout=self.config.request_timeout,
            )
            status_code = response.status_code
            content_type = response.headers.get("content-type", "")
            text = response.text if "html" in content_type else None
        except httpx.HTTPError as exc:
            page = PageData(
                url=normalized,
                depth=depth,
                status_code=None,
                content_type=None,
                fetched_at=fetched_at,
                error=str(exc),
            )
            self._results.append(page)
            return

        if text:
            page, links = self._parser.parse(
                url=normalized,
                html=text,
                depth=depth,
                status_code=status_code,
                content_type=content_type,
                fetched_at=fetched_at,
            )
        else:
            page = PageData(
                url=normalized,
                depth=depth,
                status_code=status_code,
                content_type=content_type,
                fetched_at=fetched_at,
                text_preview=None,
                word_count=0,
                links=[],
            )
            links = []

        if self.config.output.download_images and self._image_downloader:
            await self._download_images(page)

        self._results.append(page)
        self._pages_crawled += 1
        progress.update(task_id, completed=self._pages_crawled)

        await asyncio.sleep(self.config.delay_seconds)

        if depth < self.config.max_depth and self._pages_crawled < self.config.max_pages:
            for link in links:
                if self._should_visit(link, depth + 1):
                    await queue.put((link, depth + 1))

    async def _download_images(self, page: PageData) -> None:
        assert self._image_downloader is not None
        tasks = [self._image_downloader.download(image) for image in page.images]
        if not tasks:
            return
        downloaded = await asyncio.gather(*tasks)
        page.images = list(downloaded)

    def _export_results(self, summary: CrawlSummary) -> None:
        output_dir = self.config.output.directory
        pages = self._results

        if self.config.output.write_json:
            export_json(pages, output_dir / "pages.json")
        if self.config.output.write_csv:
            export_csv(pages, output_dir / "pages.csv")
        if self.config.output.write_markdown:
            export_markdown(pages, summary, output_dir / "summary.md")

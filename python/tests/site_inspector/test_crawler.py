"""Crawler regression tests; every HTTP request uses an in-memory transport."""

import asyncio
from unittest.mock import Mock

import httpx

from site_inspector.config import CrawlConfig
from site_inspector.crawler import Crawler


def crawl(seeds, *, max_pages=20, concurrency=1, handler=None, parser=None):
    async def run():
        config = CrawlConfig(
            base_url="https://example.com",
            max_pages=max_pages,
            concurrency=concurrency,
            delay_seconds=0,
            respect_robots=False,
            follow_sitemap=False,
        )
        crawler = Crawler(config)

        async def initial_urls():
            return seeds

        crawler._initial_urls = initial_urls
        if parser is not None:
            crawler._parser = parser
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            crawler._client = client
            tasks_before = asyncio.all_tasks()
            await asyncio.wait_for(crawler._crawl(Mock(), 0), timeout=2)
            # Workers must be cancelled and awaited before the crawl returns.
            assert asyncio.all_tasks() == tasks_before
        return crawler

    return asyncio.run(run())


def test_parser_error_is_recorded_and_remaining_queue_is_drained():
    parser = Mock()
    parser.parse.side_effect = RuntimeError("invalid markup")

    def handler(request):
        if request.url.path == "/bad":
            return httpx.Response(200, text="<p>bad</p>", headers={"Content-Type": "text/html"})
        return httpx.Response(200, content=b"text", headers={"Content-Type": "text/plain"})

    crawler = crawl(["https://example.com/bad", "https://example.com/good"], handler=handler, parser=parser)
    assert len(crawler._results) == 2
    assert crawler._results[0].error == "RuntimeError: invalid markup"
    assert crawler._results[1].error is None
    assert crawler._pages_crawled == 2


def test_inflight_requests_do_not_exceed_page_limit():
    requests = []

    async def handler(request):
        requests.append(str(request.url))
        # Make all workers overlap before any response completes.
        await asyncio.sleep(0.01)
        return httpx.Response(200, text="page", headers={"Content-Type": "text/plain"})

    crawler = crawl(
        [f"https://example.com/{i}" for i in range(10)],
        max_pages=3,
        concurrency=5,
        handler=handler,
    )
    assert len(requests) == len(crawler._results) == crawler._pages_crawled == 3


def test_failed_requests_also_consume_page_limit():
    requests = []

    def handler(request):
        requests.append(str(request.url))
        raise httpx.ConnectError("unavailable", request=request)

    crawler = crawl(
        [f"https://example.com/{i}" for i in range(5)], max_pages=2, concurrency=3, handler=handler
    )
    assert len(requests) == len(crawler._results) == crawler._pages_crawled == 2
    assert all(page.error == "unavailable" for page in crawler._results)


def test_duplicate_urls_are_only_fetched_once():
    requests = []

    async def handler(request):
        requests.append(str(request.url))
        await asyncio.sleep(0)
        return httpx.Response(200, text="page", headers={"Content-Type": "text/plain"})

    crawler = crawl(
        ["https://example.com/a", "https://example.com/a#fragment"], concurrency=2, handler=handler
    )
    assert len(requests) == len(crawler._results) == 1

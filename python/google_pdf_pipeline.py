#!/usr/bin/env python3
"""Google検索からPDFを取得し、テキスト抽出とLLM整形を行うユーティリティ。"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import httpx
import pdfplumber
from openai import AsyncOpenAI
from dotenv import load_dotenv
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

logger = logging.getLogger(__name__)

GOOGLE_SEARCH_API = "https://www.googleapis.com/customsearch/v1"


@dataclass
class PdfSearchResult:
    """Googleカスタム検索のPDF検索結果。"""

    title: str
    link: str
    snippet: Optional[str] = None


@dataclass
class PdfProcessingResult:
    """PDFの処理結果。"""

    title: str
    url: str
    snippet: Optional[str]
    raw_text: Optional[str]
    formatted_text: Optional[str]
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GooglePdfSearcher:
    """Googleカスタム検索APIを利用してPDFを検索する。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None,
        *,
        timeout: float = 20.0,
        default_language: Optional[str] = "lang_ja",
        safe: str = "off",
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_CSE_ID")
        if not self.api_key or not self.search_engine_id:
            raise ValueError(
                "Google検索APIキーと検索エンジンIDが設定されていません。"
                "環境変数 GOOGLE_API_KEY と GOOGLE_CSE_ID を設定してください。"
            )
        self.timeout = timeout
        self.default_language = default_language
        self.safe = safe

    async def search_pdfs(
        self,
        query: str,
        *,
        num_results: int = 3,
        language: Optional[str] = None,
    ) -> list[PdfSearchResult]:
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": f"{query} filetype:pdf",
            "fileType": "pdf",
            "num": max(1, min(num_results, 10)),
            "safe": self.safe,
        }
        if language or self.default_language:
            params["lr"] = language or self.default_language

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(GOOGLE_SEARCH_API, params=params)
            response.raise_for_status()
            payload = response.json()

        items: Iterable[dict[str, Any]] = payload.get("items", []) or []
        results: list[PdfSearchResult] = []
        for item in items:
            link = item.get("link")
            title = item.get("title")
            if not link or not title:
                continue
            snippet = item.get("snippet")
            results.append(PdfSearchResult(title=title, link=link, snippet=snippet))
        return results


class PlaywrightPdfFetcher:
    """Playwrightを用いてPDFをダウンロードする。"""

    def __init__(self, *, headless: bool = True, timeout_ms: float = 20000.0) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._playwright_cm = None
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self) -> "PlaywrightPdfFetcher":
        self._playwright_cm = async_playwright()
        self._playwright = await self._playwright_cm.__aenter__()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(accept_downloads=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright_cm:
            await self._playwright_cm.__aexit__(exc_type, exc_val, exc_tb)

    async def fetch_pdf(self, url: str) -> bytes:
        if not self._context:
            raise RuntimeError("PlaywrightPdfFetcherはコンテキストマネージャとして使用してください。")

        page: Page = await self._context.new_page()
        try:
            response = await page.goto(url, wait_until="networkidle", timeout=self.timeout_ms)
            if response:
                body = await response.body()
                content_type = (response.headers.get("content-type") or "").lower()
                if "pdf" in content_type or body.startswith(b"%PDF"):
                    return body

            try:
                download = await page.wait_for_event("download", timeout=self.timeout_ms)
                temp_path = await download.path()
                if temp_path:
                    return Path(temp_path).read_bytes()
            except PlaywrightTimeoutError:
                pass

            raise RuntimeError("PDFのダウンロードに失敗しました。")
        finally:
            await page.close()


class PdfTextExtractor:
    """PDFバイト列からテキストを抽出する。"""

    def __init__(
        self,
        *,
        use_ocr_fallback: bool = True,
        ocr_languages: str = "jpn+eng",
    ) -> None:
        self.use_ocr_fallback = use_ocr_fallback
        self.ocr_languages = ocr_languages

    async def extract_text(self, data: bytes) -> str:
        text = await asyncio.to_thread(self._extract_text_with_pdfplumber, data)
        if text.strip():
            return text

        if not self.use_ocr_fallback:
            return text

        try:
            ocr_text = await asyncio.to_thread(self._extract_text_with_ocr, data)
        except Exception as error:  # pylint: disable=broad-except
            logger.debug("OCR抽出に失敗しました: %s", error)
            return text

        return ocr_text

    def _extract_text_with_pdfplumber(self, data: bytes) -> str:
        buffer = io.BytesIO(data)
        with pdfplumber.open(buffer) as pdf:
            texts = []
            for page in pdf.pages:
                content = page.extract_text() or ""
                content = content.strip()
                if content:
                    texts.append(content)
        return "\n\n".join(texts)

    def _extract_text_with_ocr(self, data: bytes) -> str:
        from pdf2image import convert_from_bytes  # 遅延インポート
        import pytesseract  # 遅延インポート

        images = convert_from_bytes(data, fmt="png")
        texts: list[str] = []
        for image in images:
            text = pytesseract.image_to_string(image, lang=self.ocr_languages)
            text = text.strip()
            if text:
                texts.append(text)
        return "\n\n".join(texts)


class LlmFormatter:
    """OpenAI SDKを使用してテキストを整形する。"""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI APIキーが設定されていません。環境変数 OPENAI_API_KEY を設定してください。")
        self.client = AsyncOpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.system_prompt = (
            system_prompt
            or "あなたはPDFから抽出されたテキストを整形し、読みやすい日本語の要約と重要ポイントを提供するアシスタントです。"
        )

    async def format_text(
        self,
        text: str,
        *,
        metadata: Optional[PdfSearchResult] = None,
        extra_instructions: Optional[Sequence[str]] = None,
    ) -> str:
        if not text.strip():
            return ""

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        user_content_lines = [
            "以下のPDFテキストを簡潔に整形してください。",
            "入力の要点を3〜5個の箇条書きでまとめ、日本語で回答してください。",
        ]

        if metadata:
            user_content_lines.append(f"タイトル: {metadata.title}")
            user_content_lines.append(f"URL: {metadata.link}")
            if metadata.snippet:
                user_content_lines.append(f"スニペット: {metadata.snippet}")

        if extra_instructions:
            user_content_lines.extend(extra_instructions)

        user_content_lines.append("=== PDFテキスト ===")
        user_content_lines.append(text.strip())

        messages.append({"role": "user", "content": "\n".join(user_content_lines)})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        choice = response.choices[0]
        return choice.message.content or ""


class GooglePdfProcessor:
    """検索から整形までの処理をまとめるファサード。"""

    def __init__(
        self,
        *,
        searcher: Optional[GooglePdfSearcher] = None,
        fetcher: Optional[PlaywrightPdfFetcher] = None,
        extractor: Optional[PdfTextExtractor] = None,
        formatter: Optional[LlmFormatter] = None,
    ) -> None:
        self.searcher = searcher or GooglePdfSearcher()
        self.fetcher = fetcher or PlaywrightPdfFetcher()
        self.extractor = extractor or PdfTextExtractor()
        self.formatter = formatter

    async def process(
        self,
        query: str,
        *,
        limit: int = 3,
        format_with_llm: bool = True,
        extra_instructions: Optional[Sequence[str]] = None,
    ) -> list[PdfProcessingResult]:
        search_results = await self.searcher.search_pdfs(query, num_results=limit)
        if not search_results:
            return []

        results: list[PdfProcessingResult] = []

        async with self.fetcher as fetcher:
            for item in search_results:
                try:
                    pdf_bytes = await fetcher.fetch_pdf(item.link)
                except Exception as error:  # pylint: disable=broad-except
                    logger.warning("PDF取得に失敗しました (%s): %s", item.link, error)
                    results.append(
                        PdfProcessingResult(
                            title=item.title,
                            url=item.link,
                            snippet=item.snippet,
                            raw_text=None,
                            formatted_text=None,
                            error=str(error),
                        )
                    )
                    continue

                raw_text = await self.extractor.extract_text(pdf_bytes)
                formatted = None

                if format_with_llm and self.formatter:
                    try:
                        formatted = await self.formatter.format_text(
                            raw_text,
                            metadata=item,
                            extra_instructions=extra_instructions,
                        )
                    except Exception as error:  # pylint: disable=broad-except
                        logger.warning("LLM整形に失敗しました (%s): %s", item.link, error)
                        results.append(
                            PdfProcessingResult(
                                title=item.title,
                                url=item.link,
                                snippet=item.snippet,
                                raw_text=raw_text,
                                formatted_text=None,
                                error=f"LLM加工エラー: {error}",
                            )
                        )
                        continue

                results.append(
                    PdfProcessingResult(
                        title=item.title,
                        url=item.link,
                        snippet=item.snippet,
                        raw_text=raw_text,
                        formatted_text=formatted,
                    )
                )

        return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Google検索で見つけたPDFをダウンロードし、テキスト抽出・整形を行います。",
    )
    parser.add_argument("query", help="Google検索クエリ")
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=3,
        help="処理するPDFの件数 (1〜10)",
    )
    parser.add_argument(
        "--no-format",
        action="store_true",
        help="LLMによる整形をスキップします。",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="使用するOpenAIモデル（LLM整形を行う場合）。",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="追加の整形指示（LLM整形を行う場合）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="結果を保存するJSONファイルパス。",
    )
    parser.add_argument(
        "--ocr-lang",
        default="jpn+eng",
        help="OCRフォールバック時に使用する言語設定。",
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="OCRフォールバックを無効化します。",
    )
    parser.add_argument(
        "--playwright-timeout",
        type=float,
        default=20000.0,
        help="Playwrightのタイムアウト (ミリ秒)。",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Playwrightをヘッド有効（ブラウザ表示あり）で実行します。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを出力します。",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    searcher = GooglePdfSearcher()
    fetcher = PlaywrightPdfFetcher(
        headless=not args.headed,
        timeout_ms=args.playwright_timeout,
    )
    extractor = PdfTextExtractor(
        use_ocr_fallback=not args.disable_ocr,
        ocr_languages=args.ocr_lang,
    )
    formatter: Optional[LlmFormatter] = None
    if not args.no_format:
        formatter = LlmFormatter(model=args.model)

    pipeline = GooglePdfProcessor(
        searcher=searcher,
        fetcher=fetcher,
        extractor=extractor,
        formatter=formatter,
    )

    extra_instructions = [args.prompt] if args.prompt else None

    results = await pipeline.process(
        args.query,
        limit=args.limit,
        format_with_llm=not args.no_format,
        extra_instructions=extra_instructions,
    )

    output_payload = [result.to_dict() for result in results]
    serialized = json.dumps(output_payload, ensure_ascii=False, indent=2)

    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
        print(f"結果を {args.output} に保存しました。")
    else:
        print(serialized)


def main(argv: Optional[Sequence[str]] = None) -> None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

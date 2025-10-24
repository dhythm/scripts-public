from __future__ import annotations

import asyncio
import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .models import ImageData


class ImageDownloader:
    def __init__(
        self,
        client: httpx.AsyncClient,
        directory: Path,
        *,
        timeout: int = 20,
        concurrency: int = 4,
    ) -> None:
        self._client = client
        self._directory = directory
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(concurrency)

    async def download(self, image: ImageData) -> ImageData:
        if image.absolute_url is None:
            return image

        async with self._semaphore:
            try:
                response = await self._client.get(
                    str(image.absolute_url),
                    timeout=self._timeout,
                    headers={"Referer": str(image.absolute_url)},
                )
                response.raise_for_status()
            except httpx.HTTPError:
                return image

        content_type = response.headers.get("content-type")
        extension = self._guess_extension(image, content_type)
        filename = self._build_filename(image, extension)
        file_path = self._directory / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(response.content)

        image.size_bytes = len(response.content)
        image.mime_type = content_type
        image.downloaded_path = str(file_path)
        return image

    def _build_filename(self, image: ImageData, extension: str | None) -> str:
        hashed = hashlib.sha1(str(image.absolute_url).encode("utf-8")).hexdigest()
        if extension is None:
            return f"{hashed}"
        return f"{hashed}{extension}"

    @staticmethod
    def _guess_extension(image: ImageData, content_type: str | None) -> str | None:
        if content_type:
            extension = mimetypes.guess_extension(content_type.split(";")[0])
            if extension:
                return extension
        parsed = urlparse(str(image.absolute_url))
        path = Path(parsed.path)
        if path.suffix:
            return path.suffix
        return None

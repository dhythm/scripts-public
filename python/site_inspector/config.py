from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, FieldValidationInfo, HttpUrl, field_validator, model_validator


class OutputConfig(BaseModel):
    directory: Path = Field(default=Path("./reports/output"))
    write_json: bool = True
    write_csv: bool = True
    write_markdown: bool = True
    download_images: bool = False
    image_directory: Optional[Path] = None

    @field_validator("image_directory")
    @classmethod
    def _default_image_directory(cls, value: Optional[Path], info: FieldValidationInfo) -> Optional[Path]:
        if value is None:
            directory = info.data.get("directory")
            download_images = info.data.get("download_images")
            if isinstance(directory, Path):
                if download_images:
                    return directory / "images"
        return value

    @model_validator(mode="after")
    def _ensure_image_directory(self) -> "OutputConfig":
        # download_images が有効な場合は image_directory を必ず設定する
        if self.download_images and self.image_directory is None:
            self.image_directory = self.directory / "images"
        return self


class PlaywrightConfig(BaseModel):
    enabled: bool = False
    max_pages: int = 20


class CrawlConfig(BaseModel):
    base_url: HttpUrl
    user_agent: str = "SiteInspectorBot/0.1"
    max_depth: int = 2
    max_pages: int = 200
    concurrency: int = 5
    request_timeout: int = 10
    delay_seconds: float = 0.2
    respect_robots: bool = True
    follow_sitemap: bool = True
    allowed_domains: List[str] = Field(default_factory=list)
    blacklist_patterns: List[str] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)
    playwright: PlaywrightConfig = Field(default_factory=PlaywrightConfig)

    @model_validator(mode="after")
    def _populate_allowed_domains(self) -> "CrawlConfig":
        if not self.allowed_domains:
            hostname = urlparse(str(self.base_url)).hostname
            if hostname:
                self.allowed_domains = [hostname]
        return self

    @field_validator("max_depth")
    @classmethod
    def _validate_depth(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_depth は 0 以上を指定してください。")
        return value

    @field_validator("concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError("concurrency は 1 以上を指定してください。")
        return value

    def normalized_domains(self) -> set[str]:
        return {domain.lower() for domain in self.allowed_domains}

    def ensure_output_dirs(self) -> None:
        self.output.directory.mkdir(parents=True, exist_ok=True)
        if self.output.download_images:
            if self.output.image_directory is None:
                self.output.image_directory = self.output.directory / "images"
            self.output.image_directory.mkdir(parents=True, exist_ok=True)

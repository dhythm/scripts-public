from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .config import CrawlConfig
from .crawler import Crawler

app = typer.Typer(help="企業サイト向けの構成分析・情報収集ツール")
console = Console()


def load_config_from_path(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise typer.BadParameter(f"設定ファイルが見つかりません: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise typer.BadParameter("設定ファイルの形式が不正です。")
    return data


@app.command()
def crawl(
    url: Optional[str] = typer.Argument(
        None,
        help="クロール対象のベースURL（設定ファイルで指定する場合は省略可能）",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="YAML形式の設定ファイル",
    ),
    depth: Optional[int] = typer.Option(None, "--depth", help="クロール最大深度の上書き"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", help="取得ページ最大数の上書き"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="出力ディレクトリの上書き"),
    download_images: Optional[bool] = typer.Option(
        None,
        "--download-images/--no-download-images",
        help="画像ダウンロードの有効/無効を切り替え",
    ),
    respect_robots: Optional[bool] = typer.Option(
        None,
        "--respect-robots/--ignore-robots",
        help="robots.txt の遵守設定を上書き",
    ),
) -> None:
    config_dict = load_config_from_path(config_path)

    if url:
        config_dict["base_url"] = url
    elif "base_url" not in config_dict:
        raise typer.BadParameter("URL を指定するか、設定ファイルに base_url を記載してください。")

    try:
        config = CrawlConfig(**config_dict)
    except Exception as exc:
        raise typer.BadParameter(f"設定の読み込みに失敗しました: {exc}") from exc

    update_data = {}
    if depth is not None:
        update_data["max_depth"] = depth
    if max_pages is not None:
        update_data["max_pages"] = max_pages
    if respect_robots is not None:
        update_data["respect_robots"] = respect_robots

    if update_data:
        config = config.model_copy(update=update_data)

    output_update = {}
    if output_dir is not None:
        output_update["directory"] = output_dir
    if download_images is not None:
        output_update["download_images"] = download_images

    if output_update:
        config = config.model_copy(
            update={"output": config.output.model_copy(update=output_update)}
        )

    try:
        pages, summary = asyncio.run(_run_crawler(config))
    except Exception as exc:
        console.print(f"[red]クロールに失敗しました:[/] {exc}")
        raise typer.Exit(code=1) from exc

    _print_summary(summary)
    console.print(f"[green]完了しました。レポート出力先: {config.output.directory}[/]")
    console.print(f"取得済みページ: {len(pages)}")


async def _run_crawler(config: CrawlConfig):
    crawler = Crawler(config)
    return await crawler.run()


def _print_summary(summary: dict) -> None:
    table = Table(title="クロール結果サマリ")
    table.add_column("項目")
    table.add_column("値", justify="right")
    for key, label in [
        ("total_pages", "取得ページ数"),
        ("total_images", "取得画像数"),
        ("duration_seconds", "処理時間（秒）"),
        ("start_time", "開始時刻"),
        ("end_time", "終了時刻"),
    ]:
        table.add_row(label, str(summary.get(key, "")))
    console.print(table)


if __name__ == "__main__":
    app()

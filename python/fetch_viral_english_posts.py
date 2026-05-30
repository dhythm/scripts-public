#!/usr/bin/env python3
"""
xAI Grok API の X Search で、海外でバズった英語投稿を収集する CLI。

X API の Bearer トークンは不要。XAI_API_KEY のみで動作します。
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

load_dotenv()

app = typer.Typer(
    add_completion=False,
    help="xAI X Search でバズった英語投稿を収集します。",
)
console = Console()

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4.3"

# X 検索演算子の目安（x_search はセマンティック検索のため、プロンプト用ガイドとして使用）
PRESETS: dict[str, dict[str, str]] = {
    "general": {
        "label": "海外で伸びた英語投稿（汎用）",
        "guidance": "English posts with very high engagement (roughly 20,000+ likes or equivalent virality). Exclude replies. Prefer original posts.",
        "x_query_hint": "lang:en min_faves:20000 -is:reply",
    },
    "ai": {
        "label": "AI / GPT / agents 界隈",
        "guidance": "Viral English posts about AI, GPT, LLMs, or AI agents. High engagement (roughly 10,000+ likes).",
        "x_query_hint": "(AI OR GPT OR agents) lang:en min_faves:10000 -is:reply",
    },
    "startup": {
        "label": "スタートアップ / SaaS / founder",
        "guidance": "Viral English posts about startups, SaaS, founders, or building companies. Roughly 10,000+ likes.",
        "x_query_hint": "(startup OR SaaS OR founder) lang:en min_faves:10000 -is:reply",
    },
    "media": {
        "label": "メディア付きバズ投稿",
        "guidance": "Viral English posts with images or video. Very high engagement (roughly 50,000+ likes).",
        "x_query_hint": "lang:en filter:media min_faves:50000 -is:reply",
    },
    "verified": {
        "label": "Verified アカウントの伸び投稿",
        "guidance": "English posts from verified accounts with strong engagement (roughly 10,000+ likes).",
        "x_query_hint": "lang:en is:verified min_faves:10000 -is:reply",
    },
}

X_STATUS_RE = re.compile(
    r"https?://(?:www\.)?(?:x\.com|twitter\.com)/(?:[\w]+/)?status/(\d+)|(?:^|/)status/(\d+)",
    re.IGNORECASE,
)


def extract_status_id_from_url(url: str) -> Optional[str]:
    match = X_STATUS_RE.search(url)
    if not match:
        return None
    return match.group(1) or match.group(2)


@dataclass
class ViralPost:
    rank: int
    author_handle: str
    post_url: str
    post_text: str
    estimated_likes: Optional[int]
    estimated_reposts: Optional[int]
    topic: str
    why_viral: str
    posted_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("XAI_API_KEY")
    if not key:
        raise typer.BadParameter(
            "XAI_API_KEY が未設定です。環境変数または --api-key で指定してください。"
        )
    return key


def _default_date_range(days: int) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def _build_user_prompt(
    *,
    preset: str,
    topic: Optional[str],
    count: int,
    min_likes: int,
    from_date: str,
    to_date: str,
) -> str:
    preset_info = PRESETS[preset]
    topic_line = f"Additional focus: {topic}\n" if topic else ""
    return f"""Find up to {count} viral English posts on X that match this profile.

Category: {preset_info["label"]}
Search guidance: {preset_info["guidance"]}
X search operator hint (use as inspiration, not literal filter): {preset_info["x_query_hint"]}
{topic_line}Minimum engagement bar: prefer posts with at least ~{min_likes:,} likes (or clearly equivalent virality).
Date range: {from_date} through {to_date} (inclusive).
Requirements:
- English only
- Original posts preferred (not replies)
- Only include posts you can substantiate via x_search (real posts, not invented)
- Prioritize posts likely to have 1M+ impressions (use likes as proxy: 20k+ likes ≈ 1M+ views)

Return ONLY valid JSON (no markdown fences) with this schema:
{{
  "posts": [
    {{
      "rank": 1,
      "author_handle": "handle_without_at",
      "post_url": "https://x.com/.../status/...",
      "post_text": "verbatim or close excerpt of the post",
      "estimated_likes": 25000,
      "estimated_reposts": 1200,
      "topic": "short topic label",
      "why_viral": "1-2 sentences on why it spread",
      "posted_at": "YYYY-MM-DD or ISO8601, optional"
    }}
  ],
  "notes": "optional brief search notes"
}}

If fewer than {count} strong matches exist, return fewer items. Do not fabricate URLs or metrics."""


def _extract_output_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                parts.append(getattr(content, "text", "") or "")
    return "\n".join(parts).strip()


def _extract_citations(response: Any) -> list[str]:
    citations = getattr(response, "citations", None)
    if citations:
        return list(citations)
    urls: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            for ann in getattr(content, "annotations", []) or []:
                url = getattr(ann, "url", None)
                if url:
                    urls.append(url)
    return urls


def _parse_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_posts(raw: dict[str, Any], citations: list[str]) -> list[ViralPost]:
    posts_raw = raw.get("posts") or []
    citation_status_urls = [
        u for u in citations if extract_status_id_from_url(u)
    ]
    posts: list[ViralPost] = []

    for i, item in enumerate(posts_raw, start=1):
        if not isinstance(item, dict):
            continue
        url = (item.get("post_url") or "").strip()
        if not url and citation_status_urls:
            url = citation_status_urls[min(i - 1, len(citation_status_urls) - 1)]
        handle = (item.get("author_handle") or "unknown").lstrip("@")
        posts.append(
            ViralPost(
                rank=int(item.get("rank") or i),
                author_handle=handle,
                post_url=url,
                post_text=(item.get("post_text") or "").strip(),
                estimated_likes=_to_int(item.get("estimated_likes")),
                estimated_reposts=_to_int(item.get("estimated_reposts")),
                topic=(item.get("topic") or "").strip(),
                why_viral=(item.get("why_viral") or "").strip(),
                posted_at=item.get("posted_at"),
            )
        )
    return posts


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_csv(path: Path, posts: list[ViralPost]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ViralPost.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for post in posts:
            writer.writerow(post.to_dict())


def _print_table(posts: list[ViralPost]) -> None:
    table = Table(title="Viral English Posts (X)")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Handle")
    table.add_column("Likes", justify="right")
    table.add_column("Topic")
    table.add_column("URL", overflow="fold")
    for post in posts:
        likes = f"{post.estimated_likes:,}" if post.estimated_likes else "—"
        table.add_row(
            str(post.rank),
            f"@{post.author_handle}",
            likes,
            post.topic[:40] or "—",
            post.post_url or "—",
        )
    console.print(table)
    for post in posts:
        console.print(f"\n[bold]#{post.rank} @{post.author_handle}[/bold]")
        if post.post_text:
            console.print(post.post_text[:500] + ("…" if len(post.post_text) > 500 else ""))
        if post.why_viral:
            console.print(f"[dim]Why viral:[/dim] {post.why_viral}")


def fetch_viral_posts(
    *,
    api_key: str,
    preset: str = "general",
    topic: Optional[str] = None,
    count: int = 10,
    min_likes: int = 20_000,
    from_date: str,
    to_date: str,
    model: str = DEFAULT_MODEL,
    enable_image_understanding: bool = False,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=XAI_BASE_URL)
    user_prompt = _build_user_prompt(
        preset=preset,
        topic=topic,
        count=count,
        min_likes=min_likes,
        from_date=from_date,
        to_date=to_date,
    )
    tool: dict[str, Any] = {
        "type": "x_search",
        "from_date": from_date,
        "to_date": to_date,
    }
    if enable_image_understanding:
        tool["enable_image_understanding"] = True

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a social media research assistant. Use the x_search tool to find "
                    "real, high-engagement English posts on X. Be factual; cite real posts only."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        tools=[tool],
    )

    text = _extract_output_text(response)
    citations = _extract_citations(response)
    if not text:
        raise RuntimeError("モデルからテキスト応答を取得できませんでした。")

    parsed = _parse_json_payload(text)
    posts = _normalize_posts(parsed, citations)

    return {
        "preset": preset,
        "topic": topic,
        "from_date": from_date,
        "to_date": to_date,
        "model": model,
        "raw_response_text": text,
        "citations": citations,
        "notes": parsed.get("notes"),
        "posts": [p.to_dict() for p in posts],
    }


@app.command()
def fetch(
    preset: str = typer.Option(
        "general",
        "--preset",
        "-p",
        help=f"検索プリセット: {', '.join(PRESETS)}",
        case_sensitive=False,
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        help="追加キーワード（例: 'Claude', 'OpenAI'）",
    ),
    count: int = typer.Option(10, "--count", "-n", min=1, max=30),
    min_likes: int = typer.Option(
        20_000,
        "--min-likes",
        help="目安の最低いいね数（プロンプトに渡す）",
    ),
    days: int = typer.Option(7, "--days", help="直近 N 日を検索範囲にする"),
    from_date: Optional[str] = typer.Option(None, "--from-date", help="YYYY-MM-DD"),
    to_date: Optional[str] = typer.Option(None, "--to-date", help="YYYY-MM-DD"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="xAI モデル名"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="JSON 出力先（.csv を指定すると CSV も保存）",
    ),
    csv_output: Optional[Path] = typer.Option(
        None,
        "--csv",
        help="CSV 出力先（未指定時は -o が .csv なら同ファイル）",
    ),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="XAI_API_KEY"),
    images: bool = typer.Option(
        False,
        "--images",
        help="投稿内画像の理解を有効化（トークン消費増）",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="テーブル表示を省略"),
) -> None:
    """xAI X Search でバズった英語投稿を取得する。"""
    preset_key = preset.lower()
    if preset_key not in PRESETS:
        raise typer.BadParameter(
            f"未知の preset: {preset}。利用可能: {', '.join(PRESETS)}"
        )

    key = _require_api_key(api_key)
    if from_date and to_date:
        start, end = from_date, to_date
    elif from_date or to_date:
        raise typer.BadParameter("--from-date と --to-date は両方指定してください。")
    else:
        start, end = _default_date_range(days)

    console.print(
        f"[bold]Searching[/bold] preset={preset_key}  range={start}..{end}  count={count}"
    )
    result = fetch_viral_posts(
        api_key=key,
        preset=preset_key,
        topic=topic,
        count=count,
        min_likes=min_likes,
        from_date=start,
        to_date=end,
        model=model,
        enable_image_understanding=images,
    )

    posts = [ViralPost(**p) for p in result["posts"]]
    if not quiet:
        _print_table(posts)
        if result.get("notes"):
            console.print(f"\n[dim]Notes:[/dim] {result['notes']}")
        if result.get("citations"):
            console.print(f"\n[dim]Citations ({len(result['citations'])}):[/dim]")
            for url in result["citations"][:15]:
                console.print(f"  {url}")
            if len(result["citations"]) > 15:
                console.print(f"  … and {len(result['citations']) - 15} more")

    if output:
        _save_json(output, result)
        console.print(f"\n[green]Saved JSON →[/green] {output}")
        csv_path = csv_output
        if csv_path is None and output.suffix.lower() == ".csv":
            csv_path = output
        if csv_path:
            _save_csv(csv_path, posts)
            console.print(f"[green]Saved CSV →[/green] {csv_path}")
    elif csv_output:
        _save_csv(csv_output, posts)
        console.print(f"\n[green]Saved CSV →[/green] {csv_output}")


@app.command("list-presets")
def list_presets() -> None:
    """利用可能なプリセット一覧を表示する。"""
    for key, info in PRESETS.items():
        console.print(f"[bold]{key}[/bold]: {info['label']}")
        console.print(f"  hint: {info['x_query_hint']}\n")


if __name__ == "__main__":
    app()

"""PNGの一部をクロップし、ピクセルをSVGのrectに展開してベクター化するツール。

主な特徴
- ラスタ画像をSVGに埋め込まない（<image>非使用）
- 図・アイコン領域だけをクロップし、同色の横連続ピクセルを1本の<rect>にまとめて出力
- テキストは<text>で配置するため編集可能

使い方
    python -m image_to_svg.pixel_rect_svg --input input.png --output output.svg
    python -m image_to_svg.pixel_rect_svg --input input.png --output output.svg --config layout.yaml

configを省略すると本ファイル内のデフォルトレイアウト（サンプル画像向け）が適用されます。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from xml.sax.saxutils import escape

from PIL import Image

try:  # YAMLは任意（インストール済みなら使用）
    import yaml  # type: ignore
except Exception:  # pragma: no cover - 無ければJSONのみ扱う
    yaml = None


@dataclass(frozen=True)
class TextSpec:
    x: float
    y: float
    text: str
    size: int
    weight: int = 700
    anchor: str = "middle"
    baseline: str = "middle"
    fill: str = "#000"


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    font_family: str
    background: str | None
    defs: List[str]
    primitives: List[str]
    icon_bboxes: List[Tuple[str, Tuple[int, int, int, int]]]
    texts: List[TextSpec]


# サンプル画像向けデフォルトレイアウト -------------------------------------------------
def _default_texts() -> List[TextSpec]:
    return [
        TextSpec(688.0, 65.5, "業績予測とバリュエーション (2025-2027)", 48),
        TextSpec(446.5, 348.5, "支える", 32),
        TextSpec(924.0, 419.0, "展望", 32),
        TextSpec(207.0, 234.5, "高成長と収益性", 34),
        TextSpec(207.0, 473.0, "売上成長率 〜30%", 24),
        TextSpec(207.0, 510.5, "粗利率 80%超", 24),
        TextSpec(207.0, 547.0, "営業利益率 40%超", 24),
        TextSpec(207.0, 583.5, "FCF 〜10億ドル", 24),
        TextSpec(688.0, 234.5, "高い市場評価", 34),
        TextSpec(687.5, 472.5, "予想PER 高水準", 24),
        TextSpec(687.5, 510.0, "（例:186倍）", 24),
        TextSpec(687.5, 546.0, "時価総額 急上昇", 24),
        TextSpec(687.5, 583.5, "AIブームの追い風", 24),
        TextSpec(1167.5, 149.5, "強気シナリオ", 34),
        TextSpec(1168.5, 306.0, "成長持続 (AI・政府需要)", 24),
        TextSpec(1168.5, 343.0, "株価上昇余地", 24),
        TextSpec(1168.5, 378.5, "新契約への期待", 24),
        TextSpec(1167.5, 475.5, "保守的シナリオ", 34),
        TextSpec(1168.0, 631.5, "成長鈍化リスク", 24),
        TextSpec(1168.0, 667.0, "株価調整・PER圧縮", 24),
        TextSpec(1168.0, 704.0, "規制・支出削減の影響", 24),
    ]


def _default_layout(width: int, height: int) -> Layout:
    icon_bboxes = [
        ("left_icon", (80, 282, 330, 430)),
        ("mid_icon", (564, 278, 818, 420)),
        ("right_top_icon", (1070, 185, 1266, 275)),
        ("right_bottom_icon", (1040, 515, 1291, 594)),
    ]

    defs = [
        '<linearGradient id="boxGradGrey" x1="0" y1="0" x2="1" y2="1">\n'
        '    <stop offset="0%" stop-color="rgb(245,249,252)"/>\n'
        '    <stop offset="100%" stop-color="rgb(233,239,244)"/>\n'
        "  </linearGradient>",
        '<linearGradient id="boxGradBeige" x1="0" y1="0" x2="1" y2="1">\n'
        '    <stop offset="0%" stop-color="rgb(254,244,235)"/>\n'
        '    <stop offset="100%" stop-color="rgb(249,234,220)"/>\n'
        "  </linearGradient>",
        '<linearGradient id="midStroke" x1="0" y1="0" x2="1" y2="0">\n'
        '    <stop offset="0%" stop-color="#41689E"/>\n'
        '    <stop offset="100%" stop-color="#ED8440"/>\n'
        "  </linearGradient>",
    ]

    primitives = [
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        '<path d="M 374.0,398.5 L 466.0,402.5 L 468.0,381.5 L 514.5,418.0 L 468.0,455.5 L 467.0,436.5 L 374.0,437.5 Z" fill="#3C69AC"/>',
        '<path d="M 857.0,389.5 L 857.5,348.0 L 946.0,292.5 L 938.5,277.0 L 940.0,274.5 L 995.5,280.0 L 976.0,335.5 L 965.0,320.5 Z" fill="#3C69AC"/>',
        '<path d="M 860.0,450.5 L 965.0,517.5 L 977.0,503.5 L 996.5,560.0 L 940.0,564.5 L 937.5,563.0 L 946.0,547.5 L 860.0,492.5 Z" fill="#EB863E"/>',
        '<rect x="40" y="190" width="334" height="456" rx="18" ry="18" fill="url(#boxGradGrey)" stroke="#41689E" stroke-width="6" stroke-linejoin="round"/>',
        '<rect x="521" y="190" width="334" height="456" rx="18" ry="18" fill="url(#boxGradGrey)" stroke="url(#midStroke)" stroke-width="6" stroke-linejoin="round"/>',
        '<rect x="1002" y="109" width="334" height="303" rx="18" ry="18" fill="url(#boxGradGrey)" stroke="#41689E" stroke-width="6" stroke-linejoin="round"/>',
        '<rect x="1002" y="435" width="334" height="302" rx="18" ry="18" fill="url(#boxGradBeige)" stroke="#ED8440" stroke-width="6" stroke-linejoin="round"/>',
    ]

    return Layout(
        width=width,
        height=height,
        font_family=(
            "Noto Sans CJK JP, Noto Sans JP, Hiragino Kaku Gothic ProN, "
            "Meiryo, sans-serif"
        ),
        background=None,
        defs=defs,
        primitives=primitives,
        icon_bboxes=icon_bboxes,
        texts=_default_texts(),
    )


# 基本処理 ------------------------------------------------------------------------------
def rgb_to_css(rgb: Sequence[int]) -> str:
    r, g, b = rgb[:3]
    return f"rgb({r},{g},{b})"


def iter_horizontal_runs(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Iterable[Tuple[int, int, int, int, Tuple[int, int, int]]]:
    """
    bbox=(x0, y0, x1, y1) 内を走査し、同色の横連続区間を1本のrunとして返す。
    戻り値: (x, y, w, h, (r,g,b))
    """

    x0, y0, x1, y1 = bbox
    pix = img.load()

    for y in range(y0, y1):
        run_x = x0
        run_color = pix[x0, y]
        run_len = 1

        for x in range(x0 + 1, x1):
            c = pix[x, y]
            if c == run_color:
                run_len += 1
            else:
                yield run_x, y, run_len, 1, run_color
                run_x = x
                run_color = c
                run_len = 1

        yield run_x, y, run_len, 1, run_color


def emit_pixel_rects_as_svg(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
    merge_runs: bool = True,
) -> Iterable[str]:
    """クロップ領域をrect列に変換したSVG断片を返す。"""

    if merge_runs:
        iterator = iter_horizontal_runs(img, bbox)
    else:
        x0, y0, x1, y1 = bbox
        iterator = (
            (x, y, 1, 1, img.load()[x, y])
            for y in range(y0, y1)
            for x in range(x0, x1)
        )

    for x, y, w, h, col in iterator:
        yield f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{rgb_to_css(col)}"/>'


def _load_config(path: Path) -> dict:
    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text())
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAMLが見つかりません。'pip install pyyaml' を実行してください。")
        return yaml.safe_load(path.read_text())
    raise ValueError("サポートされていない設定ファイル形式です (json / yaml のみ)")


def _layout_from_dict(data: dict, width: int, height: int) -> Layout:
    viewbox = data.get("viewbox", {}) or {}
    layout_width = int(viewbox.get("width", width))
    layout_height = int(viewbox.get("height", height))

    font_family = data.get("font_family") or _default_layout(width, height).font_family
    background = data.get("background")
    defs = list(data.get("defs", []))
    primitives = list(data.get("primitives", []))

    icon_bboxes: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for entry in data.get("icon_bboxes", []):
        name = entry.get("name", "icon")
        bbox = entry.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"icon_bboxesの形式が不正です: {entry}")
        icon_bboxes.append((name, tuple(int(v) for v in bbox)))

    texts: List[TextSpec] = []
    for t in data.get("texts", []):
        texts.append(
            TextSpec(
                x=float(t["x"]),
                y=float(t["y"]),
                text=str(t["text"]),
                size=int(t.get("size", 16)),
                weight=int(t.get("weight", 700)),
                anchor=t.get("anchor", "middle"),
                baseline=t.get("baseline", "middle"),
                fill=t.get("fill", "#000"),
            )
        )

    return Layout(
        width=layout_width,
        height=layout_height,
        font_family=font_family,
        background=background,
        defs=defs,
        primitives=primitives,
        icon_bboxes=icon_bboxes,
        texts=texts,
    )


def load_layout(config_path: str | None, image_size: Tuple[int, int]) -> Layout:
    if config_path is None:
        return _default_layout(*image_size)

    path = Path(config_path)
    data = _load_config(path)
    return _layout_from_dict(data, *image_size)


def build_svg(
    img: Image.Image,
    layout: Layout,
    merge_runs: bool = True,
) -> str:
    w = layout.width or img.width
    h = layout.height or img.height

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')

    if layout.defs:
        parts.append("<defs>")
        parts.extend(layout.defs)
        parts.append("</defs>")

    if layout.background:
        parts.append(layout.background)

    parts.extend(layout.primitives)

    for name, bbox in layout.icon_bboxes:
        parts.append(f"<!-- {name}: bbox={bbox} -->")
        parts.extend(emit_pixel_rects_as_svg(img, bbox, merge_runs=merge_runs))

    for t in layout.texts:
        parts.append(
            f'<text x="{t.x}" y="{t.y}" '
            f'text-anchor="{t.anchor}" dominant-baseline="{t.baseline}" '
            f'font-family="{escape(layout.font_family)}" font-weight="{t.weight}" '
            f'font-size="{t.size}" fill="{t.fill}">{escape(t.text)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def save_svg_from_png(
    input_path: str,
    output_path: str,
    config_path: str | None = None,
    merge_runs: bool = True,
) -> None:
    img = Image.open(input_path).convert("RGB")
    layout = load_layout(config_path, img.size)

    # bbox範囲チェック
    w, h = img.size
    for name, (x0, y0, x1, y1) in layout.icon_bboxes:
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"bboxが画像サイズを超えています: {name}={x0, y0, x1, y1}, image={w}x{h}")

    svg_content = build_svg(img, layout, merge_runs=merge_runs)
    Path(output_path).write_text(svg_content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="クロップ領域をrect化し、テキストを<text>で保持するSVGを生成"
    )
    parser.add_argument("--input", required=True, help="入力PNGパス")
    parser.add_argument("--output", required=True, help="出力SVGパス")
    parser.add_argument("--config", help="レイアウト定義 (YAML/JSON)。未指定ならデフォルトレイアウト")
    parser.add_argument(
        "--no-merge-runs",
        action="store_true",
        help="横方向の同色run結合を無効化（1pxごとのrectを出力、デバッグ用途）",
    )

    args = parser.parse_args()

    save_svg_from_png(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        merge_runs=not args.no_merge_runs,
    )


if __name__ == "__main__":
    main()

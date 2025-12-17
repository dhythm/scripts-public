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


# シンプルなデフォルトレイアウト（全体をcrop_pixelsで貼るだけ）
def _full_image_layout(width: int, height: int) -> Layout:
    return Layout(
        width=width,
        height=height,
        font_family="Noto Sans CJK JP, Noto Sans JP, Hiragino Kaku Gothic ProN, Meiryo, sans-serif",
        background=None,
        defs=[],
        primitives=['<rect x="0" y="0" width="100%" height="100%" fill="white"/>'],
        icon_bboxes=[("full_image", (0, 0, width, height))],
        texts=[],
    )


# 基本処理 ------------------------------------------------------------------------------
def rgb_to_css(rgb: Sequence[int]) -> str:
    r, g, b = rgb[:3]
    return f"rgb({r},{g},{b})"


def merge_vertical_runs(
    runs: List[Tuple[int, int, int, int, Tuple[int, int, int]]]
) -> List[Tuple[int, int, int, int, Tuple[int, int, int]]]:
    """
    横runを縦方向にマージして大きな矩形を生成

    同じx座標・幅・色を持つ横runで、y座標が連続するものを1つの矩形にまとめる。

    Args:
        runs: [(x, y, w, h, (r,g,b)), ...]  # h=1の横run

    Returns:
        [(x, y, w, h, (r,g,b)), ...]  # マージ後の矩形
    """
    if not runs:
        return []

    # (x, w, color) をキーにしてrunをグループ化
    from collections import defaultdict

    groups: dict[Tuple[int, int, Tuple[int, int, int]], List[int]] = defaultdict(list)
    for x, y, w, h, color in runs:
        key = (x, w, color)
        groups[key].append(y)

    result: List[Tuple[int, int, int, int, Tuple[int, int, int]]] = []

    for (x, w, color), y_list in groups.items():
        # yでソート
        y_list.sort()

        # 連続するyをマージ
        start_y = y_list[0]
        end_y = y_list[0]

        for y in y_list[1:]:
            if y == end_y + 1:
                # 連続している
                end_y = y
            else:
                # 連続が途切れた → 矩形を出力
                result.append((x, start_y, w, end_y - start_y + 1, color))
                start_y = y
                end_y = y

        # 最後のグループを出力
        result.append((x, start_y, w, end_y - start_y + 1, color))

    return result


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
    merge_vertical: bool = True,
) -> Iterable[str]:
    """クロップ領域をrect列に変換したSVG断片を返す。

    Args:
        img: PIL Image
        bbox: (x0, y0, x1, y1)
        merge_runs: 横方向の同色runをマージするか
        merge_vertical: 縦方向の隣接矩形もマージするか（merge_runs=Trueの場合のみ有効）
    """
    if merge_runs:
        runs = list(iter_horizontal_runs(img, bbox))
        if merge_vertical:
            runs = merge_vertical_runs(runs)
        iterator = iter(runs)
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
        # レイアウト未指定時は、画像全体を1枚のcrop_pixelsとして扱う最小構成
        return _full_image_layout(*image_size)

    path = Path(config_path)
    data = _load_config(path)
    return _layout_from_dict(data, *image_size)


def build_svg(
    img: Image.Image,
    layout: Layout,
    merge_runs: bool = True,
    merge_vertical: bool = True,
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
        parts.extend(emit_pixel_rects_as_svg(img, bbox, merge_runs=merge_runs, merge_vertical=merge_vertical))

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
    merge_vertical: bool = True,
) -> None:
    img = Image.open(input_path).convert("RGB")
    layout = load_layout(config_path, img.size)

    # bbox範囲チェック
    w, h = img.size
    for name, (x0, y0, x1, y1) in layout.icon_bboxes:
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"bboxが画像サイズを超えています: {name}={x0, y0, x1, y1}, image={w}x{h}")

    svg_content = build_svg(img, layout, merge_runs=merge_runs, merge_vertical=merge_vertical)
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

"""PNGからパーツを切り出し、SVGに合成する簡易パイプライン。

使い方:
  # 1) アイコン/ロゴをPNGで切り出し
  python -m image_to_svg.pipeline_svg extract-assets --layout layout.json --input input.png

  # 2) GUIでPNGをトレースして SVG を用意（assets/*.svg に差し替え）

  # 3) 最終SVGを合成（テキストや差し替えはpatchで上書き可）
  python -m image_to_svg.pipeline_svg compose --layout layout.json --input input.png --output build/output.svg --patch patch.json

前提: layout.json の各要素に id を付けると、patchやGUIでの編集が楽になります。
依存: Pillow (uv sync 済みなら入っています)。layout.json が無い場合は画像全体をrect化するフォールバックで動作します。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

from PIL import Image
from .pixel_rect_svg import save_svg_from_png


# ---------------------------------------------------------------------------#
# 基本ユーティリティ
# ---------------------------------------------------------------------------#
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_png_crop(
    input_png: str,
    out_png: str,
    bbox: List[int],
    remove_bg: str = "none",
    tolerance: int = 18,
) -> None:
    img = Image.open(input_png).convert("RGBA")
    x0, y0, x1, y1 = map(int, bbox)
    crop = img.crop((x0, y0, x1, y1))

    if remove_bg == "auto":
        crop = remove_background_auto(crop, tolerance=tolerance)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_png)


def remove_background_auto(img_rgba: Image.Image, tolerance: int = 18) -> Image.Image:
    """四隅の平均色を背景とみなし、近い色を透明化。"""
    w, h = img_rgba.size
    px = img_rgba.load()
    corners = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
    bg = tuple(sorted(c)[1] for c in zip(*corners))  # R/G/B の中央値

    out = img_rgba.copy()
    op = out.load()
    tol2 = tolerance * tolerance
    for y in range(h):
        for x in range(w):
            r, g, b, a = op[x, y]
            dr, dg, db = r - bg[0], g - bg[1], b - bg[2]
            if (dr * dr + dg * dg + db * db) <= tol2:
                op[x, y] = (r, g, b, 0)
    return out


def apply_patch(layout: Dict[str, Any], patch: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not patch:
        return layout
    texts = patch.get("text", {})
    assets = patch.get("assets", {})
    for el in layout.get("elements", []):
        el_id = el.get("id")
        if not el_id:
            continue
        if el.get("type") == "text" and el_id in texts:
            el["text"] = texts[el_id]
        if el.get("type") == "asset" and el_id in assets:
            el["src_svg"] = assets[el_id]
    return layout


# ---------------------------------------------------------------------------#
# SVGアセット読み込み・配置
# ---------------------------------------------------------------------------#
def _parse_length(s: Optional[str]) -> float:
    if not s:
        return 0.0
    m = re.match(r"^\s*([0-9.]+)", s)
    return float(m.group(1)) if m else 0.0


def load_svg_inner(svg_path: str) -> Tuple[str, Tuple[float, float, float, float]]:
    """SVGの子要素XMLと viewBox(minx,miny,w,h) を返す。"""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    vb_attr = root.get("viewBox")
    if vb_attr:
        vals = [v for v in re.split(r"[ ,]+", vb_attr.strip()) if v]
        minx, miny, vbw, vbh = map(float, vals)
    else:
        vbw = _parse_length(root.get("width"))
        vbh = _parse_length(root.get("height"))
        minx = miny = 0.0

    inner_xml = "".join(ET.tostring(child, encoding="unicode") for child in list(root))
    return inner_xml, (minx, miny, vbw, vbh)


def emit_asset_group(
    el_id: str,
    src_svg: str,
    dest_bbox: List[float],
    preserve_aspect: str = "meet",
) -> str:
    inner, (minx, miny, vbw, vbh) = load_svg_inner(src_svg)
    x0, y0, x1, y1 = map(float, dest_bbox)
    dw, dh = (x1 - x0), (y1 - y0)

    if vbw <= 0 or vbh <= 0:
        return f'<g id="{escape(el_id)}"><title>{escape(el_id)}</title>{inner}</g>'

    sx = dw / vbw
    sy = dh / vbh
    s = min(sx, sy) if preserve_aspect == "meet" else max(sx, sy)
    tx = x0 + (dw - vbw * s) / 2.0 - minx * s
    ty = y0 + (dh - vbh * s) / 2.0 - miny * s

    return (
        f'<g id="{escape(el_id)}"><title>{escape(el_id)}</title>'
        f'<g transform="translate({tx:g},{ty:g}) scale({s:g})">{inner}</g>'
        f"</g>"
    )


# ---------------------------------------------------------------------------#
# 要素生成
# ---------------------------------------------------------------------------#
def emit_text(el: Dict[str, Any]) -> str:
    x = float(el["x"])
    y = float(el["y"])
    text = str(el.get("text", ""))
    ff = str(el.get("font_family", "sans-serif"))
    fs = int(el.get("font_size", 24))
    fw = int(el.get("font_weight", 700))
    fill = str(el.get("fill", "#000"))
    anchor = str(el.get("anchor", "middle"))
    baseline = str(el.get("baseline", "middle"))
    lh = float(el.get("line_height", 1.2))

    lines = text.split("\n")
    if len(lines) == 1:
        return (
            f'<text id="{escape(el.get("id",""))}" x="{x:g}" y="{y:g}" '
            f'text-anchor="{escape(anchor)}" dominant-baseline="{escape(baseline)}" '
            f'font-family="{escape(ff)}" font-size="{fs}" font-weight="{fw}" fill="{escape(fill)}">'
            f"{escape(text)}</text>"
        )

    mid = (len(lines) - 1) / 2.0
    y0 = y - (mid * lh * fs)
    tspans = [f'<tspan x="{x:g}" y="{y0:g}">{escape(lines[0])}</tspan>']
    for ln in lines[1:]:
        tspans.append(f'<tspan x="{x:g}" dy="{lh:.3f}em">{escape(ln)}</tspan>')

    return (
        f'<text id="{escape(el.get("id",""))}" text-anchor="{escape(anchor)}" '
        f'dominant-baseline="{escape(baseline)}" font-family="{escape(ff)}" '
        f'font-size="{fs}" font-weight="{fw}" fill="{escape(fill)}">\n'
        + "\n".join(tspans)
        + "\n</text>"
    )


# ---------------------------------------------------------------------------#
# 合成
# ---------------------------------------------------------------------------#
def compose_svg(
    layout_path: str,
    input_png: str,
    output_svg: str,
    patch_path: Optional[str],
) -> None:
    # layout が無い場合のフォールバック: 画像全体をrect化してSVG化
    if not layout_path or not Path(layout_path).exists():
        Path(output_svg).parent.mkdir(parents=True, exist_ok=True)
        save_svg_from_png(input_png, output_svg, config_path=None, merge_runs=True)
        return

    layout = load_json(layout_path)
    patch = load_json(patch_path) if patch_path else None
    layout = apply_patch(layout, patch)

    base_w, base_h = Image.open(input_png).size
    canvas = layout.get("canvas", {})
    W = int(canvas.get("width", 0) or base_w)
    H = int(canvas.get("height", 0) or base_h)
    bg = str(canvas.get("background", "white"))

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
    )
    parts.append(f'<rect x="0" y="0" width="100%" height="100%" fill="{escape(bg)}"/>')

    for el in layout.get("elements", []):
        t = el.get("type")
        el_id = el.get("id", "")

        if t == "rect":
            x, y = float(el["x"]), float(el["y"])
            w, h = float(el["w"]), float(el["h"])
            rx = float(el.get("rx", 0))
            ry = float(el.get("ry", rx))
            fill = str(el.get("fill", "none"))
            stroke = str(el.get("stroke", "none"))
            sw = float(el.get("stroke_width", 1))
            parts.append(
                f'<g id="{escape(el_id)}"><title>{escape(el_id)}</title>'
                f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
                f'rx="{rx:g}" ry="{ry:g}" fill="{escape(fill)}" '
                f'stroke="{escape(stroke)}" stroke-width="{sw:g}"/></g>'
            )

        elif t == "path":
            d = str(el["d"])
            fill = str(el.get("fill", "none"))
            stroke = str(el.get("stroke", "none"))
            sw = float(el.get("stroke_width", 1))
            parts.append(
                f'<g id="{escape(el_id)}"><title>{escape(el_id)}</title>'
                f'<path d="{escape(d)}" fill="{escape(fill)}" '
                f'stroke="{escape(stroke)}" stroke-width="{sw:g}"/></g>'
            )

        elif t == "text":
            parts.append(emit_text(el))

        elif t == "asset":
            dest = el.get("dest_bbox")
            src_svg = el.get("src_svg")
            if dest and src_svg and Path(src_svg).exists():
                parts.append(
                    emit_asset_group(
                        el_id,
                        src_svg,
                        dest,
                        preserve_aspect=el.get("preserve_aspect", "meet"),
                    )
                )
            else:
                # SVGが無い場合は枠のみ（目印）
                if dest:
                    x0, y0, x1, y1 = map(float, dest)
                    parts.append(
                        f'<g id="{escape(el_id)}"><title>{escape(el_id)}</title>'
                        f'<rect x="{x0:g}" y="{y0:g}" width="{(x1-x0):g}" height="{(y1-y0):g}" '
                        f'fill="none" stroke="magenta" stroke-width="1"/></g>'
                    )

    parts.append("</svg>")
    Path(output_svg).parent.mkdir(parents=True, exist_ok=True)
    Path(output_svg).write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------#
# アセット切り出し
# ---------------------------------------------------------------------------#
def extract_assets(layout_path: str, input_png: str) -> None:
    layout = load_json(layout_path)
    for el in layout.get("elements", []):
        if el.get("type") != "asset":
            continue
        crop = el.get("png_crop")
        out_png = el.get("src_png_out")
        if not crop or not out_png:
            continue
        bbox = crop.get("bbox")
        if not bbox:
            continue
        remove_bg = crop.get("remove_bg", "none")
        tol = int(crop.get("tolerance", 18))
        save_png_crop(input_png, out_png, bbox, remove_bg=remove_bg, tolerance=tol)
        print(f"[OK] extracted: {out_png}")


# ---------------------------------------------------------------------------#
# CLI
# ---------------------------------------------------------------------------#
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_e = sub.add_parser("extract-assets", help="layout.jsonに従ってPNGアセットを切り出す")
    ap_e.add_argument("--layout", required=True)
    ap_e.add_argument("--input", required=True)

    ap_c = sub.add_parser("compose", help="layout.json(+patch)からSVGを合成")
    ap_c.add_argument("--layout", required=False, help="省略時は画像全体をrect化するフォールバック")
    ap_c.add_argument("--input", required=True)
    ap_c.add_argument("--output", required=True)
    ap_c.add_argument("--patch", default=None)

    args = ap.parse_args()
    if args.cmd == "extract-assets":
        extract_assets(args.layout, args.input)
    elif args.cmd == "compose":
        compose_svg(args.layout, args.input, args.output, args.patch)


if __name__ == "__main__":
    main()

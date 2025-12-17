"""汎用SVG化ツール: 矩形アノテーション -> JSON -> SVG生成

コマンド:
    python -m image_to_svg.svgify annotate --input input.png --layout layout.json
    python -m image_to_svg.svgify render   --input input.png --layout layout.json --output output.svg

特徴:
- ラスタ埋め込みなし。crop_pixels要素はピクセルを<rect>群に展開して元画像と完全一致。
- text要素は編集可能な<text>で出力（複数行も対応）。
- rect/path要素で背景や枠を軽量ベクター化可能。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape

from PIL import Image


# ---------------- SVG生成: crop_pixels をrect群に変換 ---------------- #

def _rgb_css(r: int, g: int, b: int) -> str:
    return f"rgb({r},{g},{b})"


def iter_horizontal_runs_rgba(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
    skip_transparent: bool = True,
) -> Iterable[Tuple[int, int, int, int, Tuple[int, int, int, int]]]:
    """
    bbox=(x0,y0,x1,y1) を走査し、同一RGBAの横連続runを返す。
    戻り値: (x, y, w, h, (r,g,b,a))
    """
    x0, y0, x1, y1 = bbox
    px = img.load()

    for y in range(y0, y1):
        run_x = x0
        run_col = px[x0, y]
        run_len = 1

        for x in range(x0 + 1, x1):
            c = px[x, y]
            if c == run_col:
                run_len += 1
            else:
                if not (skip_transparent and run_col[3] == 0):
                    yield run_x, y, run_len, 1, run_col
                run_x = x
                run_col = c
                run_len = 1

        if not (skip_transparent and run_col[3] == 0):
            yield run_x, y, run_len, 1, run_col


def emit_crop_pixels_svg(
    img_rgba: Image.Image,
    bbox: Tuple[int, int, int, int],
    dest_xy: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
    skip_transparent: bool = True,
) -> Iterable[str]:
    """
    クロップ領域をピクセル→rectで出力（ラスタ埋め込みなし）。
    dest_xyを指定すると配置先左上座標を上書き。scaleで拡大縮小。
    """
    x0, y0, x1, y1 = bbox
    if dest_xy is None:
        dx, dy = float(x0), float(y0)
    else:
        dx, dy = dest_xy

    for x, y, w, h, (r, g, b, a) in iter_horizontal_runs_rgba(
        img_rgba, bbox, skip_transparent=skip_transparent
    ):
        sx = dx + (x - x0) * scale
        sy = dy + (y - y0) * scale
        sw = w * scale
        sh = h * scale

        if a == 255:
            yield (
                f'<rect x="{sx:g}" y="{sy:g}" width="{sw:g}" height="{sh:g}" '
                f'fill="{_rgb_css(r, g, b)}"/>'
            )
        else:
            opacity = a / 255.0
            yield (
                f'<rect x="{sx:g}" y="{sy:g}" width="{sw:g}" height="{sh:g}" '
                f'fill="{_rgb_css(r, g, b)}" fill-opacity="{opacity:.6f}"/>'
            )


def svg_text_element(el: Dict[str, Any]) -> str:
    """
    type:"text"要素を<text>として出力。改行(\n)はtspan複数行。
    """
    x = float(el["x"])
    y = float(el["y"])
    text = str(el.get("text", ""))

    font_size = int(el.get("font_size", 24))
    font_family = str(el.get("font_family", "sans-serif"))
    font_weight = int(el.get("font_weight", 700))
    fill = str(el.get("fill", "#000"))
    anchor = str(el.get("anchor", "middle"))
    baseline = str(el.get("baseline", "middle"))
    line_h = float(el.get("line_height", 1.2))  # em

    lines = text.split("\n")
    if len(lines) == 1:
        return (
            f'<text x="{x:g}" y="{y:g}" text-anchor="{escape(anchor)}" '
            f'dominant-baseline="{escape(baseline)}" '
            f'font-family="{escape(font_family)}" font-weight="{font_weight}" '
            f'font-size="{font_size}" fill="{escape(fill)}">{escape(text)}</text>'
        )

    mid = (len(lines) - 1) / 2.0
    y0 = y - (mid * line_h * font_size)
    tspans = [f'<tspan x="{x:g}" y="{y0:g}">{escape(lines[0])}</tspan>']
    for ln in lines[1:]:
        tspans.append(f'<tspan x="{x:g}" dy="{line_h:.3f}em">{escape(ln)}</tspan>')

    return (
        f'<text text-anchor="{escape(anchor)}" dominant-baseline="{escape(baseline)}" '
        f'font-family="{escape(font_family)}" font-weight="{font_weight}" '
        f'font-size="{font_size}" fill="{escape(fill)}">\n'
        + "\n".join(tspans)
        + "\n</text>"
    )


def render_svg(input_path: str, layout_path: str, output_path: str) -> None:
    layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))

    img = Image.open(input_path).convert("RGBA")
    img_w, img_h = img.size

    canvas = layout.get("canvas", {})
    W = int(canvas.get("width", img_w))
    H = int(canvas.get("height", img_h))
    background = canvas.get("background", "white")

    defs_raw: List[str] = layout.get("defs_raw", []) or []
    elements: List[Dict[str, Any]] = layout.get("elements", []) or []

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}">\n'
        )

        if defs_raw:
            out.write("<defs>\n")
            for d in defs_raw:
                out.write(d.rstrip() + "\n")
            out.write("</defs>\n")

        out.write(
            f'<rect x="0" y="0" width="100%" height="100%" fill="{escape(background)}"/>\n'
        )

        for el in elements:
            t = el.get("type")

            if t == "rect":
                x = float(el["x"]); y = float(el["y"])
                w = float(el["w"]); h = float(el["h"])
                rx = float(el.get("rx", 0)); ry = float(el.get("ry", rx))
                fill = str(el.get("fill", "none"))
                stroke = str(el.get("stroke", "none"))
                sw = float(el.get("stroke_width", 1))
                out.write(
                    f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
                    f'rx="{rx:g}" ry="{ry:g}" fill="{escape(fill)}" '
                    f'stroke="{escape(stroke)}" stroke-width="{sw:g}"/>\n'
                )

            elif t == "path":
                d = str(el["d"])
                fill = str(el.get("fill", "none"))
                stroke = str(el.get("stroke", "none"))
                sw = float(el.get("stroke_width", 1))
                out.write(
                    f'<path d="{escape(d)}" fill="{escape(fill)}" '
                    f'stroke="{escape(stroke)}" stroke-width="{sw:g}"/>\n'
                )

            elif t == "text":
                out.write(svg_text_element(el) + "\n")

            elif t == "crop_pixels":
                bbox = el["bbox"]
                x0, y0, x1, y1 = map(int, bbox)
                scale = float(el.get("scale", 1.0))
                dest = el.get("dest_xy")
                dest_xy = None if dest is None else (float(dest[0]), float(dest[1]))
                skip_transparent = bool(el.get("skip_transparent", True))

                for frag in emit_crop_pixels_svg(
                    img_rgba=img,
                    bbox=(x0, y0, x1, y1),
                    dest_xy=dest_xy,
                    scale=scale,
                    skip_transparent=skip_transparent,
                ):
                    out.write(frag + "\n")

            else:
                # 未対応タイプはスキップ
                continue

        out.write("</svg>\n")


# ---------------- アノテーション: 画像に矩形を描きlayout.jsonを生成 ---------------- #

@dataclass
class _AnnState:
    mode: str = "crop"  # crop | text | rect
    elements: List[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.elements is None:
            self.elements = []


def annotate_layout(input_path: str, layout_path: str) -> None:
    # matplotlibはrenderには不要なのでここで遅延import
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.widgets import RectangleSelector

    img = Image.open(input_path).convert("RGBA")
    W, H = img.size

    state = _AnnState()
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(
        "Drag: rectangle / Keys: [t]=text [c]=crop [r]=rect [u]=undo [s]=save [q]=quit",
        fontsize=10,
    )
    ax.set_axis_off()

    drawn_patches: List[patches.Rectangle] = []

    def _mode_color(m: str) -> str:
        if m == "text":
            return "deepskyblue"
        if m == "rect":
            return "orange"
        return "limegreen"  # crop

    def onselect(eclick, erelease) -> None:
        x0, y0 = int(min(eclick.xdata, erelease.xdata)), int(
            min(eclick.ydata, erelease.ydata)
        )
        x1, y1 = int(max(eclick.xdata, erelease.xdata)), int(
            max(eclick.ydata, erelease.ydata)
        )

        if x1 <= x0 or y1 <= y0:
            return

        mode = state.mode
        if mode == "crop":
            el = {
                "type": "crop_pixels",
                "id": f"crop_{len(state.elements)+1}",
                "bbox": [x0, y0, x1, y1],
                "dest_xy": [x0, y0],
                "scale": 1.0,
                "skip_transparent": True,
            }
        elif mode == "text":
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            est_size = max(10, int((y1 - y0) * 0.65))
            el = {
                "type": "text",
                "id": f"text_{len(state.elements)+1}",
                "x": cx,
                "y": cy,
                "text": "TODO",
                "font_size": est_size,
                "font_weight": 700,
                "font_family": "Noto Sans CJK JP, Noto Sans JP, Hiragino Kaku Gothic ProN, Meiryo, sans-serif",
                "fill": "#000000",
                "anchor": "middle",
                "baseline": "middle",
            }
        else:  # rect
            el = {
                "type": "rect",
                "id": f"rect_{len(state.elements)+1}",
                "x": x0,
                "y": y0,
                "w": (x1 - x0),
                "h": (y1 - y0),
                "rx": 0,
                "ry": 0,
                "fill": "none",
                "stroke": "#000000",
                "stroke_width": 1,
            }

        state.elements.append(el)

        p = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor=_mode_color(mode),
            facecolor="none",
        )
        ax.add_patch(p)
        drawn_patches.append(p)
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key == "t":
            state.mode = "text"
            print("Mode -> text")
        elif event.key == "c":
            state.mode = "crop"
            print("Mode -> crop")
        elif event.key == "r":
            state.mode = "rect"
            print("Mode -> rect")
        elif event.key == "u":
            if state.elements:
                state.elements.pop()
            if drawn_patches:
                drawn_patches.pop().remove()
            fig.canvas.draw_idle()
            print("Undo")
        elif event.key == "s":
            layout = {
                "version": 1,
                "canvas": {"width": W, "height": H, "background": "white"},
                "defs_raw": [],
                "elements": state.elements,
            }
            Path(layout_path).write_text(
                json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Saved: {layout_path}")
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=False,
    )

    plt.show()


# ---------------- CLI ---------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_a = sub.add_parser("annotate", help="画像に矩形を引いてlayout.jsonを作成")
    ap_a.add_argument("--input", required=True, help="入力PNG")
    ap_a.add_argument("--layout", required=True, help="出力layout.json")

    ap_r = sub.add_parser("render", help="layout.jsonからSVG生成（ラスタ非埋め込み）")
    ap_r.add_argument("--input", required=True, help="入力PNG")
    ap_r.add_argument("--layout", required=True, help="layout.json")
    ap_r.add_argument("--output", required=True, help="出力SVG")

    args = ap.parse_args()

    if args.cmd == "annotate":
        annotate_layout(args.input, args.layout)
    elif args.cmd == "render":
        render_svg(args.input, args.layout, args.output)


if __name__ == "__main__":
    main()

"""layout.json なしで PNG/JPG を自動トレースし、編集しやすいSVGを生成するスクリプト。

特徴:
- 画像を減色 (k-means) して色ぶれを抑え、連結成分ごとにグループ化した <path> を出力
- 背景は最頻色で自動推定
- GUI編集を想定し、各成分に id/title を付与
- オプションで中間情報を JSON ダンプ

使い方:
  uv run python -m image_to_svg.auto_vectorize input.png output.svg \
    --colors 32 --scale 2.0 --denoise nlmeans \
    --min-area 12 --epsilon 0.002 --morph-close 0 \
    --dump output.auto.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# データ構造
# -----------------------------------------------------------------------------
@dataclass
class PaletteEntry:
    idx: int
    rgb: Tuple[int, int, int]
    count: int


# -----------------------------------------------------------------------------
# 前処理
# -----------------------------------------------------------------------------
def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def preprocess(img_bgr: np.ndarray, scale: float, denoise: str) -> np.ndarray:
    img = img_bgr
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    if denoise == "none":
        return img
    if denoise == "bilateral":
        return cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=7)
    if denoise == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(img, None, h=8, hColor=8, templateWindowSize=7, searchWindowSize=21)
    raise ValueError(f"Unknown denoise mode: {denoise}")


def quantize_kmeans(img_bgr: np.ndarray, k: int, attempts: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    Z = img_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, label, center = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(center, 0, 255).astype(np.uint8)
    labels = label.reshape((h, w)).astype(np.int32)
    counts = np.bincount(labels.reshape(-1), minlength=k).astype(np.int64)
    return labels, centers, counts


def pick_background_label(counts: np.ndarray) -> int:
    return int(np.argmax(counts))  # 最頻色を背景とみなす


# -----------------------------------------------------------------------------
# マスク処理・輪郭抽出
# -----------------------------------------------------------------------------
def remove_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask_u8
    num, cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            out[cc == i] = 255
    return out


def mask_to_svg_path(mask_u8: np.ndarray, offset_xy: Tuple[int, int], epsilon_ratio: float, min_area: int) -> str:
    mask = remove_small_components(mask_u8, min_area=min_area)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return ""

    ox, oy = offset_xy
    d_parts: List[str] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(0.5, epsilon_ratio * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx[:, 0, :].astype(np.float32)
        if pts.shape[0] < 3:
            continue
        pts[:, 0] += ox
        pts[:, 1] += oy
        coords = " ".join(f"{p[0]:g},{p[1]:g}" for p in pts)
        first, *rest = coords.split(" ")
        if rest:
            d_parts.append(f"M {first} L {' '.join(rest)} Z")
        else:
            d_parts.append(f"M {first} Z")
    return " ".join(d_parts)


# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
def auto_vectorize(
    input_path: str,
    output_svg: str,
    dump_json: Optional[str],
    colors: int,
    scale: float,
    denoise: str,
    min_area: int,
    epsilon: float,
    morph_close: int,
) -> None:
    img0 = read_image_bgr(input_path)
    h0, w0 = img0.shape[:2]

    img = preprocess(img0, scale=scale, denoise=denoise)
    h, w = img.shape[:2]

    labels, centers_bgr, counts = quantize_kmeans(img, k=colors, attempts=3)
    bg_label = pick_background_label(counts)
    bg_bgr = centers_bgr[bg_label]
    bg_rgb = (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))

    # 前景マスク
    fg = (labels != bg_label).astype(np.uint8) * 255
    num_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    # dump（任意）
    if dump_json:
        palette = []
        for i in range(colors):
            b, g, r = map(int, centers_bgr[i])
            palette.append(PaletteEntry(i, (r, g, b), int(counts[i])))
        comps = []
        for cid in range(1, num_cc):
            x = int(stats[cid, cv2.CC_STAT_LEFT])
            y = int(stats[cid, cv2.CC_STAT_TOP])
            cw = int(stats[cid, cv2.CC_STAT_WIDTH])
            ch = int(stats[cid, cv2.CC_STAT_HEIGHT])
            area = int(stats[cid, cv2.CC_STAT_AREA])
            comps.append({"id": cid, "bbox": [x, y, cw, ch], "area": area})
        dump = {
            "input": input_path,
            "original_size": [w0, h0],
            "processed_size": [w, h],
            "scale": scale,
            "colors": colors,
            "denoise": denoise,
            "background_label": bg_label,
            "background_rgb": list(bg_rgb),
            "palette": [{"idx": p.idx, "rgb": list(p.rgb), "count": p.count} for p in palette],
            "components": comps,
        }
        Path(dump_json).write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")

    inv_scale = 1.0 / scale
    svg_parts: List[str] = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w0}" height="{h0}" viewBox="0 0 {w0} {h0}">')
    svg_parts.append(f'<rect x="0" y="0" width="100%" height="100%" fill="rgb({bg_rgb[0]},{bg_rgb[1]},{bg_rgb[2]})"/>')
    svg_parts.append(f'<g id="vectorized" transform="scale({inv_scale:g})" shape-rendering="geometricPrecision">')

    for cid in range(1, num_cc):
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        cw = int(stats[cid, cv2.CC_STAT_WIDTH])
        ch = int(stats[cid, cv2.CC_STAT_HEIGHT])
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        cc_roi = cc_map[y : y + ch, x : x + cw]
        lab_roi = labels[y : y + ch, x : x + cw]
        present = [int(p) for p in np.unique(lab_roi[cc_roi == cid]) if int(p) != bg_label]
        if not present:
            continue

        svg_parts.append(f'  <g id="cc_{cid:05d}" data-bbox="{x},{y},{cw},{ch}">')
        svg_parts.append(f'    <title>cc_{cid:05d} bbox=({x},{y},{cw},{ch})</title>')

        for color_idx in present:
            color_mask = ((lab_roi == color_idx) & (cc_roi == cid)).astype(np.uint8) * 255
            if morph_close > 0:
                k = morph_close
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            d = mask_to_svg_path(color_mask, offset_xy=(x, y), epsilon_ratio=epsilon, min_area=min_area)
            if not d:
                continue
            b, g, r = map(int, centers_bgr[color_idx])
            svg_parts.append(
                f'    <path id="cc_{cid:05d}_c{color_idx:02d}" '
                f'fill="rgb({r},{g},{b})" fill-rule="evenodd" d="{d}"/>'
            )

        svg_parts.append("  </g>")

    svg_parts.append("</g></svg>")
    Path(output_svg).write_text("\n".join(svg_parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="layout不要の自動トレース→SVG化")
    ap.add_argument("input", help="入力画像 (png/jpg)")
    ap.add_argument("output", help="出力SVG")
    ap.add_argument("--dump", default=None, help="中間JSONを書き出すパス")
    ap.add_argument("--colors", type=int, default=32, help="減色数 (16-64 推奨)")
    ap.add_argument("--scale", type=float, default=2.0, help="処理時の拡大倍率 (1-4)")
    ap.add_argument("--denoise", choices=["none", "bilateral", "nlmeans"], default="nlmeans", help="ノイズ除去モード")
    ap.add_argument("--min-area", type=int, default=12, help="小領域除去・輪郭無視の閾値")
    ap.add_argument("--epsilon", type=float, default=0.002, help="輪郭簡略化率（小さいほど忠実）")
    ap.add_argument("--morph-close", type=int, default=0, help="モルフォロジーCloseのカーネルサイズ (0で無効)")
    args = ap.parse_args()

    auto_vectorize(
        input_path=args.input,
        output_svg=args.output,
        dump_json=args.dump,
        colors=args.colors,
        scale=args.scale,
        denoise=args.denoise,
        min_area=args.min_area,
        epsilon=args.epsilon,
        morph_close=args.morph_close,
    )


if __name__ == "__main__":
    main()

"""SVGのrect群を解析してポリゴンpathに最適化するツール。

rect群で構成されたSVGを入力として、同色のrectをグループ化し、
連結成分の輪郭を抽出してポリゴンpathとして出力する。

使い方:
    python -m image_to_svg.svg_optimizer --input rects.svg --output optimized.svg
    python -m image_to_svg.svg_optimizer --input rects.svg --output optimized.svg --epsilon 2.0
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union
from xml.etree import ElementTree as ET

# OpenCV/NumPyの遅延インポート（利用可能かチェック）
_OPENCV_AVAILABLE = False
try:
    import cv2
    import numpy as np

    _OPENCV_AVAILABLE = True
except ImportError:
    pass

# EasyOCRの遅延インポート
_EASYOCR_AVAILABLE = False
_easyocr_reader = None
try:
    import easyocr

    _EASYOCR_AVAILABLE = True
except ImportError:
    pass


@dataclass
class Rect:
    """SVGのrect要素を表すデータクラス"""

    x: int
    y: int
    width: int
    height: int
    color: str  # "rgb(r,g,b)" 形式


# =============================================================================
# 色ユーティリティ（アンチエイリアス対策 / 色数削減用）
# =============================================================================


_RGB_RE = re.compile(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")


def _parse_rgb(color: str) -> Tuple[int, int, int] | None:
    """"rgb(r,g,b)" 形式を (r,g,b) にパース。"""
    m = _RGB_RE.fullmatch(color.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _rgb_to_css(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"rgb({r},{g},{b})"


def _rgb_dist(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    # ユークリッド距離（簡易）
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _is_gray(rgb: Tuple[int, int, int], tol: int = 0) -> bool:
    r, g, b = rgb
    return abs(r - g) <= tol and abs(g - b) <= tol and abs(r - b) <= tol


def _luma(rgb: Tuple[int, int, int]) -> float:
    # ITU-R BT.601 近似（0..255）
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def build_area_by_color(rects: List[Rect]) -> Dict[str, int]:
    """色ごとの総面積（≒ピクセル数）を算出。"""
    area: Dict[str, int] = defaultdict(int)
    for r in rects:
        if r.width > 0 and r.height > 0:
            area[r.color] += r.width * r.height
    return dict(area)


def build_color_map(
    area_by_color: Dict[str, int],
    *,
    max_colors: int = 0,
    anchor_min_dist: float = 0.0,
    snap_gray: bool = False,
    gray_tol: int = 0,
    gray_threshold: float = 128.0,
) -> Dict[str, str]:
    """色の置換マップを作る。

    - snap_gray=True のとき(ほぼ)グレーを白/黒に二値化（アンチエイリアスの粗さ抑制に効く）
    - max_colors>0 のとき、代表色（アンカー）に寄せて色数を削減
    - anchor_min_dist>0 のとき、アンカー同士が近すぎる場合は追加しない（グレーが乱立するのを防ぐ）
    """

    if not area_by_color:
        return {}

    # 1) まず任意でグレーを白/黒へスナップ
    pre_map: Dict[str, str] = {}
    for c in area_by_color.keys():
        rgb = _parse_rgb(c)
        if rgb is None:
            pre_map[c] = c
            continue
        if snap_gray and _is_gray(rgb, tol=gray_tol):
            pre_map[c] = "rgb(0,0,0)" if _luma(rgb) < gray_threshold else "rgb(255,255,255)"
        else:
            pre_map[c] = c

    # スナップ後の面積を再集計
    merged_area: Dict[str, int] = defaultdict(int)
    for c, a in area_by_color.items():
        merged_area[pre_map[c]] += a
    merged_area = dict(merged_area)

    # max_colors 無効なら pre_map が最終
    if max_colors <= 0 or len(merged_area) <= max_colors:
        # pre_map は、元色→スナップ先なので、merged_areaのキーに存在しない場合はそのまま
        return pre_map

    # 2) アンカーを面積順に選ぶ（近すぎる色はスキップ）
    by_area = sorted(merged_area.items(), key=lambda x: -x[1])
    anchors: List[str] = []
    anchors_rgb: List[Tuple[int, int, int]] = []

    for c, _a in by_area:
        rgb = _parse_rgb(c)
        if rgb is None:
            continue
        if not anchors:
            anchors.append(c)
            anchors_rgb.append(rgb)
            if len(anchors) >= max_colors:
                break
            continue

        if anchor_min_dist > 0:
            if min(_rgb_dist(rgb, ar) for ar in anchors_rgb) < anchor_min_dist:
                continue
        anchors.append(c)
        anchors_rgb.append(rgb)
        if len(anchors) >= max_colors:
            break

    # もし距離制約でアンカーが足りない場合は、残りを距離無視で取る
    if len(anchors) < max_colors:
        for c, _a in by_area:
            if c in anchors:
                continue
            rgb = _parse_rgb(c)
            if rgb is None:
                continue
            anchors.append(c)
            anchors_rgb.append(rgb)
            if len(anchors) >= max_colors:
                break

    # 3) スナップ後の各色を最近傍アンカーへ
    post_map: Dict[str, str] = {}
    for c in merged_area.keys():
        rgb = _parse_rgb(c)
        if rgb is None or not anchors:
            post_map[c] = c
            continue
        best_i = 0
        best_d = 1e18
        for i, ar in enumerate(anchors_rgb):
            d = _rgb_dist(rgb, ar)
            if d < best_d:
                best_d = d
                best_i = i
        post_map[c] = anchors[best_i]

    # 4) 元色→スナップ先→アンカー へ変換
    final_map: Dict[str, str] = {}
    for orig, snapped in pre_map.items():
        final_map[orig] = post_map.get(snapped, snapped)
    return final_map


# =============================================================================
# OCR機能（テキスト検出・<text>要素生成）
# =============================================================================


@dataclass
class TextRegion:
    """OCRで検出されたテキスト領域"""

    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    text: str
    confidence: float


def _get_easyocr_reader(languages: List[str]) -> "easyocr.Reader":
    """EasyOCRリーダーを取得（シングルトン）"""
    global _easyocr_reader
    if _easyocr_reader is None:
        print(f"EasyOCRを初期化中（言語: {languages}）...")
        _easyocr_reader = easyocr.Reader(languages, gpu=False)
    return _easyocr_reader


def rects_to_image(
    width: int, height: int, rects: List["Rect"]
) -> "np.ndarray":
    """
    rect群からRGB画像（NumPy配列）を生成

    Args:
        width: 画像幅
        height: 画像高さ
        rects: rect要素のリスト

    Returns:
        RGB画像（uint8, shape: (height, width, 3)）
    """
    # 白背景で初期化
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    for rect in rects:
        rgb = _parse_rgb(rect.color)
        if rgb is None:
            continue
        y1 = rect.y
        y2 = min(rect.y + rect.height, height)
        x1 = rect.x
        x2 = min(rect.x + rect.width, width)
        # BGRではなくRGBで設定
        image[y1:y2, x1:x2] = rgb

    return image


def detect_text_with_ocr(
    image: "np.ndarray",
    languages: List[str] = ["ja", "en"],
    min_confidence: float = 0.5,
) -> List[TextRegion]:
    """
    OCRでテキスト領域を検出

    Args:
        image: RGB画像（NumPy配列）
        languages: OCR言語リスト
        min_confidence: 最小信頼度

    Returns:
        検出されたTextRegionのリスト
    """
    if not _EASYOCR_AVAILABLE:
        print("警告: EasyOCRが利用できません。pip install easyocr を実行してください。")
        return []

    reader = _get_easyocr_reader(languages)

    # EasyOCRはRGB画像を受け取る
    results = reader.readtext(image)

    text_regions = []
    for bbox, text, confidence in results:
        if confidence < min_confidence:
            continue

        # bboxは[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]の形式
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x = int(min(x_coords))
        y = int(min(y_coords))
        w = int(max(x_coords) - x)
        h = int(max(y_coords) - y)

        text_regions.append(TextRegion(
            bbox=(x, y, w, h),
            text=text,
            confidence=confidence,
        ))

    return text_regions


def estimate_font_size(height: int) -> int:
    """テキスト領域の高さからフォントサイズを推定"""
    # 経験的な係数（高さの約80%がフォントサイズ）
    return max(8, int(height * 0.8))


def text_regions_to_svg_elements(
    text_regions: List[TextRegion],
    font_family: str = "Noto Sans CJK JP, Noto Sans JP, Hiragino Kaku Gothic ProN, Meiryo, sans-serif",
) -> List[str]:
    """
    TextRegionリストをSVG <text>要素に変換

    Args:
        text_regions: OCRで検出されたテキスト領域
        font_family: フォントファミリー

    Returns:
        SVG <text>要素の文字列リスト
    """
    from xml.sax.saxutils import escape

    elements = []
    for region in text_regions:
        x, y, w, h = region.bbox
        font_size = estimate_font_size(h)

        # テキストのベースライン位置（領域の下端に近い位置）
        text_y = y + int(h * 0.85)
        text_x = x

        escaped_text = escape(region.text)

        element = (
            f'<text x="{text_x}" y="{text_y}" '
            f'font-family="{font_family}" '
            f'font-size="{font_size}" '
            f'fill="rgb(0,0,0)">{escaped_text}</text>'
        )
        elements.append(element)

    return elements


def create_text_mask(
    width: int,
    height: int,
    text_regions: List[TextRegion],
    padding: int = 2,
) -> "np.ndarray":
    """
    テキスト領域をマスクするビットマップを作成

    Args:
        width: 画像幅
        height: 画像高さ
        text_regions: テキスト領域リスト
        padding: 領域を少し拡大するパディング

    Returns:
        マスク画像（255=テキスト領域、0=それ以外）
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for region in text_regions:
        x, y, w, h = region.bbox
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        mask[y1:y2, x1:x2] = 255

    return mask


def _parse_dimension(value: str | None, default: int = 0) -> int:
    """
    SVGの寸法値をパース（パーセント、px単位に対応）

    Args:
        value: 寸法値の文字列（例: "100", "100px", "100%"）
        default: パースできない場合のデフォルト値

    Returns:
        整数値（パーセントの場合は0を返す）
    """
    if value is None:
        return default
    value = value.strip()
    if value.endswith("%"):
        return 0  # パーセントは後でviewBoxから取得
    if value.endswith("px"):
        value = value[:-2]
    try:
        return int(float(value))
    except ValueError:
        return default


def parse_svg_rects(svg_path: str) -> Tuple[int, int, List[Rect]]:
    """
    SVGファイルからrect要素をパース

    Args:
        svg_path: 入力SVGファイルパス

    Returns:
        (width, height, rects): SVGサイズとrectリスト
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # SVGサイズ取得（width/heightがパーセントの場合はviewBoxを使用）
    width = _parse_dimension(root.get("width"))
    height = _parse_dimension(root.get("height"))

    # viewBoxからサイズを取得（width/heightが0の場合）
    if width == 0 or height == 0:
        viewbox = root.get("viewBox")
        if viewbox:
            parts = viewbox.split()
            if len(parts) >= 4:
                if width == 0:
                    width = int(float(parts[2]))
                if height == 0:
                    height = int(float(parts[3]))

    rects = []
    for elem in root.iter():
        if elem.tag.endswith("rect"):
            x = _parse_dimension(elem.get("x"))
            y = _parse_dimension(elem.get("y"))
            w = _parse_dimension(elem.get("width"))
            h = _parse_dimension(elem.get("height"))
            fill = elem.get("fill", "")
            if fill and w > 0 and h > 0:
                rects.append(Rect(x, y, w, h, fill))

    return width, height, rects


def rects_to_bitmap(
    width: int, height: int, rects: List[Rect]
) -> Dict[str, List[List[bool]]]:
    """
    rect群を色ごとの2Dビットマップに変換

    Args:
        width: 画像幅
        height: 画像高さ
        rects: rect要素のリスト

    Returns:
        {color: 2D bool array} の辞書
    """
    bitmaps: Dict[str, List[List[bool]]] = defaultdict(
        lambda: [[False] * width for _ in range(height)]
    )

    for rect in rects:
        bitmap = bitmaps[rect.color]
        for y in range(rect.y, min(rect.y + rect.height, height)):
            for x in range(rect.x, min(rect.x + rect.width, width)):
                bitmap[y][x] = True

    return dict(bitmaps)


# =============================================================================
# OpenCV版（高速）
# =============================================================================


def rects_to_bitmap_numpy(
    width: int, height: int, rects: List[Rect]
) -> Dict[str, "np.ndarray"]:
    """
    rect群を色ごとのNumPy配列に変換（OpenCV用）

    Args:
        width: 画像幅
        height: 画像高さ
        rects: rect要素のリスト

    Returns:
        {color: numpy uint8 array} の辞書
    """
    bitmaps: Dict[str, np.ndarray] = {}

    for rect in rects:
        if rect.color not in bitmaps:
            bitmaps[rect.color] = np.zeros((height, width), dtype=np.uint8)

        bitmap = bitmaps[rect.color]
        y1 = rect.y
        y2 = min(rect.y + rect.height, height)
        x1 = rect.x
        x2 = min(rect.x + rect.width, width)
        bitmap[y1:y2, x1:x2] = 255

    return bitmaps


def rects_to_bitmap_numpy_mapped(
    width: int,
    height: int,
    rects: List[Rect],
    color_map: Dict[str, str],
) -> Dict[str, "np.ndarray"]:
    """rect群を色変換しつつ色ごとのNumPy配列に変換（OpenCV用）。

    Args:
        width: 画像幅
        height: 画像高さ
        rects: rect要素
        color_map: 元色 → 変換後色 のマップ

    Returns:
        {mapped_color: numpy uint8 array}
    """
    bitmaps: Dict[str, np.ndarray] = {}

    for rect in rects:
        mapped = color_map.get(rect.color, rect.color)
        if mapped not in bitmaps:
            bitmaps[mapped] = np.zeros((height, width), dtype=np.uint8)

        bitmap = bitmaps[mapped]
        y1 = rect.y
        y2 = min(rect.y + rect.height, height)
        x1 = rect.x
        x2 = min(rect.x + rect.width, width)
        bitmap[y1:y2, x1:x2] = 255

    return bitmaps


def trace_contour_numpy(
    bitmap: "np.ndarray",
) -> List[List[Tuple[int, int]]]:
    """
    NumPyを使って高速に境界エッジを収集し、輪郭をトレース

    純Python版のtrace_contour_simpleと同じアルゴリズムだが、
    NumPyのベクトル演算で高速化。

    Args:
        bitmap: uint8のNumPy配列（255=塗りつぶし、0=背景）

    Returns:
        輪郭のリスト [[(x, y), ...], ...]
    """
    h, w = bitmap.shape
    filled = bitmap > 0

    # パディングを追加して境界チェックを簡略化
    padded = np.pad(filled, 1, mode="constant", constant_values=False)

    # 境界エッジを検出（隣接ピクセルとの差分）
    # 上辺: 現在のピクセルが塗りつぶしで、上隣が空
    top_edges = filled & ~padded[:-2, 1:-1]
    # 下辺: 現在のピクセルが塗りつぶしで、下隣が空
    bottom_edges = filled & ~padded[2:, 1:-1]
    # 左辺: 現在のピクセルが塗りつぶしで、左隣が空
    left_edges = filled & ~padded[1:-1, :-2]
    # 右辺: 現在のピクセルが塗りつぶしで、右隣が空
    right_edges = filled & ~padded[1:-1, 2:]

    # エッジをセットに収集
    edges: set = set()

    # 上辺エッジ
    ys, xs = np.where(top_edges)
    for x, y in zip(xs, ys):
        edges.add(((x, y), (x + 1, y)))

    # 下辺エッジ（逆方向）
    ys, xs = np.where(bottom_edges)
    for x, y in zip(xs, ys):
        edges.add(((x + 1, y + 1), (x, y + 1)))

    # 左辺エッジ（逆方向）
    ys, xs = np.where(left_edges)
    for x, y in zip(xs, ys):
        edges.add(((x, y + 1), (x, y)))

    # 右辺エッジ
    ys, xs = np.where(right_edges)
    for x, y in zip(xs, ys):
        edges.add(((x + 1, y), (x + 1, y + 1)))

    if not edges:
        return []

    # エッジをチェーン化
    edge_from_start: Dict[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = {}
    for edge in edges:
        start_pt, end_pt = edge
        edge_from_start[start_pt] = edge

    # 全ての輪郭をトレース
    all_contours = []
    used_edges: set = set()

    for start_pt in edge_from_start:
        if start_pt in used_edges:
            continue

        # この開始点から輪郭をトレース
        contour = [start_pt]
        current_pt = start_pt
        visited_in_this_contour: set = set()

        max_steps = len(edges) + 1
        for _ in range(max_steps):
            if current_pt not in edge_from_start:
                break

            edge = edge_from_start[current_pt]
            if edge in visited_in_this_contour:
                break

            visited_in_this_contour.add(edge)
            used_edges.add(edge[0])
            _, end_pt = edge
            contour.append(end_pt)
            current_pt = end_pt

            if current_pt == start_pt:
                break

        if len(contour) < 4:  # 閉じた輪郭には最低4点必要（三角形+戻り）
            continue

        # 連続する同一方向のエッジを統合（角だけを残す）
        simplified = [contour[0]]
        for i in range(1, len(contour) - 1):
            prev = contour[i - 1]
            curr = contour[i]
            next_pt = contour[i + 1]

            dx1 = curr[0] - prev[0]
            dy1 = curr[1] - prev[1]
            dx2 = next_pt[0] - curr[0]
            dy2 = next_pt[1] - curr[1]

            if (dx1, dy1) != (dx2, dy2):
                simplified.append(curr)

        if len(simplified) >= 3:
            all_contours.append(simplified)

    return all_contours


def find_contours_opencv(
    bitmap: "np.ndarray", epsilon: float = 1.0
) -> List[List[Tuple[int, int]]]:
    """
    NumPy高速化された境界エッジ収集と輪郭トレース

    trace_contour_numpy()のエッジ上書きバグを修正した版。
    同じ開始点から複数のエッジが出る場合も正しく処理する。

    Args:
        bitmap: uint8のNumPy配列（255=塗りつぶし、0=背景）
        epsilon: 輪郭簡略化の許容誤差（approxPolyDPに渡す）

    Returns:
        輪郭のリスト [[(x, y), ...], ...]  境界座標
    """
    h, w = bitmap.shape
    filled = bitmap > 0

    # パディングを追加して境界チェックを簡略化
    padded = np.pad(filled, 1, mode="constant", constant_values=False)

    # 境界エッジを検出（隣接ピクセルとの差分）
    top_edges = filled & ~padded[:-2, 1:-1]
    bottom_edges = filled & ~padded[2:, 1:-1]
    left_edges = filled & ~padded[1:-1, :-2]
    right_edges = filled & ~padded[1:-1, 2:]

    # エッジをリストに収集（順序付き）
    # 各エッジは (start_pt, end_pt) のタプル
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    # 上辺エッジ: (x, y) → (x+1, y)
    ys, xs = np.where(top_edges)
    for x, y in zip(xs, ys):
        edges.append(((x, y), (x + 1, y)))

    # 下辺エッジ: (x+1, y+1) → (x, y+1)
    ys, xs = np.where(bottom_edges)
    for x, y in zip(xs, ys):
        edges.append(((x + 1, y + 1), (x, y + 1)))

    # 左辺エッジ: (x, y+1) → (x, y)
    ys, xs = np.where(left_edges)
    for x, y in zip(xs, ys):
        edges.append(((x, y + 1), (x, y)))

    # 右辺エッジ: (x+1, y) → (x+1, y+1)
    ys, xs = np.where(right_edges)
    for x, y in zip(xs, ys):
        edges.append(((x + 1, y), (x + 1, y + 1)))

    if not edges:
        return []

    # エッジをチェーン化（複数エッジ対応版）
    # start_pt → [edge1, edge2, ...] のマッピング
    from collections import defaultdict

    edge_from_start: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], Tuple[int, int]]]] = defaultdict(list)
    for edge in edges:
        start_pt, end_pt = edge
        edge_from_start[start_pt].append(edge)

    # 全ての輪郭をトレース
    all_contours = []
    used_edges: set = set()

    for start_pt_key, edge_list in edge_from_start.items():
        for initial_edge in edge_list:
            if initial_edge in used_edges:
                continue

            # この開始点から輪郭をトレース
            start_pt = initial_edge[0]
            contour = [start_pt]
            current_pt = start_pt
            current_edge = initial_edge

            max_steps = len(edges) + 1
            for _ in range(max_steps):
                used_edges.add(current_edge)
                _, end_pt = current_edge
                contour.append(end_pt)
                current_pt = end_pt

                if current_pt == start_pt:
                    break

                # 次のエッジを探す
                next_edges = edge_from_start.get(current_pt, [])
                next_edge = None
                for e in next_edges:
                    if e not in used_edges:
                        next_edge = e
                        break

                if next_edge is None:
                    break

                current_edge = next_edge

            if len(contour) < 4:  # 閉じた輪郭には最低4点必要
                continue

            # 連続する同一方向のエッジを統合（角だけを残す）
            simplified = [contour[0]]
            for i in range(1, len(contour) - 1):
                prev = contour[i - 1]
                curr = contour[i]
                next_pt = contour[i + 1]

                dx1 = curr[0] - prev[0]
                dy1 = curr[1] - prev[1]
                dx2 = next_pt[0] - curr[0]
                dy2 = next_pt[1] - curr[1]

                if (dx1, dy1) != (dx2, dy2):
                    simplified.append(curr)

            if len(simplified) >= 3:
                # approxPolyDPで更に簡略化
                pts_array = np.array(simplified, dtype=np.float32).reshape(-1, 1, 2)
                approx = cv2.approxPolyDP(pts_array, epsilon, closed=True)
                final_points = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
                if len(final_points) >= 3:
                    all_contours.append(final_points)

    return all_contours


# =============================================================================
# 輪郭 → 複合パス（穴対応）
# =============================================================================


def _signed_area(points: List[Tuple[int, int]]) -> float:
    """多角形の符号付き面積（shoelace）。"""
    if len(points) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _bbox(points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _find_inside_point(points: List[Tuple[int, int]]) -> Tuple[float, float]:
    """輪郭の内部点を1つ返す（包含判定用）。"""
    x0, y0, x1, y1 = _bbox(points)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    cnt = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    def inside(px: float, py: float) -> bool:
        return cv2.pointPolygonTest(cnt, (float(px), float(py)), False) > 0

    if inside(cx, cy):
        return cx, cy

    # 近傍探索（軽量）
    for r in range(1, 20):
        for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r), (r, r), (-r, r), (r, -r), (-r, -r)):
            px, py = cx + dx, cy + dy
            if inside(px, py):
                return px, py

    # bbox内を粗く走査（細長い輪郭などで中心が外に出るケースの救済）
    # 小さいbboxなら 1px 刻み、大きいbboxなら粗めに
    bw = max(0, x1 - x0)
    bh = max(0, y1 - y0)
    step = max(1, int(min(bw, bh) // 10))
    for yy in range(int(y0) + 1, int(y1), step):
        for xx in range(int(x0) + 1, int(x1), step):
            if inside(xx + 0.5, yy + 0.5):
                return xx + 0.5, yy + 0.5

    # fallback
    return points[0][0] + 0.5, points[0][1] + 0.5


def _contour_to_d(points: List[Tuple[int, int]]) -> str:
    if len(points) < 3:
        return ""
    d = f"M {points[0][0]} {points[0][1]}"
    for x, y in points[1:]:
        d += f" L {x} {y}"
    d += " Z"
    return d


def group_contours_by_nesting(
    contours: List[List[Tuple[int, int]]],
    *,
    min_area: float = 0.0,
) -> List[List[List[Tuple[int, int]]]]:
    """輪郭を包含（ネスト）関係でグループ化し、rootごとに返す。

    evenodd fill-rule を使う前提で、外周/穴/島をすべて同一pathのサブパスとして
    まとめると、穴が潰れずに描画できる。
    """

    # 面積の小さいノイズを除去しつつ情報を作る
    polys: List[List[Tuple[int, int]]] = []
    areas: List[float] = []
    testpts: List[Tuple[float, float]] = []
    bboxes: List[Tuple[int, int, int, int]] = []

    for p in contours:
        a = abs(_signed_area(p))
        if a < max(0.0, min_area):
            continue
        polys.append(p)
        areas.append(a)
        bboxes.append(_bbox(p))
        testpts.append(_find_inside_point(p))

    if not polys:
        return []

    # 面積降順で処理（親は必ず自分より大きい面積）
    order = sorted(range(len(polys)), key=lambda i: -areas[i])
    polys = [polys[i] for i in order]
    areas = [areas[i] for i in order]
    testpts = [testpts[i] for i in order]
    bboxes = [bboxes[i] for i in order]

    cnt_np = [np.array(p, dtype=np.float32).reshape(-1, 1, 2) for p in polys]

    parent = [-1] * len(polys)
    for i in range(len(polys)):
        px, py = testpts[i]
        best_parent = -1
        best_area = 1e30
        for j in range(i):
            # bbox でまず粗く弾く
            x0, y0, x1, y1 = bboxes[j]
            if not (x0 <= px <= x1 and y0 <= py <= y1):
                continue
            if cv2.pointPolygonTest(cnt_np[j], (float(px), float(py)), False) > 0:
                if areas[j] < best_area:
                    best_area = areas[j]
                    best_parent = j
        parent[i] = best_parent

    children: List[List[int]] = [[] for _ in range(len(polys))]
    roots: List[int] = []
    for i, p in enumerate(parent):
        if p == -1:
            roots.append(i)
        else:
            children[p].append(i)

    def collect_subtree(r: int, out: List[int]) -> None:
        out.append(r)
        for ch in children[r]:
            collect_subtree(ch, out)

    groups: List[List[List[Tuple[int, int]]]] = []
    for r in roots:
        idxs: List[int] = []
        collect_subtree(r, idxs)
        groups.append([polys[k] for k in idxs])
    return groups


def optimize_svg_opencv(
    input_path: str,
    output_path: str,
    epsilon: float = 1.0,
    *,
    max_colors: int = 0,
    anchor_min_dist: float = 0.0,
    snap_gray: bool = False,
    gray_tol: int = 0,
    gray_threshold: float = 128.0,
    skip_background: bool = False,
    min_area: float = 0.0,
    use_ocr: bool = False,
    ocr_languages: List[str] | None = None,
    ocr_min_confidence: float = 0.5,
) -> dict:
    """
    OpenCVを使用した高速版optimize_svg

    Args:
        input_path: 入力SVGパス（rect群）
        output_path: 出力SVGパス（path群）
        epsilon: 輪郭簡略化の許容誤差
        use_ocr: OCRでテキスト検出を行う
        ocr_languages: OCR言語リスト（デフォルト: ["ja", "en"]）
        ocr_min_confidence: OCR最小信頼度

    Returns:
        {"input_rects": N, "output_paths": M, "colors": K, "text_regions": L}
    """
    # 1. SVGパース
    print(f"SVGをパース中: {input_path}")
    width, height, rects = parse_svg_rects(input_path)
    print(f"  サイズ: {width}x{height}, rect数: {len(rects)}")

    # 2. 色の面積集計 → 背景推定（最大面積の色）
    area_by_color = build_area_by_color(rects)
    bg_original = max(area_by_color.items(), key=lambda x: x[1])[0] if area_by_color else "rgb(255,255,255)"

    # 3. 色の置換マップ作成（任意: グレー二値化 / 色数削減）
    color_map = build_color_map(
        area_by_color,
        max_colors=max_colors,
        anchor_min_dist=anchor_min_dist,
        snap_gray=snap_gray,
        gray_tol=gray_tol,
        gray_threshold=gray_threshold,
    )
    bg_color = color_map.get(bg_original, bg_original)

    # 4. 色ごとにNumPyビットマップ化（置換後の色で集計）
    print("ビットマップに変換中（NumPy）...")
    bitmaps = rects_to_bitmap_numpy_mapped(width, height, rects, color_map)
    print(f"  色数: {len(bitmaps)}")
    if skip_background:
        print(f"  背景色(推定): {bg_color}  ※path化せず<rect>で出力")

    # 4.5. OCR処理（オプション）
    text_regions: List[TextRegion] = []
    text_mask: "np.ndarray | None" = None
    if use_ocr:
        if not _EASYOCR_AVAILABLE:
            print("警告: EasyOCRが利用できません。--ocr オプションは無視されます。")
            print("  インストール: pip install easyocr")
        else:
            print("OCRでテキストを検出中...")
            # rect群から画像を生成
            image = rects_to_image(width, height, rects)
            # OCR実行
            languages = ocr_languages if ocr_languages else ["ja", "en"]
            text_regions = detect_text_with_ocr(image, languages, ocr_min_confidence)
            print(f"  検出テキスト数: {len(text_regions)}")
            for region in text_regions:
                print(f"    - \"{region.text}\" (信頼度: {region.confidence:.2f})")
            # テキスト領域のマスクを作成
            if text_regions:
                text_mask = create_text_mask(width, height, text_regions, padding=2)

    # 5. 各色の輪郭を抽出して、rootごとの複合パスにまとめる
    print("輪郭を抽出中（OpenCV）...")
    paths: List[str] = []

    # 背景は最背面に1枚
    if skip_background:
        paths.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg_color}"/>')

    # 描画順: 面積の大きい色を先に描画 → 小さなディテールを後に
    mapped_area: Dict[str, int] = defaultdict(int)
    for orig, a in area_by_color.items():
        mapped_area[color_map.get(orig, orig)] += a

    items = sorted(bitmaps.items(), key=lambda kv: -mapped_area.get(kv[0], 0))

    for i, (color, bitmap) in enumerate(items):
        if skip_background and color == bg_color:
            continue

        # テキスト領域をマスク（OCR有効時）
        if text_mask is not None:
            bitmap = bitmap.copy()
            bitmap[text_mask > 0] = 0

        contours = find_contours_opencv(bitmap, epsilon)
        groups = group_contours_by_nesting(contours, min_area=min_area)

        for polys in groups:
            d_parts = [_contour_to_d(p) for p in polys]
            d = " ".join([x for x in d_parts if x])
            if d:
                paths.append(f'<path d="{d}" fill="{color}" fill-rule="evenodd"/>')

        # 進捗表示（色が多い場合）
        if len(items) > 10 and (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(items)} 色処理完了")

    print(f"  path数: {len(paths)}")

    # 6. テキスト要素を生成（OCR有効時）
    text_elements: List[str] = []
    if text_regions:
        text_elements = text_regions_to_svg_elements(text_regions)
        print(f"  text要素数: {len(text_elements)}")

    # 7. SVG出力
    svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    svg_content += "\n".join(paths)
    if text_elements:
        svg_content += "\n" + "\n".join(text_elements)
    svg_content += "\n</svg>"

    Path(output_path).write_text(svg_content, encoding="utf-8")
    print(f"出力: {output_path}")

    return {
        "input_rects": len(rects),
        "output_paths": len(paths),
        "colors": len(bitmaps),
        "text_regions": len(text_regions),
    }


# =============================================================================
# 純Python版（フォールバック）
# =============================================================================


def trace_contour_simple(
    bitmap: List[List[bool]],
    start_x: int,
    start_y: int,
) -> List[Tuple[int, int]]:
    """
    連結成分の境界をトレースする

    ピクセルの外周を時計回りにトレースして頂点列を返す。
    Pavlidis contour tracing algorithm を使用。

    Args:
        bitmap: 2Dビットマップ
        start_x: 開始ピクセルのX座標（境界の左端）
        start_y: 開始ピクセルのY座標

    Returns:
        境界頂点のリスト [(x, y), ...]
    """
    height = len(bitmap)
    width = len(bitmap[0]) if height > 0 else 0

    def is_filled(x: int, y: int) -> bool:
        if 0 <= x < width and 0 <= y < height:
            return bitmap[y][x]
        return False

    if not is_filled(start_x, start_y):
        return []

    # 境界エッジを収集してチェーン化する方法
    # 各エッジは (x1, y1, x2, y2) のセグメント

    # まず、全ての境界セグメントを収集
    # ピクセル境界は4方向：上、右、下、左
    edges: set = set()

    # 領域内の全ピクセルを走査して境界エッジを収集
    for y in range(height):
        for x in range(width):
            if bitmap[y][x]:
                # 上辺：上隣が空なら境界
                if y == 0 or not bitmap[y - 1][x]:
                    edges.add(((x, y), (x + 1, y)))
                # 下辺：下隣が空なら境界
                if y == height - 1 or not bitmap[y + 1][x]:
                    edges.add(((x + 1, y + 1), (x, y + 1)))
                # 左辺：左隣が空なら境界
                if x == 0 or not bitmap[y][x - 1]:
                    edges.add(((x, y + 1), (x, y)))
                # 右辺：右隣が空なら境界
                if x == width - 1 or not bitmap[y][x + 1]:
                    edges.add(((x + 1, y), (x + 1, y + 1)))

    if not edges:
        return []

    # エッジをチェーン化（end_point → edge のマップを作成）
    edge_from_start: Dict[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = {}
    for edge in edges:
        start_pt, end_pt = edge
        edge_from_start[start_pt] = edge

    # 開始点から時計回りにトレース
    # start_x, start_y ピクセルの左上角から開始
    start_pt = (start_x, start_y)

    if start_pt not in edge_from_start:
        # 左上角から始まるエッジがない場合、別の開始点を探す
        for pt in edge_from_start:
            if pt[0] == start_x:  # 同じx座標の点
                start_pt = pt
                break
        else:
            start_pt = next(iter(edge_from_start.keys()))

    contour = [start_pt]
    current_pt = start_pt
    visited_edges: set = set()

    max_steps = len(edges) + 1
    for _ in range(max_steps):
        if current_pt not in edge_from_start:
            break

        edge = edge_from_start[current_pt]
        if edge in visited_edges:
            break

        visited_edges.add(edge)
        _, end_pt = edge
        contour.append(end_pt)
        current_pt = end_pt

        if current_pt == start_pt:
            break

    # 連続する同一方向のエッジを統合（角だけを残す）
    if len(contour) < 3:
        return contour

    simplified = [contour[0]]
    for i in range(1, len(contour) - 1):
        prev = contour[i - 1]
        curr = contour[i]
        next_pt = contour[i + 1]

        # 方向が変わるかチェック
        dx1 = curr[0] - prev[0]
        dy1 = curr[1] - prev[1]
        dx2 = next_pt[0] - curr[0]
        dy2 = next_pt[1] - curr[1]

        if (dx1, dy1) != (dx2, dy2):
            simplified.append(curr)

    # 最後の点と最初の点の方向もチェック
    if len(contour) >= 3:
        prev = contour[-2]
        curr = contour[-1]
        next_pt = contour[1]  # 閉じているので最初に戻る

        dx1 = curr[0] - prev[0]
        dy1 = curr[1] - prev[1]
        dx2 = next_pt[0] - simplified[0][0]
        dy2 = next_pt[1] - simplified[0][1]

        if (dx1, dy1) != (dx2, dy2) and curr != simplified[0]:
            simplified.append(curr)

    return simplified


def simplify_contour(
    points: List[Tuple[int, int]], epsilon: float = 1.0
) -> List[Tuple[int, int]]:
    """
    Ramer-Douglas-Peucker アルゴリズムで輪郭を簡略化

    Args:
        points: 頂点リスト
        epsilon: 許容誤差（大きいほど簡略化）

    Returns:
        簡略化された頂点リスト
    """
    if len(points) < 3:
        return points

    # もっとも遠い点を見つける
    start, end = points[0], points[-1]
    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(points) - 1):
        px, py = points[i]
        x1, y1 = start
        x2, y2 = end

        # 線分の長さ
        line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if line_len_sq == 0:
            dist = ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        else:
            t = max(
                0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq)
            )
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            dist = ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left = simplify_contour(points[: max_idx + 1], epsilon)
        right = simplify_contour(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def find_boundary_start(bitmap: List[List[bool]], visited: List[List[bool]]) -> Tuple[int, int] | None:
    """
    まだ処理されていない連結成分の境界開始点を見つける

    Args:
        bitmap: 2Dビットマップ
        visited: 訪問済みフラグ

    Returns:
        (x, y) または None
    """
    height = len(bitmap)
    width = len(bitmap[0]) if height > 0 else 0

    for y in range(height):
        for x in range(width):
            if bitmap[y][x] and not visited[y][x]:
                # 境界上のピクセルかチェック（左隣が空いている）
                if x == 0 or not bitmap[y][x - 1]:
                    return (x, y)
    return None


def flood_fill_mark(
    bitmap: List[List[bool]],
    visited: List[List[bool]],
    start_x: int,
    start_y: int,
) -> None:
    """
    連結成分全体をvisitedにマーク（flood fill）

    Args:
        bitmap: 2Dビットマップ
        visited: 訪問済みフラグ
        start_x: 開始X座標
        start_y: 開始Y座標
    """
    height = len(bitmap)
    width = len(bitmap[0]) if height > 0 else 0

    stack = [(start_x, start_y)]
    while stack:
        cx, cy = stack.pop()
        if 0 <= cx < width and 0 <= cy < height:
            if bitmap[cy][cx] and not visited[cy][cx]:
                visited[cy][cx] = True
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])


def find_connected_components(
    bitmap: List[List[bool]],
) -> List[List[Tuple[int, int]]]:
    """
    ビットマップから連結成分を検出し、各成分の輪郭を返す

    Args:
        bitmap: 2Dビットマップ

    Returns:
        輪郭のリスト [[(x, y), ...], ...]
    """
    height = len(bitmap)
    width = len(bitmap[0]) if height > 0 else 0
    visited = [[False] * width for _ in range(height)]

    components = []

    while True:
        start = find_boundary_start(bitmap, visited)
        if start is None:
            break

        x, y = start
        # 境界をトレース
        contour = trace_contour_simple(bitmap, x, y)
        if len(contour) >= 3:
            components.append(contour)

        # この成分全体をvisitedにマーク
        flood_fill_mark(bitmap, visited, x, y)

    return components


def contour_to_svg_path(points: List[Tuple[int, int]], color: str) -> str:
    """
    輪郭点列をSVG path要素に変換

    Args:
        points: 頂点リスト
        color: 塗りつぶし色

    Returns:
        SVG path要素の文字列
    """
    if len(points) < 3:
        return ""

    d = f"M {points[0][0]} {points[0][1]}"
    for x, y in points[1:]:
        d += f" L {x} {y}"
    d += " Z"

    return f'<path d="{d}" fill="{color}"/>'


def optimize_svg_pure(
    input_path: str,
    output_path: str,
    epsilon: float = 1.0,
) -> dict:
    """
    純Python版optimize_svg（フォールバック用）

    Args:
        input_path: 入力SVGパス（rect群）
        output_path: 出力SVGパス（path群）
        epsilon: 輪郭簡略化の許容誤差

    Returns:
        {"input_rects": N, "output_paths": M, "colors": K}
    """
    # 1. SVGパース
    print(f"SVGをパース中: {input_path}")
    width, height, rects = parse_svg_rects(input_path)
    print(f"  サイズ: {width}x{height}, rect数: {len(rects)}")

    # 2. 色ごとにビットマップ化
    print("ビットマップに変換中（純Python）...")
    bitmaps = rects_to_bitmap(width, height, rects)
    print(f"  色数: {len(bitmaps)}")

    # 3. 各色の連結成分を検出してpath化
    print("輪郭を抽出中（純Python - 大きなSVGでは時間がかかります）...")
    paths = []
    for i, (color, bitmap) in enumerate(bitmaps.items()):
        components = find_connected_components(bitmap)
        for contour in components:
            simplified = simplify_contour(contour, epsilon)
            path_str = contour_to_svg_path(simplified, color)
            if path_str:
                paths.append(path_str)

        # 進捗表示
        if len(bitmaps) > 5:
            print(f"    {i + 1}/{len(bitmaps)} 色処理完了")

    print(f"  path数: {len(paths)}")

    # 4. SVG出力
    svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    svg_content += "\n".join(paths)
    svg_content += "\n</svg>"

    Path(output_path).write_text(svg_content, encoding="utf-8")
    print(f"出力: {output_path}")

    return {
        "input_rects": len(rects),
        "output_paths": len(paths),
        "colors": len(bitmaps),
    }


def optimize_svg(
    input_path: str,
    output_path: str,
    epsilon: float = 1.0,
    *,
    max_colors: int = 0,
    anchor_min_dist: float = 0.0,
    snap_gray: bool = False,
    gray_tol: int = 0,
    gray_threshold: float = 128.0,
    skip_background: bool = False,
    min_area: float = 0.0,
    use_ocr: bool = False,
    ocr_languages: List[str] | None = None,
    ocr_min_confidence: float = 0.5,
) -> dict:
    """
    rect群SVGをポリゴンpath SVGに最適化

    OpenCVが利用可能な場合は高速版を使用、
    そうでない場合は純Python版にフォールバック。

    Args:
        input_path: 入力SVGパス（rect群）
        output_path: 出力SVGパス（path群）
        epsilon: 輪郭簡略化の許容誤差
        use_ocr: OCRでテキスト検出を行う
        ocr_languages: OCR言語リスト（デフォルト: ["ja", "en"]）
        ocr_min_confidence: OCR最小信頼度

    Returns:
        {"input_rects": N, "output_paths": M, "colors": K, "text_regions": L}
    """
    if _OPENCV_AVAILABLE:
        print("OpenCV版を使用（高速）")
        return optimize_svg_opencv(
            input_path,
            output_path,
            epsilon,
            max_colors=max_colors,
            anchor_min_dist=anchor_min_dist,
            snap_gray=snap_gray,
            gray_tol=gray_tol,
            gray_threshold=gray_threshold,
            skip_background=skip_background,
            min_area=min_area,
            use_ocr=use_ocr,
            ocr_languages=ocr_languages,
            ocr_min_confidence=ocr_min_confidence,
        )
    else:
        print("警告: OpenCVが見つかりません。純Python版を使用（低速）")
        print("  高速化するには: pip install opencv-python numpy")
        if use_ocr:
            print("  注意: OCR機能はOpenCV版でのみ利用可能です")
        return optimize_svg_pure(input_path, output_path, epsilon)


def main() -> None:
    """CLIエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="SVGのrect群をポリゴンpathに最適化"
    )
    parser.add_argument("--input", required=True, help="入力SVGパス（rect群）")
    parser.add_argument(
        "--output",
        help="出力SVGパス（未指定時は入力ファイル名_optimized.svg）",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="輪郭簡略化の許容誤差（デフォルト1.0、大きいほど簡略化）",
    )

    # --- 品質/編集性を上げるためのオプション ---
    parser.add_argument(
        "--clean",
        action="store_true",
        help="スクリーンショット/図表向けの推奨設定をまとめて適用（背景スキップ + グレー二値化 + 小ノイズ除去）",
    )
    parser.add_argument(
        "--skip-background",
        action="store_true",
        help="最大面積の色（背景想定）をpath化せず、全体背景の<rect> 1枚にする",
    )
    parser.add_argument(
        "--snap-gray",
        action="store_true",
        help="(ほぼ)グレーを白/黒に二値化してアンチエイリアス由来の粗さを抑える",
    )
    parser.add_argument(
        "--gray-tol",
        type=int,
        default=0,
        help="グレー判定の許容差（例: 3）  ※snap-gray時のみ",
    )
    parser.add_argument(
        "--gray-threshold",
        type=float,
        default=128.0,
        help="グレー二値化の閾値（輝度<閾値なら黒、それ以外は白）※snap-gray時のみ",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=0,
        help="代表色（アンカー）へ寄せて色数を削減（0で無効）",
    )
    parser.add_argument(
        "--anchor-min-dist",
        type=float,
        default=0.0,
        help="アンカー同士が近い色の場合は追加しない距離（例: 25）",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="この面積未満の輪郭を捨てる（小さなゴミ除去）。例: 2、10",
    )

    # --- OCRオプション ---
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="OCRでテキストを検出し、<text>要素として出力（EasyOCRが必要）",
    )
    parser.add_argument(
        "--ocr-languages",
        type=str,
        default="ja,en",
        help="OCR言語（カンマ区切り、デフォルト: ja,en）",
    )
    parser.add_argument(
        "--ocr-min-confidence",
        type=float,
        default=0.5,
        help="OCR最小信頼度（デフォルト: 0.5）",
    )

    args = parser.parse_args()

    # 出力パスのデフォルト設定
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.with_stem(input_path.stem + "_optimized"))

    # cleanプリセット（デフォルト値が未変更のときだけ上書き）
    skip_background = args.skip_background or args.clean
    snap_gray = args.snap_gray or args.clean
    gray_tol = args.gray_tol if (args.gray_tol != 0) else (3 if args.clean else 0)
    min_area = args.min_area if (args.min_area != 0.0) else (2.0 if args.clean else 0.0)

    # OCR言語をリストに変換
    ocr_languages = [lang.strip() for lang in args.ocr_languages.split(",")]

    result = optimize_svg(
        args.input,
        output_path,
        args.epsilon,
        max_colors=args.max_colors,
        anchor_min_dist=args.anchor_min_dist,
        snap_gray=snap_gray,
        gray_tol=gray_tol,
        gray_threshold=args.gray_threshold,
        skip_background=skip_background,
        min_area=min_area,
        use_ocr=args.ocr,
        ocr_languages=ocr_languages,
        ocr_min_confidence=args.ocr_min_confidence,
    )
    print(f"\n結果:")
    print(f"  入力rect数: {result['input_rects']}")
    print(f"  出力path数: {result['output_paths']}")
    print(f"  色数: {result['colors']}")
    if result.get("text_regions", 0) > 0:
        print(f"  テキスト領域数: {result['text_regions']}")
    print(f"  削減率: {(1 - result['output_paths'] / result['input_rects']) * 100:.1f}%")


if __name__ == "__main__":
    main()

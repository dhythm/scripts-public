"""SVGのrect群を解析してポリゴンpathに最適化するツール。

rect群で構成されたSVGを入力として、同色のrectをグループ化し、
連結成分の輪郭を抽出してポリゴンpathとして出力する。

使い方:
    python -m image_to_svg.svg_optimizer --input rects.svg --output optimized.svg
    python -m image_to_svg.svg_optimizer --input rects.svg --output optimized.svg --epsilon 2.0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET


@dataclass
class Rect:
    """SVGのrect要素を表すデータクラス"""

    x: int
    y: int
    width: int
    height: int
    color: str  # "rgb(r,g,b)" 形式


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

    # SVGサイズ取得
    width = int(float(root.get("width", 0)))
    height = int(float(root.get("height", 0)))

    rects = []
    for elem in root.iter():
        if elem.tag.endswith("rect"):
            x = int(float(elem.get("x", 0)))
            y = int(float(elem.get("y", 0)))
            w = int(float(elem.get("width", 0)))
            h = int(float(elem.get("height", 0)))
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


def optimize_svg(
    input_path: str,
    output_path: str,
    epsilon: float = 1.0,
) -> dict:
    """
    rect群SVGをポリゴンpath SVGに最適化

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
    print("ビットマップに変換中...")
    bitmaps = rects_to_bitmap(width, height, rects)
    print(f"  色数: {len(bitmaps)}")

    # 3. 各色の連結成分を検出してpath化
    print("輪郭を抽出中...")
    paths = []
    for color, bitmap in bitmaps.items():
        components = find_connected_components(bitmap)
        for contour in components:
            simplified = simplify_contour(contour, epsilon)
            path_str = contour_to_svg_path(simplified, color)
            if path_str:
                paths.append(path_str)

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

    args = parser.parse_args()

    # 出力パスのデフォルト設定
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.with_stem(input_path.stem + "_optimized"))

    result = optimize_svg(args.input, output_path, args.epsilon)
    print(f"\n結果:")
    print(f"  入力rect数: {result['input_rects']}")
    print(f"  出力path数: {result['output_paths']}")
    print(f"  色数: {result['colors']}")
    print(f"  削減率: {(1 - result['output_paths'] / result['input_rects']) * 100:.1f}%")


if __name__ == "__main__":
    main()

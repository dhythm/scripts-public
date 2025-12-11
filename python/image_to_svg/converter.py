"""画像からSVGへの変換処理 - Step by Step実装"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import svgwrite
from PIL import Image


def denoise_image(input_path: str, output_path: str) -> None:
    """
    JPEG/PNG画像のノイズを除去してPNGとして保存

    Args:
        input_path: 入力画像パス
        output_path: 出力PNGパス
    """
    # 画像読み込み
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"画像を読み込めません: {input_path}")

    # 非局所平均法でノイズ除去（カラー）
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        10,  # h
        10,  # hColor
        7,   # templateWindowSize
        21,  # searchWindowSize
    )

    # OpenCV BGR → RGB
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    # Pillow Imageとして保存
    pil_img = Image.fromarray(denoised_rgb)
    pil_img.save(output_path, "PNG")


def remove_antialiasing(input_path: str, output_path: str, threshold: int = 30) -> None:
    """
    アンチエイリアシングを除去（中間色を最も近い主要色に置換）

    Args:
        input_path: 入力画像パス
        output_path: 出力PNGパス
        threshold: 色差の閾値（これ以下の差は同じ色とみなす）
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"画像を読み込めません: {input_path}")

    # Step 1: メディアンフィルタで軽くぼかす（塩胡椒ノイズ除去）
    blurred = cv2.medianBlur(img, 3)

    # Step 2: バイラテラルフィルタ（エッジ保持しながら平滑化）
    # d: ピクセル近傍の直径
    # sigmaColor: 色空間でのフィルタシグマ
    # sigmaSpace: 座標空間でのフィルタシグマ
    bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)

    # Step 3: 色の量子化（各チャンネルを32段階に）
    quantize_level = 8  # 256/8 = 32段階
    quantized = (bilateral // quantize_level) * quantize_level + quantize_level // 2

    # BGR → RGB → 保存
    rgb_image = cv2.cvtColor(quantized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    pil_img.save(output_path, "PNG")


def count_unique_colors(image_path: str) -> int:
    """
    画像内の固有色数をカウント

    Args:
        image_path: 画像パス

    Returns:
        固有色の数
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"画像を読み込めません: {image_path}")

    pixels = img.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    return len(unique_colors)


def reduce_colors_kmeans(
    input_path: str,
    output_path: str,
    num_colors: int = 8,
) -> Tuple[int, list]:
    """
    K-meansで色数を削減してPNGとして保存

    Args:
        input_path: 入力画像パス
        output_path: 出力PNGパス
        num_colors: 目標色数

    Returns:
        (実際の色数, 使用された色のRGBリスト)
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"画像を読み込めません: {input_path}")

    # LAB色空間に変換（知覚的に均一）
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # ピクセルデータを2D配列に変形
    pixels = lab_image.reshape(-1, 3).astype(np.float32)

    # K-meansクラスタリング
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        num_colors,
        None,
        criteria,
        10,  # attempts
        cv2.KMEANS_PP_CENTERS,
    )

    # LAB → BGR
    centers_lab = centers.reshape(1, -1, 3).astype(np.uint8)
    centers_bgr = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2BGR)
    centers_bgr = centers_bgr.reshape(-1, 3)

    # 量子化された画像を生成
    labels_flat = labels.flatten()
    quantized_pixels = centers_bgr[labels_flat]
    quantized_image = quantized_pixels.reshape(img.shape).astype(np.uint8)

    # 実際の色数と色リスト
    unique_colors = np.unique(quantized_pixels, axis=0)
    colors_rgb = [(int(c[2]), int(c[1]), int(c[0])) for c in unique_colors]

    # BGR → RGB → 保存
    rgb_image = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    pil_img.save(output_path, "PNG")

    return len(unique_colors), colors_rgb


def posterize_only(input_path: str, output_path: str, levels: int = 8) -> None:
    """
    画像をポスタリゼーションのみ実行（色の階調を減らす）

    Args:
        input_path: 入力画像パス
        output_path: 出力PNGパス
        levels: 量子化レベル（256/levelsの段階に）
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"画像を読み込めません: {input_path}")

    # 各チャンネルを量子化
    divisor = 256 // levels
    posterized = (img // divisor) * divisor + divisor // 2

    # BGR → RGB → 保存
    rgb_image = cv2.cvtColor(posterized.astype(np.uint8), cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    pil_img.save(output_path, "PNG")


def apply_mode_filter(input_path: str, output_path: str, kernel_size: int = 3) -> None:
    """
    モードフィルタを適用（各ピクセルを近傍の最頻色に置換）
    アンチエイリアシングの中間色を周囲の主要色に吸収させる

    Args:
        input_path: 入力画像パス
        output_path: 出力PNGパス
        kernel_size: カーネルサイズ（奇数）
    """
    from scipy import ndimage
    from collections import Counter

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"画像を読み込めません: {input_path}")

    h, w = img.shape[:2]
    pad = kernel_size // 2

    # パディング
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    result = np.zeros_like(img)

    # 各ピクセルについて近傍の最頻色を取得
    for y in range(h):
        for x in range(w):
            # 近傍領域を取得
            region = padded[y:y + kernel_size, x:x + kernel_size]
            # ピクセルをタプルに変換してカウント
            pixels = [tuple(p) for p in region.reshape(-1, 3)]
            # 最頻色を取得
            most_common = Counter(pixels).most_common(1)[0][0]
            result[y, x] = most_common

    # BGR → RGB → 保存
    rgb_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    pil_img.save(output_path, "PNG")


def extract_contours_with_holes(
    image: np.ndarray,
    color_bgr: Tuple[int, int, int],
    min_area: int = 5,
) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """
    特定の色の輪郭を穴も含めて抽出

    Args:
        image: BGR画像
        color_bgr: 対象の色（BGR）
        min_area: 最小面積

    Returns:
        [(外側輪郭, [穴の輪郭リスト]), ...] のリスト
    """
    # この色のマスクを作成
    color_array = np.array(color_bgr, dtype=np.uint8)
    mask = cv2.inRange(image, color_array, color_array)

    # 輪郭検出（階層構造も取得）
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,  # 2レベル階層（外側と穴）
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None or len(contours) == 0:
        return []

    hierarchy = hierarchy[0]  # shape: (n, 4) - [next, prev, child, parent]

    result = []

    # 外側の輪郭（parent == -1）を探す
    for i, contour in enumerate(contours):
        parent = hierarchy[i][3]

        # 親がいない = 外側の輪郭
        if parent == -1:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # この輪郭の子（穴）を収集
            holes = []
            child_idx = hierarchy[i][2]  # 最初の子
            while child_idx != -1:
                hole_contour = contours[child_idx]
                hole_area = cv2.contourArea(hole_contour)
                if hole_area >= min_area:
                    holes.append(hole_contour)
                child_idx = hierarchy[child_idx][0]  # 次の兄弟

            result.append((contour, holes))

    return result


def contour_to_svg_path_with_holes(
    outer: np.ndarray,
    holes: List[np.ndarray],
    epsilon_factor: float = 0.0005,
) -> str:
    """
    外側輪郭と穴をSVGパス文字列に変換（fill-rule: evenodd用）

    Args:
        outer: 外側の輪郭
        holes: 穴の輪郭リスト
        epsilon_factor: 輪郭近似の精度

    Returns:
        SVGパス文字列
    """
    def contour_to_path_d(contour: np.ndarray) -> str:
        if len(contour) == 0:
            return ""

        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        points = approx.reshape(-1, 2)
        if len(points) < 3:
            return ""

        parts = [f"M {points[0][0]} {points[0][1]}"]
        for point in points[1:]:
            parts.append(f"L {point[0]} {point[1]}")
        parts.append("Z")

        return " ".join(parts)

    # 外側の輪郭
    path_d = contour_to_path_d(outer)
    if not path_d:
        return ""

    # 穴を追加
    for hole in holes:
        hole_path = contour_to_path_d(hole)
        if hole_path:
            path_d += " " + hole_path

    return path_d


def detect_background_color(image: np.ndarray) -> Tuple[int, int, int]:
    """
    画像の背景色を検出（四隅の最頻色）

    Args:
        image: BGR画像

    Returns:
        背景色（BGR）
    """
    h, w = image.shape[:2]
    corner_size = max(5, min(20, h // 20, w // 20))

    corners = [
        image[0:corner_size, 0:corner_size],
        image[0:corner_size, w - corner_size:w],
        image[h - corner_size:h, 0:corner_size],
        image[h - corner_size:h, w - corner_size:w],
    ]

    all_pixels = []
    for corner in corners:
        pixels = corner.reshape(-1, 3)
        all_pixels.extend([tuple(p) for p in pixels])

    most_common = Counter(all_pixels).most_common(1)[0][0]
    return most_common


def create_svg_from_quantized(
    quantized_image: np.ndarray,
    output_path: str,
    min_area: int = 5,
) -> int:
    """
    量子化済み画像からSVGを生成
    穴（内側の輪郭）も正しく処理し、面積の大きいパスから描画

    Args:
        quantized_image: 量子化済みBGR画像
        output_path: 出力SVGパス
        min_area: 最小面積

    Returns:
        生成されたパス数
    """
    h, w = quantized_image.shape[:2]

    # 使用されている色を取得
    pixels = quantized_image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    # 背景色を検出
    bg_color = detect_background_color(quantized_image)

    # 全てのパスを収集（面積でソートするため）
    all_paths = []  # [(area, color_rgb, path_d), ...]

    # 各色について処理（背景色以外）
    for color_bgr in unique_colors:
        color_tuple = tuple(color_bgr)

        # 背景色はスキップ
        if color_tuple == bg_color:
            continue

        # この色の輪郭を抽出（穴も含む）
        contours_with_holes = extract_contours_with_holes(
            quantized_image,
            color_tuple,
            min_area,
        )

        color_rgb = f"rgb({color_bgr[2]},{color_bgr[1]},{color_bgr[0]})"

        for outer, holes in contours_with_holes:
            path_d = contour_to_svg_path_with_holes(outer, holes)
            if path_d:
                area = cv2.contourArea(outer)
                all_paths.append((area, color_rgb, path_d))

    # 面積の大きい順にソート（大きいものを先に描画）
    all_paths.sort(key=lambda x: x[0], reverse=True)

    # SVG作成
    dwg = svgwrite.Drawing(output_path, size=(w, h))

    # 背景を追加
    dwg.add(dwg.rect(
        insert=(0, 0),
        size=(w, h),
        fill=f"rgb({bg_color[2]},{bg_color[1]},{bg_color[0]})",
    ))

    # パスを追加（面積の大きい順、fill-rule: evenoddで穴を表現）
    for area, color_rgb, path_d in all_paths:
        dwg.add(dwg.path(
            d=path_d,
            fill=color_rgb,
            stroke="none",
            fill_rule="evenodd",
        ))

    dwg.save()
    return len(all_paths)


def process_step_by_step(
    input_path: str,
    output_dir: str,
    num_colors: int = 10,
) -> dict:
    """
    ステップバイステップで処理し、中間結果を保存

    Args:
        input_path: 入力画像パス
        output_dir: 出力ディレクトリ
        num_colors: 目標色数

    Returns:
        各ステップの情報を含む辞書
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_name = Path(input_path).stem
    results = {}

    # Step 0: 元画像の色数
    original_colors = count_unique_colors(input_path)
    results["original_colors"] = original_colors
    print(f"Step 0 - 元画像の色数: {original_colors}")

    # Step 1: K-meansで色数削減
    reduced_path = str(output_path / f"{input_name}_1_reduced.png")
    actual_colors, color_list = reduce_colors_kmeans(
        input_path,
        reduced_path,
        num_colors,
    )
    results["reduced_path"] = reduced_path
    results["reduced_colors"] = actual_colors
    results["color_list"] = color_list
    print(f"Step 1 - K-means後: {actual_colors}色")
    print(f"使用色: {color_list}")

    # Step 2: SVG生成
    svg_path = str(output_path / f"{input_name}_2_output.svg")
    quantized_img = cv2.imread(reduced_path)
    path_count = create_svg_from_quantized(quantized_img, svg_path)
    results["svg_path"] = svg_path
    results["path_count"] = path_count
    print(f"Step 2 - SVG生成: {path_count}パス")

    return results


# CLI用（後で整備）
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("使い方: python converter.py <入力画像> <出力ディレクトリ> [色数]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    num_colors = int(sys.argv[3]) if len(sys.argv) >= 4 else 8

    process_step_by_step(input_path, output_dir, num_colors)

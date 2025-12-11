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
    各クラスタは元画像に存在する最頻色で置き換える（新しい色を生成しない）

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

    # 元のBGRピクセルを保持
    original_pixels = img.reshape(-1, 3)

    # LAB色空間に変換（知覚的に均一）
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # ピクセルデータを2D配列に変形
    pixels_lab = lab_image.reshape(-1, 3).astype(np.float32)

    # K-meansクラスタリング
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(
        pixels_lab,
        num_colors,
        None,
        criteria,
        10,  # attempts
        cv2.KMEANS_PP_CENTERS,
    )

    labels_flat = labels.flatten()

    # 各クラスタの代表色を元画像の最頻色から選択
    representative_colors = np.zeros((num_colors, 3), dtype=np.uint8)
    for cluster_id in range(num_colors):
        mask = labels_flat == cluster_id
        if not np.any(mask):
            continue

        cluster_pixels = original_pixels[mask]
        # クラスタ内の最頻色を取得
        unique, counts = np.unique(cluster_pixels, axis=0, return_counts=True)
        most_frequent_color = unique[np.argmax(counts)]
        representative_colors[cluster_id] = most_frequent_color

    # 量子化された画像を生成（元画像の色のみを使用）
    quantized_pixels = representative_colors[labels_flat]
    quantized_image = quantized_pixels.reshape(img.shape).astype(np.uint8)

    # 実際の色数と色リスト
    unique_colors = np.unique(quantized_pixels, axis=0)
    colors_rgb = [(int(c[2]), int(c[1]), int(c[0])) for c in unique_colors]

    # BGR → RGB → 保存
    rgb_image = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    pil_img.save(output_path, "PNG")

    return len(unique_colors), colors_rgb


def reduce_colors_by_frequency(
    input_path: str,
    output_path: str,
    target_colors: int = 256,
    initial_threshold: float = 10.0,
    threshold_step: float = 5.0,
    max_iterations: int = 20,
    intermediate_dir: str | None = None,
) -> Tuple[int, list]:
    """
    効率的に近似色を最頻色に統合して色数を削減

    アルゴリズム:
    1. 全色をLAB色空間に変換
    2. 出現頻度順にソート
    3. 各色について、閾値以内の近似色を最頻色に統合
    4. 色数が目標以下になるまで閾値を上げて繰り返し

    Args:
        input_path: 入力画像パス
        output_path: 出力PNGパス
        target_colors: 目標色数
        initial_threshold: LAB色空間での初期色差閾値
        threshold_step: 閾値の増加幅
        max_iterations: 最大反復回数
        intermediate_dir: 中間結果を保存するディレクトリ（Noneの場合は保存しない）

    Returns:
        (実際の色数, 使用された色のRGBリスト)
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"画像を読み込めません: {input_path}")

    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3)

    # 中間結果保存用ディレクトリ
    if intermediate_dir:
        intermediate_path = Path(intermediate_dir)
        intermediate_path.mkdir(parents=True, exist_ok=True)

    # 全色とその出現回数・位置を取得
    unique_colors, inverse, counts = np.unique(
        pixels, axis=0, return_inverse=True, return_counts=True
    )
    print(f"  初期色数: {len(unique_colors)}")

    # BGR → LAB に一括変換
    unique_colors_lab = cv2.cvtColor(
        unique_colors.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    threshold = initial_threshold

    for iteration in range(max_iterations):
        n_colors = len(unique_colors)
        print(f"  Iteration {iteration + 1}: {n_colors}色 (閾値={threshold:.1f})")

        if n_colors <= target_colors:
            break

        # 出現頻度順にインデックスをソート
        sorted_indices = np.argsort(-counts)

        # マッピング配列（自分自身にマップ = 変更なし）
        mapping = np.arange(n_colors)
        processed = np.zeros(n_colors, dtype=bool)

        for idx in sorted_indices:
            if processed[idx]:
                continue

            # この色のLAB値
            color_lab = unique_colors_lab[idx]

            # 未処理の色との距離を計算
            unprocessed_mask = ~processed
            unprocessed_indices = np.where(unprocessed_mask)[0]

            if len(unprocessed_indices) == 0:
                break

            unprocessed_labs = unique_colors_lab[unprocessed_indices]
            distances = np.sqrt(np.sum((unprocessed_labs - color_lab) ** 2, axis=1))

            # 閾値以内の色を見つける
            similar_mask = distances <= threshold
            similar_indices = unprocessed_indices[similar_mask]

            # 類似色の中で最頻色を見つける
            similar_counts = counts[similar_indices]
            most_frequent_idx = similar_indices[np.argmax(similar_counts)]

            # マッピングを設定
            for sim_idx in similar_indices:
                mapping[sim_idx] = most_frequent_idx
                processed[sim_idx] = True

        # マッピングを適用して色を統合
        new_color_indices = mapping[inverse]
        new_pixels = unique_colors[new_color_indices]

        # 新しいユニーク色を取得
        new_unique_colors, new_inverse, new_counts = np.unique(
            new_pixels, axis=0, return_inverse=True, return_counts=True
        )

        if len(new_unique_colors) == n_colors:
            # 変化なし → 閾値を上げる
            threshold += threshold_step
            continue

        # 更新
        unique_colors = new_unique_colors
        inverse = new_inverse
        counts = new_counts

        # 中間結果を保存（256色以下になったら）
        if intermediate_dir and len(unique_colors) <= 256:
            intermediate_pixels = unique_colors[inverse]
            intermediate_image = intermediate_pixels.reshape(h, w, 3).astype(np.uint8)
            intermediate_rgb = cv2.cvtColor(intermediate_image, cv2.COLOR_BGR2RGB)
            intermediate_pil = Image.fromarray(intermediate_rgb)
            intermediate_pil.save(
                intermediate_path / f"iter_{iteration + 1:02d}_{len(unique_colors)}colors.png",
                "PNG",
            )

        # LABも再計算
        unique_colors_lab = cv2.cvtColor(
            unique_colors.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)

        if len(unique_colors) <= target_colors:
            break

        # 閾値を上げる
        threshold += threshold_step

    # 最終結果を画像に適用
    final_pixels = unique_colors[inverse]
    result_image = final_pixels.reshape(h, w, 3).astype(np.uint8)

    # 色リスト（RGB）
    colors_rgb = [(int(c[2]), int(c[1]), int(c[0])) for c in unique_colors]

    # BGR → RGB → 保存
    rgb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    pil_img.save(output_path, "PNG")

    print(f"  最終: {len(unique_colors)}色")
    return len(unique_colors), colors_rgb


def sharpen_boundaries_morphology(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    形態学的処理で境界を明確化（滑らかさを保持）

    各色のマスクに対してクロージング（膨張→収縮）を適用し、
    小さな穴や境界のギザギザを滑らかにする

    Args:
        image: BGR画像（色数が少ない前提）
        kernel_size: モルフォロジーカーネルサイズ

    Returns:
        境界が明確化された画像
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)

    # 使用されている色を取得
    unique_colors = np.unique(pixels, axis=0)
    print(f"    処理対象: {len(unique_colors)}色")

    # 各色のマスクを作成し、モルフォロジー処理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 結果画像（優先度マップも作成）
    result = image.copy()
    processed_mask = np.zeros((h, w), dtype=bool)

    # 面積の大きい色から処理（背景→前景の順）
    color_areas = []
    for color in unique_colors:
        mask = np.all(image == color, axis=2)
        area = np.sum(mask)
        color_areas.append((color, area))

    color_areas.sort(key=lambda x: -x[1])  # 面積降順

    for color, area in color_areas:
        # この色のマスク
        mask = np.all(image == color, axis=2).astype(np.uint8) * 255

        # クロージング（膨張→収縮）で小さな穴を埋め、境界を滑らかに
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # オープニング（収縮→膨張）で小さなノイズを除去
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # まだ処理されていない領域にのみ適用
        new_mask = (opened > 0) & ~processed_mask
        result[new_mask] = color
        processed_mask |= new_mask

    return result


def reduce_colors_with_sharp_boundaries(
    input_path: str,
    output_path: str,
    target_colors: int = 10,
    initial_threshold: float = 10.0,
    threshold_step: float = 5.0,
    max_iterations: int = 20,
    boundary_iterations: int = 2,
) -> Tuple[int, list]:
    """
    2段階処理で色数削減と境界明確化を行う

    1. グローバル色統合（出現頻度ベース）
    2. 境界明確化（隣接ピクセルの最頻色に置換）

    Args:
        input_path: 入力画像パス
        output_path: 出力PNGパス
        target_colors: 目標色数
        initial_threshold: 初期色差閾値
        threshold_step: 閾値増加幅
        max_iterations: 色統合の最大反復回数
        boundary_iterations: 境界明確化の反復回数

    Returns:
        (実際の色数, 使用された色のRGBリスト)
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"画像を読み込めません: {input_path}")

    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3)

    # === Step 1: グローバル色統合 ===
    print("  Step 1: グローバル色統合...")

    unique_colors, inverse, counts = np.unique(
        pixels, axis=0, return_inverse=True, return_counts=True
    )
    print(f"    初期色数: {len(unique_colors)}")

    unique_colors_lab = cv2.cvtColor(
        unique_colors.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    threshold = initial_threshold

    for iteration in range(max_iterations):
        n_colors = len(unique_colors)
        print(f"    Iteration {iteration + 1}: {n_colors}色 (閾値={threshold:.1f})")

        if n_colors <= target_colors:
            break

        sorted_indices = np.argsort(-counts)
        mapping = np.arange(n_colors)
        processed = np.zeros(n_colors, dtype=bool)

        for idx in sorted_indices:
            if processed[idx]:
                continue

            color_lab = unique_colors_lab[idx]
            unprocessed_mask = ~processed
            unprocessed_indices = np.where(unprocessed_mask)[0]

            if len(unprocessed_indices) == 0:
                break

            unprocessed_labs = unique_colors_lab[unprocessed_indices]
            distances = np.sqrt(np.sum((unprocessed_labs - color_lab) ** 2, axis=1))

            similar_mask = distances <= threshold
            similar_indices = unprocessed_indices[similar_mask]

            similar_counts = counts[similar_indices]
            most_frequent_idx = similar_indices[np.argmax(similar_counts)]

            for sim_idx in similar_indices:
                mapping[sim_idx] = most_frequent_idx
                processed[sim_idx] = True

        new_color_indices = mapping[inverse]
        new_pixels = unique_colors[new_color_indices]

        new_unique_colors, new_inverse, new_counts = np.unique(
            new_pixels, axis=0, return_inverse=True, return_counts=True
        )

        if len(new_unique_colors) == n_colors:
            threshold += threshold_step
            continue

        unique_colors = new_unique_colors
        inverse = new_inverse
        counts = new_counts

        unique_colors_lab = cv2.cvtColor(
            unique_colors.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)

        if len(unique_colors) <= target_colors:
            break

        threshold += threshold_step

    # 中間画像を生成
    intermediate_pixels = unique_colors[inverse]
    intermediate_image = intermediate_pixels.reshape(h, w, 3).astype(np.uint8)
    print(f"    色統合後: {len(unique_colors)}色")

    # === Step 2: 境界明確化（形態学的処理）===
    print("  Step 2: 境界明確化...")
    result_image = sharpen_boundaries_morphology(intermediate_image, kernel_size=3)

    # 最終色数を確認
    final_pixels = result_image.reshape(-1, 3)
    final_unique_colors = np.unique(final_pixels, axis=0)
    colors_rgb = [(int(c[2]), int(c[1]), int(c[0])) for c in final_unique_colors]

    # 保存
    rgb_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    pil_img.save(output_path, "PNG")

    print(f"  最終: {len(final_unique_colors)}色")
    return len(final_unique_colors), colors_rgb


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
    num_colors: int = 256,
    use_iterative: bool = True,
    save_intermediate: bool = False,
) -> dict:
    """
    ステップバイステップで処理し、中間結果を保存

    Args:
        input_path: 入力画像パス
        output_dir: 出力ディレクトリ
        num_colors: 目標色数
        use_iterative: True=反復的色統合、False=K-means
        save_intermediate: True=各イテレーションの中間結果をPNGで保存

    Returns:
        各ステップの情報を含む辞書
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_name = Path(input_path).stem
    results = {}

    # 中間結果保存用ディレクトリ
    intermediate_dir = None
    if save_intermediate:
        intermediate_dir = str(output_path / f"{input_name}_intermediate")

    # Step 0: 元画像の色数
    original_colors = count_unique_colors(input_path)
    results["original_colors"] = original_colors
    print(f"Step 0 - 元画像の色数: {original_colors}")

    # Step 1: 色数削減
    reduced_path = str(output_path / f"{input_name}_1_reduced.png")

    if use_iterative:
        print("Step 1 - 反復的色統合...")
        actual_colors, color_list = reduce_colors_by_frequency(
            input_path,
            reduced_path,
            target_colors=num_colors,
            intermediate_dir=intermediate_dir,
        )
    else:
        print("Step 1 - K-means...")
        actual_colors, color_list = reduce_colors_kmeans(
            input_path,
            reduced_path,
            num_colors,
        )

    results["reduced_path"] = reduced_path
    results["reduced_colors"] = actual_colors
    results["color_list"] = color_list
    print(f"Step 1 完了: {actual_colors}色")
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

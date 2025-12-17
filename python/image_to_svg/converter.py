"""画像からSVGへの変換処理 - Step by Step実装"""
from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import svgwrite
import vtracer
from PIL import Image


def _log(message: str, start_time: float | None = None) -> None:
    """タイムスタンプ付きログ出力"""
    elapsed = ""
    if start_time is not None:
        elapsed = f" ({time.time() - start_time:.2f}s)"
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {message}")


def upscale_image(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    画像を拡大（最近傍補間で色を保持）

    Args:
        image: BGR画像
        scale: 拡大倍率

    Returns:
        拡大された画像
    """
    h, w = image.shape[:2]
    new_h, new_w = h * scale, w * scale
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def downscale_image(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    画像を縮小（最近傍補間で色を保持）

    Args:
        image: BGR画像
        scale: 縮小倍率

    Returns:
        縮小された画像
    """
    h, w = image.shape[:2]
    new_h, new_w = h // scale, w // scale
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


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


def reduce_colors_region_based(
    image: np.ndarray,
    target_colors: int = 15,
    min_region_area: int = 50,
) -> Tuple[np.ndarray, int, list]:
    """
    領域ベースの色統合で色数を削減

    アルゴリズム:
    1. 各色の連結成分（領域）を検出
    2. 小さい領域を隣接する最大領域の色に吸収
    3. 残った領域をLAB色空間でクラスタリング
    4. 各クラスタの代表色は元画像の最頻色から選択

    Args:
        image: BGR画像（既に色数が削減されている前提）
        target_colors: 目標色数
        min_region_area: 最小領域面積（これ以下は吸収）

    Returns:
        (処理後の画像, 実際の色数, 使用された色のRGBリスト)
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)

    # 使用されている色を取得
    unique_colors = np.unique(pixels, axis=0)
    print(f"    入力色数: {len(unique_colors)}")

    # === Step 1: 各色の連結成分を検出し、小さい領域を吸収 ===
    print("    Step 1: 小領域の吸収...")

    # 結果画像を初期化
    result = image.copy()

    # 各色について処理
    for color in unique_colors:
        # この色のマスク
        color_mask = np.all(image == color, axis=2).astype(np.uint8)

        # 連結成分ラベリング
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            color_mask, connectivity=8
        )

        # 各領域について処理（ラベル0は背景なのでスキップ）
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area < min_region_area:
                # 小さい領域 → 隣接する最大領域の色に置換
                region_mask = labels == label_id

                # この領域の境界ピクセルを見つける
                kernel = np.ones((3, 3), dtype=np.uint8)
                dilated = cv2.dilate(region_mask.astype(np.uint8), kernel)
                boundary = dilated.astype(bool) & ~region_mask

                # 境界の色をカウント
                if np.any(boundary):
                    boundary_colors = image[boundary]
                    unique_boundary, counts = np.unique(
                        boundary_colors, axis=0, return_counts=True
                    )
                    # 最頻色で置換
                    dominant_color = unique_boundary[np.argmax(counts)]
                    result[region_mask] = dominant_color

    # === Step 2: 残った色をクラスタリング ===
    print("    Step 2: 色のクラスタリング...")

    result_pixels = result.reshape(-1, 3)
    unique_colors_after = np.unique(result_pixels, axis=0)
    print(f"    吸収後の色数: {len(unique_colors_after)}")

    if len(unique_colors_after) <= target_colors:
        # 既に目標以下なら終了
        colors_rgb = [(int(c[2]), int(c[1]), int(c[0])) for c in unique_colors_after]
        return result, len(unique_colors_after), colors_rgb

    # LAB色空間でK-meansクラスタリング
    unique_colors_lab = cv2.cvtColor(
        unique_colors_after.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels_km, _ = cv2.kmeans(
        unique_colors_lab,
        target_colors,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    labels_km = labels_km.flatten()

    # 各クラスタの代表色を決定（そのクラスタ内で最も面積が大きい色）
    representative_colors = np.zeros((target_colors, 3), dtype=np.uint8)
    for cluster_id in range(target_colors):
        cluster_mask = labels_km == cluster_id
        if not np.any(cluster_mask):
            continue

        cluster_colors = unique_colors_after[cluster_mask]

        # 各色の面積を計算
        max_area = 0
        best_color = cluster_colors[0]
        for color in cluster_colors:
            area = np.sum(np.all(result == color, axis=2))
            if area > max_area:
                max_area = area
                best_color = color

        representative_colors[cluster_id] = best_color

    # 色のマッピングを作成
    color_to_cluster = {}
    for i, color in enumerate(unique_colors_after):
        cluster_id = labels_km[i]
        color_to_cluster[tuple(color)] = representative_colors[cluster_id]

    # 画像を更新
    final_result = np.zeros_like(result)
    for y in range(h):
        for x in range(w):
            old_color = tuple(result[y, x])
            if old_color in color_to_cluster:
                final_result[y, x] = color_to_cluster[old_color]
            else:
                final_result[y, x] = result[y, x]

    # 最終色数
    final_pixels = final_result.reshape(-1, 3)
    final_unique = np.unique(final_pixels, axis=0)
    colors_rgb = [(int(c[2]), int(c[1]), int(c[0])) for c in final_unique]

    print(f"    最終色数: {len(final_unique)}")
    return final_result, len(final_unique), colors_rgb


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
    original_size: Tuple[int, int] | None = None,
) -> int:
    """
    量子化済み画像からSVGを生成
    穴（内側の輪郭）も正しく処理し、面積の大きいパスから描画

    Args:
        quantized_image: 量子化済みBGR画像
        output_path: 出力SVGパス
        min_area: 最小面積
        original_size: 元のサイズ (width, height)、指定時はviewBoxを使用

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

    # SVG作成（original_sizeが指定されていればviewBoxを使用）
    if original_size:
        display_w, display_h = original_size
        dwg = svgwrite.Drawing(
            output_path,
            size=(display_w, display_h),
            viewBox=f"0 0 {w} {h}",
        )
    else:
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


def merge_similar_colors(
    image: np.ndarray,
    threshold: float = 5.0,
    min_pixel_ratio: float = 0.001,
) -> np.ndarray:
    """
    類似色をマージして色数を削減

    K-means後のアンチエイリアス由来の微小な色差を統合する
    LAB色空間で閾値以内の色を最頻色にマージ

    Args:
        image: BGR画像
        threshold: LAB色空間での色差閾値（デフォルト5.0、低いほど保守的）
        min_pixel_ratio: マージ対象の最小ピクセル比率（これ以上の色は保護）

    Returns:
        類似色がマージされた画像
    """
    h, w = image.shape[:2]
    total_pixels = h * w
    pixels = image.reshape(-1, 3)

    # ユニーク色と出現回数を取得
    unique_colors, inverse, counts = np.unique(
        pixels, axis=0, return_inverse=True, return_counts=True
    )

    if len(unique_colors) <= 1:
        return image

    # 最小ピクセル数（これ以上の色は保護）
    min_pixels = int(total_pixels * min_pixel_ratio)

    # BGR → LAB
    unique_colors_lab = cv2.cvtColor(
        unique_colors.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    # 出現頻度順にソート
    sorted_indices = np.argsort(-counts)

    # マッピング配列
    mapping = np.arange(len(unique_colors))
    processed = np.zeros(len(unique_colors), dtype=bool)

    for idx in sorted_indices:
        if processed[idx]:
            continue

        color_lab = unique_colors_lab[idx]

        # この色を処理済みとしてマーク
        processed[idx] = True
        mapping[idx] = idx

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

        # 類似色をマージ（ただし、十分なピクセル数を持つ色は保護）
        for sim_idx in similar_indices:
            # この色が十分なピクセル数を持つ場合は保護（マージしない）
            if counts[sim_idx] >= min_pixels:
                continue
            mapping[sim_idx] = idx
            processed[sim_idx] = True

    # マッピングを適用
    new_color_indices = mapping[inverse]
    new_pixels = unique_colors[new_color_indices]

    return new_pixels.reshape(h, w, 3).astype(np.uint8)


def vectorize_with_vtracer(
    input_path: str,
    output_path: str,
    mode: str = "spline",
    filter_speckle: int = 16,
    color_precision: int = 8,
    corner_threshold: int = 60,
    length_threshold: float = 4.0,
    splice_threshold: int = 45,
    original_size: tuple[int, int] | None = None,
) -> int:
    """
    vtracerを使用して画像をSVGに変換

    Args:
        input_path: 入力画像パス
        output_path: 出力SVGパス
        mode: 'spline'（ベジェ曲線）or 'polygon'（多角形）or 'none'
        filter_speckle: ノイズ除去閾値（小さい領域を除去）
        color_precision: 色精度（1-8、高いほど色数が多い）
        corner_threshold: コーナー検出角度（度）
        length_threshold: 長さ閾値
        splice_threshold: 接合閾値（度）
        original_size: 元のサイズ(width, height)、指定時はviewBoxを調整

    Returns:
        生成されたパス数
    """
    # 一時ファイルに出力してから読み込む
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        tmp_path = tmp.name

    vtracer.convert_image_to_svg_py(
        input_path,
        tmp_path,
        colormode="color",
        hierarchical="stacked",
        mode=mode,
        filter_speckle=filter_speckle,
        color_precision=color_precision,
        corner_threshold=corner_threshold,
        length_threshold=length_threshold,
        splice_threshold=splice_threshold,
    )

    svg_str = Path(tmp_path).read_text()
    Path(tmp_path).unlink()

    # viewBoxの調整が必要な場合
    if original_size is not None:
        orig_w, orig_h = original_size
        # vtracerが生成したSVGのサイズを元のサイズに変更
        # width="xxx" height="yyy" を width="orig_w" height="orig_h" に置換
        import re

        # viewBoxは維持し、表示サイズのみ変更
        svg_str = re.sub(
            r'width="(\d+)" height="(\d+)"',
            f'width="{orig_w}" height="{orig_h}"',
            svg_str,
            count=1,
        )

    Path(output_path).write_text(svg_str)

    # パス数をカウント
    path_count = svg_str.count("<path")
    return path_count


def process_step_by_step(
    input_path: str,
    output_dir: str,
    num_colors: int = 256,
    use_iterative: bool = True,
    save_intermediate: bool = False,
    final_colors: int = 256,
    min_region_area: int = 50,
    upscale_factor: int = 2,
    use_vtracer: bool = True,
    vtracer_mode: str = "spline",
    skip_region_absorption: bool = True,
    color_merge_threshold: float = 5.0,
    filter_speckle: int = 16,
) -> dict:
    """
    ステップバイステップで処理し、中間結果を保存

    Args:
        input_path: 入力画像パス
        output_dir: 出力ディレクトリ
        num_colors: 第1段階の目標色数（デフォルト256）
        use_iterative: True=反復的色統合、False=K-means
        save_intermediate: True=各イテレーションの中間結果をPNGで保存
        final_colors: 第2段階（領域ベース統合）の最終目標色数
        min_region_area: 最小領域面積（これ以下は隣接領域に吸収）
        upscale_factor: 拡大倍率（1=拡大なし、2=2倍、4=4倍）
        use_vtracer: True=vtracerでSVG生成（ベジェ曲線）、False=従来の直線パス
        vtracer_mode: 'spline'（ベジェ曲線）or 'polygon'（多角形）
        skip_region_absorption: True=小領域吸収をスキップ（デフォルト、高速）
        color_merge_threshold: 類似色マージの閾値（LAB色空間での距離、デフォルト5.0）
        filter_speckle: vtracerのノイズ除去閾値（デフォルト16、大きいほど小領域を除去）

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
        Path(intermediate_dir).mkdir(parents=True, exist_ok=True)

    # Step 0: 元画像の読み込みと情報取得
    total_start = time.time()
    step_start = time.time()
    original_img = cv2.imread(input_path)
    if original_img is None:
        raise ValueError(f"画像を読み込めません: {input_path}")

    original_h, original_w = original_img.shape[:2]
    original_colors = count_unique_colors(input_path)
    results["original_colors"] = original_colors
    results["original_size"] = (original_w, original_h)
    _log(f"Step 0 - 元画像: {original_w}x{original_h}, {original_colors}色", step_start)

    # Step 1: 画像を拡大（最近傍補間で色を保持）
    step_start = time.time()
    if upscale_factor > 1:
        _log(f"Step 1 - 画像を{upscale_factor}倍に拡大...")
        upscaled_img = upscale_image(original_img, upscale_factor)
        upscaled_h, upscaled_w = upscaled_img.shape[:2]
        _log(f"  拡大後サイズ: {upscaled_w}x{upscaled_h}", step_start)

        # 拡大画像を一時保存
        upscaled_path = str(output_path / f"{input_name}_1_upscaled.png")
        upscaled_rgb = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(upscaled_rgb).save(upscaled_path, "PNG")
        results["upscaled_path"] = upscaled_path

        # 小領域の閾値も拡大に合わせて調整
        adjusted_min_region_area = min_region_area * (upscale_factor ** 2)
    else:
        upscaled_img = original_img
        adjusted_min_region_area = min_region_area

    # Step 2: 色数削減（拡大画像に対して）
    step_start = time.time()
    _log(f"Step 2 - 色数削減（目標: {final_colors}色）...")

    # K-meansで直接目標色数に削減（元画像の色を保持）
    h, w = upscaled_img.shape[:2]
    pixels = upscaled_img.reshape(-1, 3)

    # LAB色空間でK-means
    lab_img = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2LAB)
    pixels_lab = lab_img.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(
        pixels_lab,
        final_colors,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    labels_flat = labels.flatten()

    # 各クラスタの代表色を元画像の最頻色から選択
    representative_colors = np.zeros((final_colors, 3), dtype=np.uint8)
    for cluster_id in range(final_colors):
        mask = labels_flat == cluster_id
        if not np.any(mask):
            continue
        cluster_pixels = pixels[mask]
        unique, counts = np.unique(cluster_pixels, axis=0, return_counts=True)
        representative_colors[cluster_id] = unique[np.argmax(counts)]

    # 量子化画像を生成
    quantized_pixels = representative_colors[labels_flat]
    quantized_img = quantized_pixels.reshape(h, w, 3).astype(np.uint8)

    # 量子化結果を保存（マージ前）
    quantized_path = str(output_path / f"{input_name}_2_quantized.png")
    quantized_rgb = cv2.cvtColor(quantized_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(quantized_rgb).save(quantized_path, "PNG")
    results["quantized_path"] = quantized_path

    unique_colors = np.unique(quantized_pixels, axis=0)
    _log(f"  K-means後の色数: {len(unique_colors)}", step_start)

    # Step 2.5: 類似色のマージ（アンチエイリアス由来の微小な色差を統合）
    step_start = time.time()
    _log(f"Step 2.5 - 類似色のマージ（閾値: {color_merge_threshold}）...")
    quantized_img = merge_similar_colors(quantized_img, threshold=color_merge_threshold)

    # マージ結果を保存
    merged_path = str(output_path / f"{input_name}_2b_merged.png")
    merged_rgb = cv2.cvtColor(quantized_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(merged_rgb).save(merged_path, "PNG")
    results["merged_path"] = merged_path

    merged_pixels = quantized_img.reshape(-1, 3)
    unique_colors = np.unique(merged_pixels, axis=0)
    _log(f"  マージ後の色数: {len(unique_colors)}", step_start)

    # Step 3: 小領域の吸収
    step_start = time.time()
    if skip_region_absorption:
        _log("Step 3 - 小領域の吸収をスキップ（vtracerのfilter_speckleで代替）")
        cleaned_img = quantized_img
    else:
        _log(f"Step 3 - 小領域の吸収（閾値: {adjusted_min_region_area}px）...")
        cleaned_img = quantized_img.copy()

        for color in unique_colors:
            color_mask = np.all(quantized_img == color, axis=2).astype(np.uint8)
            num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
                color_mask, connectivity=8
            )

            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < adjusted_min_region_area:
                    region_mask = labels_cc == label_id
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    dilated = cv2.dilate(region_mask.astype(np.uint8), kernel)
                    boundary = dilated.astype(bool) & ~region_mask

                    if np.any(boundary):
                        boundary_colors = quantized_img[boundary]
                        unique_boundary, counts = np.unique(
                            boundary_colors, axis=0, return_counts=True
                        )
                        dominant_color = unique_boundary[np.argmax(counts)]
                        cleaned_img[region_mask] = dominant_color

    # クリーニング結果を保存
    cleaned_path = str(output_path / f"{input_name}_3_cleaned.png")
    cleaned_rgb = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(cleaned_rgb).save(cleaned_path, "PNG")
    results["cleaned_path"] = cleaned_path

    final_pixels = cleaned_img.reshape(-1, 3)
    final_unique = np.unique(final_pixels, axis=0)
    colors_rgb = [(int(c[2]), int(c[1]), int(c[0])) for c in final_unique]
    _log(f"  処理後の色数: {len(final_unique)}", step_start)
    _log(f"  使用色: {colors_rgb}")

    results["final_colors"] = len(final_unique)
    results["color_list"] = colors_rgb

    # Step 4: SVG生成（高解像度画像から、viewBoxで元サイズに）
    step_start = time.time()
    method_name = "vtracer" if use_vtracer else "legacy"
    _log(f"Step 4 - SVG生成（{method_name}）...")
    svg_path = str(output_path / f"{input_name}_4_output.svg")

    if use_vtracer:
        # vtracerでSVG生成（ベジェ曲線対応）
        path_count = vectorize_with_vtracer(
            cleaned_path,
            svg_path,
            mode=vtracer_mode,
            filter_speckle=filter_speckle,
            color_precision=8,  # 前処理済みなので最大精度
            original_size=(original_w, original_h) if upscale_factor > 1 else None,
        )
    else:
        # 従来の直線パスSVG生成
        path_count = create_svg_from_quantized(
            cleaned_img,
            svg_path,
            original_size=(original_w, original_h) if upscale_factor > 1 else None,
        )
    results["svg_path"] = svg_path
    results["path_count"] = path_count
    results["method"] = method_name
    _log(f"  SVG生成: {path_count}パス", step_start)
    _log(f"完了（総処理時間: {time.time() - total_start:.2f}s）")

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

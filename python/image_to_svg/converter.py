"""画像からSVGへの変換処理"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple, TypedDict

import cv2
import numpy as np
import svgwrite


# 調整可能なパラメータ
NUM_COLORS = 6  # K-meansのクラスタ数（人の目で見える色数に近く）
EPSILON_FACTOR = 0.001  # 輪郭近似の精度（周長に対する比率）
MIN_AREA_RATIO = 0.00005  # 画像面積の0.005%
MIN_AREA_ABSOLUTE = 20  # 最小面積絶対値（px）
KERNEL_SIZE = 1  # モルフォロジーカーネルサイズ（最小 - 小さな領域を保持）
KMEANS_ATTEMPTS = 10  # K-means試行回数
UPSCALE_FACTOR = 2  # 画像拡大倍率（輪郭抽出の精度向上用）

# デノイズパラメータ
DENOISE_H = 10  # Non-local means denoising strength
DENOISE_TEMPLATE_WINDOW = 7  # テンプレートウィンドウサイズ
DENOISE_SEARCH_WINDOW = 21  # 検索ウィンドウサイズ
MEDIAN_BLUR_SIZE = 3  # メディアンフィルタサイズ（奇数）
COLOR_QUANTIZE_LEVELS = 8  # 色の量子化レベル（256/8=32段階に量子化）


class ContourPath(TypedDict):
    """輪郭パスを表す型"""
    points: np.ndarray
    color_rgb: Tuple[int, int, int]
    is_hole: bool


def quantize_colors(
    image: np.ndarray,
    num_colors: int = NUM_COLORS,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    画像の色をK-meansで量子化する（LAB色空間を使用）

    Args:
        image: BGR画像（OpenCV形式）
        num_colors: 量子化後の最大色数

    Returns:
        量子化された画像（BGR）, 色のリスト（RGB形式）
    """
    # まず画像内の固有色を取得
    pixels_bgr = image.reshape(-1, 3)
    unique_bgr = np.unique(pixels_bgr, axis=0)

    # 固有色が指定数以下ならそのまま使用
    if len(unique_bgr) <= num_colors:
        # 各ピクセルを最も近い固有色に割り当て（既に固有なのでそのまま）
        colors_rgb: List[Tuple[int, int, int]] = []
        for bgr in unique_bgr:
            colors_rgb.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))
        return image.copy(), colors_rgb

    # LAB色空間に変換（知覚的に均一）
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # ピクセルデータを2D配列に変形
    pixels = lab_image.reshape(-1, 3).astype(np.float32)

    # K-means クラスタリング
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        num_colors,
        None,
        criteria,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS,
    )

    # LAB→BGRに戻す
    centers_lab = centers.reshape(1, -1, 3).astype(np.uint8)
    centers_bgr = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2BGR)
    centers_bgr = centers_bgr.reshape(-1, 3)

    # 量子化された画像を生成
    labels_flat = labels.flatten()
    quantized_pixels = centers_bgr[labels_flat]
    quantized_image = quantized_pixels.reshape(image.shape).astype(np.uint8)

    # 実際に使用された色のみを取得（重複を除く）
    unique_quantized = np.unique(quantized_pixels, axis=0)

    # BGR→RGBに変換した色リスト
    colors_rgb = []
    for bgr in unique_quantized:
        colors_rgb.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))

    return quantized_image, colors_rgb


def create_color_masks(
    quantized_image: np.ndarray,
    colors: List[Tuple[int, int, int]],
) -> List[np.ndarray]:
    """
    各色のマスク画像を生成

    Args:
        quantized_image: 量子化された画像（BGR）
        colors: 色のリスト（RGB形式）

    Returns:
        マスク画像のリスト
    """
    masks: List[np.ndarray] = []

    for color_rgb in colors:
        # RGB→BGR変換
        color_bgr = np.array([color_rgb[2], color_rgb[1], color_rgb[0]], dtype=np.uint8)

        # この色と一致するピクセルのマスクを作成
        mask = cv2.inRange(quantized_image, color_bgr, color_bgr)
        masks.append(mask)

    return masks


def extract_contours(
    mask: np.ndarray,
    epsilon_factor: float = EPSILON_FACTOR,
    min_area: int = MIN_AREA_ABSOLUTE,
) -> List[dict]:
    """
    マスク画像から輪郭を抽出

    Args:
        mask: 二値マスク画像
        epsilon_factor: 輪郭近似の精度
        min_area: 最小面積閾値

    Returns:
        輪郭パスのリスト（各要素は points, is_hole キーを持つdict）
    """
    # 輪郭検出（階層構造も取得）
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_TREE,  # 階層構造を保持（穴も検出）
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None:
        return []

    result: List[dict] = []
    hierarchy = hierarchy[0]  # shape: (n, 4)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # 最小面積でフィルタリング
        if area < min_area:
            continue

        # 輪郭を近似
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 親がいる場合は穴
        parent_idx = hierarchy[i][3]
        is_hole = parent_idx >= 0

        result.append({
            "points": approx,
            "is_hole": is_hole,
        })

    return result


def apply_noise_reduction(
    mask: np.ndarray,
    kernel_size: int = KERNEL_SIZE,
) -> np.ndarray:
    """
    モルフォロジー演算でノイズ除去

    Args:
        mask: 二値マスク画像
        kernel_size: カーネルサイズ

    Returns:
        ノイズ除去後のマスク
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # オープニング（小さな白いノイズを除去）
    result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # クロージング（小さな穴を埋める）
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result


def contour_to_svg_path(contour: np.ndarray) -> str:
    """
    OpenCV輪郭をSVGパス文字列に変換

    Args:
        contour: OpenCV輪郭

    Returns:
        SVGパス文字列（例: "M 10 10 L 20 10 L 20 20 Z"）
    """
    if len(contour) == 0:
        return ""

    # 最初の点
    points = contour.reshape(-1, 2)
    path_parts = [f"M {points[0][0]} {points[0][1]}"]

    # 残りの点
    for point in points[1:]:
        path_parts.append(f"L {point[0]} {point[1]}")

    # パスを閉じる
    path_parts.append("Z")

    return " ".join(path_parts)


def create_svg(
    contour_paths: List[dict],
    width: int,
    height: int,
    background_color: Tuple[int, int, int] | None = None,
) -> str:
    """
    輪郭パスからSVGを生成

    Args:
        contour_paths: 輪郭パスのリスト
        width: 画像幅
        height: 画像高さ
        background_color: 背景色（RGB）

    Returns:
        SVG文字列
    """
    dwg = svgwrite.Drawing(size=(width, height))

    # 背景を追加
    if background_color:
        r, g, b = background_color
        dwg.add(dwg.rect(
            insert=(0, 0),
            size=(width, height),
            fill=f"rgb({r},{g},{b})",
        ))

    # 輪郭パスを追加
    for contour_path in contour_paths:
        points = contour_path["points"]
        color_rgb = contour_path.get("color_rgb", (0, 0, 0))
        is_hole = contour_path.get("is_hole", False)

        path_d = contour_to_svg_path(points)
        if not path_d:
            continue

        r, g, b = color_rgb

        # 穴の場合は fill-rule で処理
        if is_hole:
            fill = "none"
        else:
            fill = f"rgb({r},{g},{b})"

        dwg.add(dwg.path(d=path_d, fill=fill, stroke="none"))

    return dwg.tostring()


def _is_similar_color(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    threshold: int = 10,
) -> bool:
    """
    2つの色が類似しているか判定

    Args:
        color1: RGB色1
        color2: RGB色2
        threshold: 各チャンネルの許容差

    Returns:
        類似している場合True
    """
    return all(abs(int(c1) - int(c2)) <= threshold for c1, c2 in zip(color1, color2))


def detect_background_color(image: np.ndarray) -> Tuple[int, int, int]:
    """
    画像の背景色を検出（四隅の色から推定）

    Args:
        image: BGR画像

    Returns:
        背景色（RGB）
    """
    h, w = image.shape[:2]

    # 四隅の色を取得（各コーナーから複数ピクセルをサンプリング）
    corner_size = min(10, h // 10, w // 10)
    corners = [
        image[0:corner_size, 0:corner_size],  # 左上
        image[0:corner_size, w - corner_size:w],  # 右上
        image[h - corner_size:h, 0:corner_size],  # 左下
        image[h - corner_size:h, w - corner_size:w],  # 右下
    ]

    # 全ての四隅のピクセルを集める
    all_pixels = []
    for corner in corners:
        pixels = corner.reshape(-1, 3)
        all_pixels.extend([tuple(p) for p in pixels])

    # 最頻色を取得
    counter = Counter(all_pixels)
    most_common_bgr = counter.most_common(1)[0][0]

    # BGR→RGB変換
    return (most_common_bgr[2], most_common_bgr[1], most_common_bgr[0])


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    JPEG/PNG画像のノイズを除去してクリーンな画像にする

    Args:
        image: BGR画像

    Returns:
        デノイズされた画像
    """
    # 1. Non-local means denoising（強力なノイズ除去）
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        DENOISE_H,
        DENOISE_H,
        DENOISE_TEMPLATE_WINDOW,
        DENOISE_SEARCH_WINDOW,
    )

    # 2. メディアンフィルタ（塩胡椒ノイズ、アンチエイリアシング除去）
    denoised = cv2.medianBlur(denoised, MEDIAN_BLUR_SIZE)

    return denoised


def posterize_image(image: np.ndarray, levels: int = COLOR_QUANTIZE_LEVELS) -> np.ndarray:
    """
    画像をポスタリゼーション（色の階調を減らす）

    Args:
        image: BGR画像
        levels: 量子化レベル

    Returns:
        ポスタリゼーションされた画像
    """
    # 各チャンネルを指定レベルに量子化
    # 例: levels=32 なら 256/32=8段階に
    divisor = 256 // levels
    posterized = (image // divisor) * divisor + divisor // 2
    return posterized.astype(np.uint8)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    画像の前処理（デノイズ → ポスタリゼーション）

    Args:
        image: BGR画像

    Returns:
        前処理後の画像
    """
    # Step 1: デノイズ（JPEG/PNGアーティファクト除去）
    denoised = denoise_image(image)

    # Step 2: ポスタリゼーション（色の階調を減らす）
    posterized = posterize_image(denoised, COLOR_QUANTIZE_LEVELS)

    return posterized


def convert_image_to_svg(
    input_path: str,
    output_path: str,
) -> None:
    """
    画像をSVGに変換

    Args:
        input_path: 入力画像パス
        output_path: 出力SVGパス
    """
    # 画像読み込み
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"画像を読み込めません: {input_path}")

    h, w = image.shape[:2]

    # 前処理（アンチエイリアシングノイズ軽減）
    preprocessed = preprocess_image(image)

    # 背景色を検出（元画像から）
    bg_color = detect_background_color(image)

    # 画像を拡大（輪郭検出の精度向上）
    scale = UPSCALE_FACTOR
    upscaled = cv2.resize(
        preprocessed,
        (w * scale, h * scale),
        interpolation=cv2.INTER_NEAREST,  # 最近傍補間でエッジを保持
    )

    # 色量子化（拡大画像に対して）
    quantized, colors = quantize_colors(upscaled, NUM_COLORS)

    # 各色のマスクを生成
    masks = create_color_masks(quantized, colors)

    # 全ての輪郭パスを収集
    all_contour_paths: List[dict] = []

    # 最小面積の計算（拡大後のスケールで）
    min_area = max(int(h * w * MIN_AREA_RATIO), MIN_AREA_ABSOLUTE) * (scale ** 2)

    for color_rgb, mask in zip(colors, masks):
        # 背景色と同じ色はスキップ（背景は別途描画済み）
        if _is_similar_color(color_rgb, bg_color, threshold=20):
            continue

        # ノイズ除去
        cleaned_mask = apply_noise_reduction(mask)

        # 輪郭抽出
        contours = extract_contours(cleaned_mask, min_area=min_area)

        # 座標をスケールダウンして色情報を追加
        for contour in contours:
            # 座標を元のスケールに戻す
            scaled_points = (contour["points"].astype(np.float32) / scale).astype(np.int32)
            contour["points"] = scaled_points
            contour["color_rgb"] = color_rgb
            all_contour_paths.append(contour)

    # SVG生成（元の画像サイズで）
    svg_content = create_svg(all_contour_paths, w, h, bg_color)

    # ファイルに保存
    Path(output_path).write_text(svg_content, encoding="utf-8")

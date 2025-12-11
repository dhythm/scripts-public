"""image_to_svg モジュールのテスト"""
from __future__ import annotations

import sys
from pathlib import Path

# python ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from image_to_svg import (
    quantize_colors,
    create_color_masks,
    extract_contours,
    apply_noise_reduction,
    contour_to_svg_path,
    create_svg,
    detect_background_color,
)


class TestQuantizeColors:
    """色量子化のテスト"""

    def test_quantize_solid_color_image(self):
        """単色画像は1色に量子化される"""
        # 赤一色の画像（BGR形式）
        image = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
        quantized, colors = quantize_colors(image, num_colors=4)

        assert len(colors) == 1
        assert colors[0] == (255, 0, 0)  # RGB形式で返す

    def test_quantize_two_color_image(self):
        """2色画像は2色に量子化される"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :] = [0, 0, 255]  # 上半分: 赤（BGR）
        image[50:, :] = [255, 0, 0]  # 下半分: 青（BGR）

        quantized, colors = quantize_colors(image, num_colors=4)

        assert len(colors) == 2

    def test_quantize_reduces_to_max_colors(self):
        """指定した色数以下に減色される"""
        # グラデーション画像（多くの色を含む）
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            image[i, :] = [i * 2 % 256, i * 3 % 256, i * 5 % 256]

        quantized, colors = quantize_colors(image, num_colors=8)

        assert len(colors) <= 8


class TestCreateColorMasks:
    """マスク生成のテスト"""

    def test_mask_covers_all_pixels(self):
        """全マスクの合計が全ピクセルをカバーする"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :] = [0, 0, 255]
        image[50:, :] = [255, 0, 0]

        quantized, colors = quantize_colors(image, num_colors=4)
        masks = create_color_masks(quantized, colors)

        # 全マスクの論理和が全ピクセルをカバー
        combined = np.zeros((100, 100), dtype=np.uint8)
        for mask in masks:
            combined = np.logical_or(combined, mask > 0)

        assert np.all(combined)

    def test_masks_are_mutually_exclusive(self):
        """マスクは互いに排他的"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :] = [0, 0, 255]
        image[50:, :] = [255, 0, 0]

        quantized, colors = quantize_colors(image, num_colors=4)
        masks = create_color_masks(quantized, colors)

        # 任意の2つのマスクは重複しない
        for i, mask1 in enumerate(masks):
            for j, mask2 in enumerate(masks):
                if i != j:
                    overlap = np.logical_and(mask1 > 0, mask2 > 0)
                    assert not np.any(overlap)


class TestExtractContours:
    """輪郭抽出のテスト"""

    def test_extract_square_contour(self):
        """正方形の輪郭が正しく抽出される"""
        import cv2

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)

        contours = extract_contours(mask, min_area=100)

        assert len(contours) >= 1
        # 最大の輪郭の面積をチェック
        areas = [cv2.contourArea(c["points"]) for c in contours]
        assert max(areas) > 3000  # 60x60 = 3600 に近い値

    def test_extract_contour_with_hole(self):
        """穴のある図形の輪郭が正しく抽出される（外側と穴）"""
        import cv2

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (90, 90), 255, -1)
        cv2.rectangle(mask, (30, 30), (70, 70), 0, -1)  # 穴

        contours = extract_contours(mask, min_area=100)

        # 外側の輪郭と穴の輪郭
        assert len(contours) >= 2

    def test_small_contours_filtered_out(self):
        """小さな輪郭はフィルタリングされる"""
        import cv2

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (40, 40), (60, 60), 255, -1)  # 400px
        cv2.rectangle(mask, (10, 10), (15, 15), 255, -1)  # 25px（小さい）

        contours = extract_contours(mask, min_area=100)

        # 小さい輪郭はフィルタされる
        assert len(contours) == 1


class TestApplyNoiseReduction:
    """ノイズ除去のテスト"""

    def test_removes_small_white_spots(self):
        """小さな白いスポットが除去される"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255  # 1ピクセルのノイズ

        result = apply_noise_reduction(mask, kernel_size=3)

        assert result[50, 50] == 0

    def test_preserves_large_regions(self):
        """大きな領域は保持される"""
        import cv2

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)

        result = apply_noise_reduction(mask, kernel_size=3)

        # 中心部分は保持される
        assert result[50, 50] == 255


class TestContourToSvgPath:
    """輪郭からSVGパス変換のテスト"""

    def test_simple_polygon_path(self):
        """シンプルなポリゴンが正しいパス文字列になる"""
        contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])
        path = contour_to_svg_path(contour)

        assert path.startswith("M")
        assert "10 10" in path or "10,10" in path
        assert path.endswith("Z")

    def test_path_is_closed(self):
        """パスが閉じている（Zで終わる）"""
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]]])
        path = contour_to_svg_path(contour)

        assert path.endswith("Z")


class TestCreateSvg:
    """SVG生成のテスト"""

    def test_svg_has_correct_dimensions(self):
        """SVGのサイズが正しい"""
        contour_paths = []
        svg_content = create_svg(contour_paths, width=200, height=100)

        assert 'width="200"' in svg_content
        assert 'height="100"' in svg_content

    def test_svg_is_valid_xml(self):
        """生成されたSVGが有効なXML"""
        import xml.etree.ElementTree as ET

        contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])
        contour_paths = [{"points": contour, "color_rgb": (255, 0, 0), "is_hole": False}]
        svg_content = create_svg(contour_paths, width=100, height=100)

        # XMLとしてパースできる
        ET.fromstring(svg_content)


class TestDetectBackgroundColor:
    """背景色検出のテスト"""

    def test_white_background(self):
        """白背景が検出される"""
        image = np.full((100, 100, 3), [255, 255, 255], dtype=np.uint8)
        color = detect_background_color(image)

        assert color == (255, 255, 255)

    def test_colored_background(self):
        """色付き背景が検出される（四隅の最頻色）"""
        # 四隅が青の画像
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:] = [255, 0, 0]  # 青（BGR）
        # 中央に別の色
        image[30:70, 30:70] = [0, 255, 0]

        color = detect_background_color(image)

        assert color == (0, 0, 255)  # RGB形式で青

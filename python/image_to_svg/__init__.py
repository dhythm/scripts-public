"""画像からSVGへの変換ツール"""
from __future__ import annotations

# converter は OpenCV 依存のため環境によっては未インストールになり得る。遅延インポートで安全に扱う。
try:  # pragma: no cover - 依存が揃っている場合のみ実行
    from .converter import (
        denoise_image,
        remove_antialiasing,
        count_unique_colors,
        reduce_colors_kmeans,
        merge_similar_colors,
        process_step_by_step,
        vectorize_with_vtracer,
    )
    _converter_available = True
except Exception as _converter_import_error:  # pragma: no cover - 依存不足時のフォールバック
    _converter_available = False

    def _lazy_fail(*_args, **_kwargs):  # type: ignore
        raise ImportError(
            "converter系機能を使うには 'opencv-python' などの依存をインストールしてください。"
        ) from _converter_import_error

    denoise_image = remove_antialiasing = count_unique_colors = reduce_colors_kmeans = (  # type: ignore  # noqa: E501
        merge_similar_colors
    ) = process_step_by_step = vectorize_with_vtracer = _lazy_fail  # type: ignore

# ピクセル矩形SVG生成（cv2不要）
from .pixel_rect_svg import build_svg, save_svg_from_png

__all__ = [
    "denoise_image",
    "remove_antialiasing",
    "count_unique_colors",
    "reduce_colors_kmeans",
    "merge_similar_colors",
    "process_step_by_step",
    "vectorize_with_vtracer",
    "build_svg",
    "save_svg_from_png",
]

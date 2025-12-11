"""画像からSVGへの変換CLI"""
from __future__ import annotations

import sys
from pathlib import Path

from .converter import convert_image_to_svg


def main() -> None:
    """CLIエントリーポイント"""
    if len(sys.argv) < 2:
        print("使い方: python -m image_to_svg <入力画像> [出力SVG]")
        print()
        print("例:")
        print("  python -m image_to_svg input.png output.svg")
        print("  python -m image_to_svg input.png  # → input.svg が生成される")
        sys.exit(1)

    input_path = sys.argv[1]

    # 入力ファイルの存在確認
    if not Path(input_path).exists():
        print(f"エラー: ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # 出力パスの決定
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # 拡張子を .svg に変更
        output_path = str(Path(input_path).with_suffix(".svg"))

    print(f"変換中: {input_path} → {output_path}")

    try:
        convert_image_to_svg(input_path, output_path)
        print(f"完了: {output_path}")
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

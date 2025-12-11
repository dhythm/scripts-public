"""画像からSVGへの変換CLI"""
from __future__ import annotations

import sys
from pathlib import Path

from .converter import process_step_by_step


def main() -> None:
    """CLIエントリーポイント"""
    if len(sys.argv) < 2:
        print("使い方: python -m image_to_svg <入力画像> [出力ディレクトリ] [色数]")
        print()
        print("例:")
        print("  python -m image_to_svg input.png ./output")
        print("  python -m image_to_svg input.png ./output 10")
        sys.exit(1)

    input_path = sys.argv[1]

    # 入力ファイルの存在確認
    if not Path(input_path).exists():
        print(f"エラー: ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # 出力ディレクトリの決定
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = "./output"

    # 色数の決定
    num_colors = int(sys.argv[3]) if len(sys.argv) >= 4 else 10

    print(f"変換中: {input_path} → {output_dir}")

    try:
        results = process_step_by_step(input_path, output_dir, num_colors)
        print(f"完了: {results['svg_path']}")
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

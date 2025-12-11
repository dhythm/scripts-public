"""画像からSVGへの変換CLI"""
from __future__ import annotations

import sys
from pathlib import Path

from .converter import process_step_by_step


def main() -> None:
    """CLIエントリーポイント"""
    if len(sys.argv) < 2:
        print("使い方: python -m image_to_svg <入力画像> [出力ディレクトリ] [色数] [--intermediate]")
        print()
        print("例:")
        print("  python -m image_to_svg input.png ./output")
        print("  python -m image_to_svg input.png ./output 10")
        print("  python -m image_to_svg input.png ./output 10 --intermediate")
        print()
        print("オプション:")
        print("  --intermediate  各イテレーションの中間結果をPNGで保存")
        sys.exit(1)

    # --intermediate フラグの確認
    save_intermediate = "--intermediate" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--intermediate"]

    input_path = args[0]

    # 入力ファイルの存在確認
    if not Path(input_path).exists():
        print(f"エラー: ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # 出力ディレクトリの決定
    if len(args) >= 2:
        output_dir = args[1]
    else:
        output_dir = "./output"

    # 色数の決定
    num_colors = int(args[2]) if len(args) >= 3 else 256

    print(f"変換中: {input_path} → {output_dir}")
    if save_intermediate:
        print("中間結果を保存します")

    try:
        results = process_step_by_step(
            input_path, output_dir, num_colors, save_intermediate=save_intermediate
        )
        print(f"完了: {results['svg_path']}")
        if save_intermediate:
            print(f"中間結果: {output_dir}/{Path(input_path).stem}_intermediate/")
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

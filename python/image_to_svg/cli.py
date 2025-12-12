"""画像からSVGへの変換CLI"""
from __future__ import annotations

import sys
from pathlib import Path

from .converter import process_step_by_step


def main() -> None:
    """CLIエントリーポイント"""
    if len(sys.argv) < 2:
        print("使い方: python -m image_to_svg <入力画像> [出力ディレクトリ] [最終色数] [オプション]")
        print()
        print("例:")
        print("  python -m image_to_svg input.png ./output")
        print("  python -m image_to_svg input.png ./output 256")
        print("  python -m image_to_svg input.png ./output 256 --scale 4")
        print("  python -m image_to_svg input.png ./output 256 --intermediate")
        print("  python -m image_to_svg input.png ./output 256 --legacy")
        print("  python -m image_to_svg input.png ./output 256 --polygon")
        print()
        print("オプション:")
        print("  --scale N       拡大倍率（デフォルト: 4、大きいほど境界が滑らか）")
        print("  --intermediate  各ステップの中間結果をPNGで保存")
        print("  --legacy        従来の直線パスSVG生成を使用（デフォルト: vtracer）")
        print("  --polygon       vtracer使用時に多角形モードを使用（デフォルト: spline）")
        print("  --absorption    小領域吸収を有効化（デフォルト: スキップ、低速）")
        print()
        print("処理フロー:")
        print("  1. 画像を拡大（最近傍補間で色を保持）")
        print("  2. K-meansで色数削減（元画像の色を保持）")
        print("  3. 小領域の吸収（--absorptionで有効化、デフォルトはスキップ）")
        print("  4. SVG生成（vtracer: ベジェ曲線、legacy: 直線パス）")
        sys.exit(1)

    # オプションの解析
    save_intermediate = "--intermediate" in sys.argv
    use_legacy = "--legacy" in sys.argv
    use_polygon = "--polygon" in sys.argv
    use_absorption = "--absorption" in sys.argv

    # --scale オプションの解析
    upscale_factor = 4  # デフォルト
    args_filtered = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--scale" and i + 1 < len(sys.argv):
            upscale_factor = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] in ("--intermediate", "--legacy", "--polygon", "--absorption"):
            i += 1
        else:
            args_filtered.append(sys.argv[i])
            i += 1

    if not args_filtered:
        print("エラー: 入力画像を指定してください")
        sys.exit(1)

    input_path = args_filtered[0]

    # 入力ファイルの存在確認
    if not Path(input_path).exists():
        print(f"エラー: ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # 出力ディレクトリの決定
    output_dir = args_filtered[1] if len(args_filtered) >= 2 else "./output"

    # 最終色数の決定
    final_colors = int(args_filtered[2]) if len(args_filtered) >= 3 else 256

    # vtracer設定
    use_vtracer = not use_legacy
    vtracer_mode = "polygon" if use_polygon else "spline"

    print(f"変換中: {input_path} → {output_dir}")
    print(f"目標色数: {final_colors}")
    print(f"拡大倍率: {upscale_factor}倍")
    print(f"SVG生成: {'vtracer (' + vtracer_mode + ')' if use_vtracer else 'legacy'}")
    print(f"小領域吸収: {'有効' if use_absorption else 'スキップ'}")
    if save_intermediate:
        print("中間結果を保存します")

    try:
        results = process_step_by_step(
            input_path,
            output_dir,
            save_intermediate=save_intermediate,
            final_colors=final_colors,
            upscale_factor=upscale_factor,
            use_vtracer=use_vtracer,
            vtracer_mode=vtracer_mode,
            skip_region_absorption=not use_absorption,
        )
        print(f"完了: {results['svg_path']}")
        print(f"最終色数: {results['final_colors']}")
        print(f"パス数: {results['path_count']}")
        if save_intermediate:
            print(f"中間結果: {output_dir}/{Path(input_path).stem}_intermediate/")
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

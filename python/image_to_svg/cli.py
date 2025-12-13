"""画像からSVGへの変換CLI"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pixel_rect_svg import save_svg_from_png


def main() -> None:
    """CLIエントリーポイント"""
    argv = sys.argv[1:]

    # --- pixel-rect モード（OpenCV不要） --------------------------------------------
    if "--pixel-rect" in argv:
        pr = argparse.ArgumentParser(
            description="クロップ領域を<rect>群としてSVG化（ラスタ埋め込みなし）"
        )
        pr.add_argument("--pixel-rect", action="store_true", help="pixel-rect モードを有効化")
        pr.add_argument("--input", required=True, help="入力PNGパス")
        pr.add_argument(
            "--output",
            help="出力SVGパス（未指定時は入力と同名で拡張子.svg）",
        )
        pr.add_argument("--config", help="レイアウト定義 (YAML/JSON)")
        pr.add_argument(
            "--no-merge-runs",
            action="store_true",
            help="横方向run結合を無効化（1pxごとにrectを出力）",
        )
        args = pr.parse_args(argv)

        if not Path(args.input).exists():
            print(f"エラー: ファイルが見つかりません: {args.input}")
            sys.exit(1)

        output = args.output or str(Path(args.input).with_suffix(".svg"))

        try:
            save_svg_from_png(
                input_path=args.input,
                output_path=output,
                config_path=args.config,
                merge_runs=not args.no_merge_runs,
            )
            print(f"完了: {output}")
        except Exception as e:  # pragma: no cover - 実行時エラー表示
            print(f"エラー: {e}")
            sys.exit(1)
        return

    # --- 既存（vtracer等）モード ----------------------------------------------------
    if len(argv) < 1:
        print("使い方: python -m image_to_svg <入力画像> [出力ディレクトリ] [最終色数] [オプション]")
        print()
        print("例:")
        print("  python -m image_to_svg input.png ./output")
        print("  python -m image_to_svg input.png ./output 256 --scale 4 --intermediate")
        print()
        print("オプション:")
        print("  --scale N         拡大倍率（デフォルト: 4、大きいほど境界が滑らか）")
        print("  --filter-speckle N  vtracerのノイズ除去閾値（デフォルト: 16）")
        print("  --color-merge N   類似色マージ閾値（デフォルト: 10.0、LAB色空間）")
        print("  --intermediate    各ステップの中間結果をPNGで保存")
        print("  --legacy          従来の直線パスSVG生成を使用（デフォルト: vtracer）")
        print("  --polygon         vtracer使用時に多角形モードを使用（デフォルト: spline）")
        print("  --absorption      小領域吸収を有効化（デフォルト: スキップ）")
        sys.exit(1)

    # オプションの解析（従来処理）
    save_intermediate = "--intermediate" in argv
    use_legacy = "--legacy" in argv
    use_polygon = "--polygon" in argv
    use_absorption = "--absorption" in argv

    upscale_factor = 4  # デフォルト
    filter_speckle = 16  # デフォルト
    color_merge_threshold = 10.0  # デフォルト
    args_filtered = []
    i = 0
    while i < len(argv):
        if argv[i] == "--scale" and i + 1 < len(argv):
            upscale_factor = int(argv[i + 1])
            i += 2
        elif argv[i] == "--filter-speckle" and i + 1 < len(argv):
            filter_speckle = int(argv[i + 1])
            i += 2
        elif argv[i] == "--color-merge" and i + 1 < len(argv):
            color_merge_threshold = float(argv[i + 1])
            i += 2
        elif argv[i] in ("--intermediate", "--legacy", "--polygon", "--absorption"):
            i += 1
        else:
            args_filtered.append(argv[i])
            i += 1

    if not args_filtered:
        print("エラー: 入力画像を指定してください")
        sys.exit(1)

    input_path = args_filtered[0]

    # 入力ファイルの存在確認
    if not Path(input_path).exists():
        print(f"エラー: ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # --pixel-rect モードなら専用処理に切り替え
    if use_pixel_rect:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--input", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--config")
        parsed, _ = parser.parse_known_args(args_filtered)

        try:
            save_svg_from_png(
                input_path=parsed.input,
                output_path=parsed.output,
                config_path=parsed.config,
                merge_runs=not no_merge_runs,
            )
            print(f"完了: {parsed.output}")
        except Exception as e:
            print(f"エラー: {e}")
            sys.exit(1)
        return

    # 出力ディレクトリの決定（通常モード）
    output_dir = args_filtered[1] if len(args_filtered) >= 2 else "./output"

    # 最終色数の決定（通常モード）
    final_colors = int(args_filtered[2]) if len(args_filtered) >= 3 else 256

    # vtracer設定
    use_vtracer = not use_legacy
    vtracer_mode = "polygon" if use_polygon else "spline"

    print(f"変換中: {input_path} → {output_dir}")
    print(f"目標色数: {final_colors}")
    print(f"拡大倍率: {upscale_factor}倍")
    print(f"SVG生成: {'vtracer (' + vtracer_mode + ')' if use_vtracer else 'legacy'}")
    print(f"filter_speckle: {filter_speckle}")
    print(f"類似色マージ閾値: {color_merge_threshold}")
    print(f"小領域吸収: {'有効' if use_absorption else 'スキップ'}")
    if save_intermediate:
        print("中間結果を保存します")

    try:
        # converter依存はここで遅延インポート（cv2未導入でもpixel-rectモードが動くようにする）
        from .converter import process_step_by_step

        results = process_step_by_step(
            input_path,
            output_dir,
            save_intermediate=save_intermediate,
            final_colors=final_colors,
            upscale_factor=upscale_factor,
            use_vtracer=use_vtracer,
            vtracer_mode=vtracer_mode,
            skip_region_absorption=not use_absorption,
            color_merge_threshold=color_merge_threshold,
            filter_speckle=filter_speckle,
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

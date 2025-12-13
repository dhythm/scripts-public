"""画像からSVGへの変換CLI"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pixel_rect_svg import save_svg_from_png


def _merge_similar_colors_internal(image, threshold: float = 10.0):
    """
    類似色をマージして色数を削減（cli.py内部用）

    converterモジュールに依存せずにLAB色空間で類似色をマージする。
    """
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)

    # ユニーク色と出現回数を取得
    unique_colors, inverse, counts = np.unique(
        pixels, axis=0, return_inverse=True, return_counts=True
    )

    if len(unique_colors) <= 1:
        return image

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

        # 未処理の色との距離を計算
        unprocessed_mask = ~processed
        unprocessed_indices = np.where(unprocessed_mask)[0]

        if len(unprocessed_indices) == 0:
            break

        unprocessed_labs = unique_colors_lab[unprocessed_indices]
        distances = np.sqrt(np.sum((unprocessed_labs - color_lab) ** 2, axis=1))

        # 閾値以内の色をマージ
        similar_mask = distances <= threshold
        similar_indices = unprocessed_indices[similar_mask]

        # 最頻色にマッピング
        for sim_idx in similar_indices:
            mapping[sim_idx] = idx
            processed[sim_idx] = True

    # マッピングを適用
    new_color_indices = mapping[inverse]
    new_pixels = unique_colors[new_color_indices]

    return new_pixels.reshape(h, w, 3).astype(np.uint8)


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
        pr.add_argument(
            "--no-merge-vertical",
            action="store_true",
            help="縦方向マージを無効化",
        )
        pr.add_argument(
            "--preprocess",
            action="store_true",
            help="k-means色削減を事前実行（OpenCV必要）",
        )
        pr.add_argument(
            "--colors",
            type=int,
            default=256,
            help="色削減の目標色数（--preprocess時、デフォルト256）",
        )
        pr.add_argument(
            "--keep-temp",
            action="store_true",
            help="前処理で生成した一時PNGを保持（--preprocess時）",
        )
        args = pr.parse_args(argv)

        if not Path(args.input).exists():
            print(f"エラー: ファイルが見つかりません: {args.input}")
            sys.exit(1)

        output = args.output or str(Path(args.input).with_suffix(".svg"))

        try:
            input_for_svg = args.input
            temp_png_path = None

            # 前処理（色削減）
            if args.preprocess:
                print(f"前処理: k-means色削減（目標: {args.colors}色）...")
                import tempfile
                import os

                try:
                    import cv2
                    import numpy as np
                    from PIL import Image
                except ImportError as e:
                    raise RuntimeError(
                        f"--preprocess にはOpenCV (cv2) が必要です: {e}\n"
                        "pip install opencv-python でインストールしてください"
                    )

                # 一時ファイルを作成
                temp_fd, temp_png_path = tempfile.mkstemp(suffix="_reduced.png")
                os.close(temp_fd)

                # k-meansで色削減（converterのインポートを避けて直接実装）
                img = cv2.imread(args.input)
                if img is None:
                    raise ValueError(f"画像を読み込めません: {args.input}")

                h, w = img.shape[:2]
                pixels = img.reshape(-1, 3)

                # LAB色空間でK-means
                lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                pixels_lab = lab_img.reshape(-1, 3).astype(np.float32)

                # クラスタ数を画像サイズに応じて調整（ピクセル数より多くできない）
                num_pixels = len(pixels_lab)
                num_colors = min(args.colors, num_pixels)
                if num_colors < args.colors:
                    print(f"  注意: ピクセル数({num_pixels})が目標色数より少ないため、{num_colors}色に調整")

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, _ = cv2.kmeans(
                    pixels_lab,
                    num_colors,
                    None,
                    criteria,
                    10,
                    cv2.KMEANS_PP_CENTERS,
                )
                labels_flat = labels.flatten()

                # 各クラスタの代表色を元画像の最頻色から選択
                representative_colors = np.zeros((num_colors, 3), dtype=np.uint8)
                for cluster_id in range(num_colors):
                    mask = labels_flat == cluster_id
                    if not np.any(mask):
                        continue
                    cluster_pixels = pixels[mask]
                    unique, counts = np.unique(cluster_pixels, axis=0, return_counts=True)
                    representative_colors[cluster_id] = unique[np.argmax(counts)]

                # 量子化画像を生成
                quantized_pixels = representative_colors[labels_flat]
                quantized_img = quantized_pixels.reshape(h, w, 3).astype(np.uint8)

                unique_colors_kmeans = np.unique(quantized_pixels, axis=0)
                print(f"  k-means後: {len(unique_colors_kmeans)}色")

                # 類似色マージ（LAB色空間）
                merged = _merge_similar_colors_internal(quantized_img, threshold=10.0)

                # 最終色数をカウント
                unique_colors = np.unique(merged.reshape(-1, 3), axis=0)
                print(f"  マージ後: {len(unique_colors)}色")

                # 保存
                merged_rgb = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
                Image.fromarray(merged_rgb).save(temp_png_path, "PNG")

                input_for_svg = temp_png_path

            save_svg_from_png(
                input_path=input_for_svg,
                output_path=output,
                config_path=args.config,
                merge_runs=not args.no_merge_runs,
                merge_vertical=not args.no_merge_vertical,
            )
            print(f"完了: {output}")

            # 一時ファイルの削除
            if temp_png_path and not args.keep_temp:
                Path(temp_png_path).unlink(missing_ok=True)
            elif temp_png_path and args.keep_temp:
                print(f"一時PNG保持: {temp_png_path}")

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

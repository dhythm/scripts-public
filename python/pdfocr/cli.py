"""pdfocr のコマンドライン

使用方法:
    uv run python -m pdfocr <PDF> [-o 出力] [--format txt|md] [--engine ndl|yomitoku]
                                  [--ocr auto|always|never] [--dpi 300] [--force]
    uv run python -m pdfocr setup {ndl,yomitoku} [--update]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from pdfocr.engines import ENGINES, run_ndl, run_yomitoku, setup_ndl, setup_yomitoku
from pdfocr.output import save_text, validate_output
from pdfocr.pipeline import DEFAULT_DPI, OCR_MODES, convert
from pdfocr.postprocess import (
    dedupe_consecutive_pages,
    format_text_to_markdown,
    normalize_special_chars,
    remove_spaces_between_japanese,
)
from pdfocr.textlayer import DEFAULT_FOOTER_RATIO, DEFAULT_HEADER_RATIO, extract_text_pages



def _run_setup(engine: str, update: bool) -> None:
    # テストで差し替えられるよう、呼び出し時にモジュールの関数を参照する
    if engine == "ndl":
        setup_ndl(update=update)
    else:
        setup_yomitoku(update=update)


def _ocr_runner(engine: str):
    return run_ndl if engine == "ndl" else run_yomitoku


def _setup_main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="pdfocr setup", description="OCRエンジンをセットアップする")
    parser.add_argument("engine", choices=ENGINES)
    parser.add_argument("--update", action="store_true", help="既存のインストールを更新する")
    args = parser.parse_args(argv)
    _run_setup(args.engine, args.update)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfocr",
        description="PDFをローカルでOCRしてテキスト/Markdownにする（テキスト層があるページはそのまま抽出）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
エンジン:
  ndl       NDLOCR-Lite（デフォルト）。縦書きの和書に強く、GPUなしで高速。ルビは本文に含めない
  yomitoku  YomiToku。読み順の推定や表に強い。ルビは除去する。個人利用は無料（商用は有料）

初回は使用するエンジンをセットアップしてください:
  uv run python -m pdfocr setup ndl
""",
    )
    parser.add_argument("pdf", type=Path, help="入力PDF")
    parser.add_argument("-o", "--output", type=Path, help="出力ファイル（省略時はPDFと同じ場所に .txt / .md）")
    parser.add_argument("--format", choices=["txt", "md"], default="txt", help="出力形式 (default: txt)")
    parser.add_argument("--engine", choices=ENGINES, default="ndl", help="OCRエンジン (default: ndl)")
    parser.add_argument(
        "--ocr", choices=OCR_MODES, default="auto",
        help="auto: テキスト層が使えないページだけOCR / always: 全ページOCR / never: テキスト層のみ (default: auto)",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"OCR時の画像化の解像度 (default: {DEFAULT_DPI})")
    parser.add_argument(
        "--keep-margins", action="store_true",
        help="テキスト層の抽出で、ページ上下の帯にある柱（繰り返し現れる見出し）とページ番号を除外しない",
    )
    parser.add_argument("--force", action="store_true", help="既存の出力ファイルを上書きする")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["setup"]:
        try:
            return _setup_main(argv[1:])
        except Exception as e:
            print(f"エラー: セットアップに失敗しました: {e}", file=sys.stderr)
            return 1

    args = build_parser().parse_args(argv)
    output = args.output or args.pdf.with_suffix(f".{args.format}")

    try:
        if not args.pdf.is_file():
            raise FileNotFoundError(f"PDFが見つかりません: {args.pdf}")
        # 時間のかかる抽出・OCRの前に出力先を検証する
        validate_output(args.pdf, output, overwrite=args.force)

        header_ratio, footer_ratio = (0.0, 0.0) if args.keep_margins else (DEFAULT_HEADER_RATIO, DEFAULT_FOOTER_RATIO)
        pages = convert(
            args.pdf,
            ocr_mode=args.ocr,
            run_ocr=_ocr_runner(args.engine),
            dpi=args.dpi,
            extract=lambda path: extract_text_pages(path, header_ratio, footer_ratio),
        )

        pages, skipped = dedupe_consecutive_pages(pages)
        if skipped:
            print(f"連続する重複ページを {skipped} ページスキップしました。")
        text = "\n\n".join(page.strip() for page in pages if page.strip())
        text = remove_spaces_between_japanese(normalize_special_chars(text))
        if not text.strip():
            raise RuntimeError("テキストを抽出できませんでした。")
        if args.format == "md":
            text = format_text_to_markdown(text, args.pdf.name)

        save_text(args.pdf, output, text, overwrite=args.force)
        print(f"保存しました: {output}")
        return 0
    except KeyboardInterrupt:
        print("\n処理が中断されました。", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

"""pdfocr パッケージのテスト"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdfocr.engines import parse_indra_jsonl, parse_ndl_json, parse_yomitoku_json
from pdfocr.postprocess import (
    dedupe_consecutive_pages,
    format_text_to_markdown,
    normalize_special_chars,
)
from pdfocr.textlayer import find_running_heads, needs_ocr, select_body_boxes


class TestDedupeConsecutivePages:
    """連続する同一ページの除去（pdf_to_markdown.py から移植）"""

    def test_skips_consecutive_duplicates_ignoring_whitespace(self):
        pages, skipped = dedupe_consecutive_pages(["本文 A", "本文  A\n", "本文 B"])
        assert pages == ["本文 A", "本文 B"]
        assert skipped == 1

    def test_keeps_non_consecutive_duplicates(self):
        pages, skipped = dedupe_consecutive_pages(["A", "B", "A"])
        assert pages == ["A", "B", "A"]
        assert skipped == 0


class TestNormalizeSpecialChars:
    """縦書き用の約物を横書きに変換（pdf_to_markdown.py から移植）"""

    def test_vertical_forms_are_converted(self):
        assert normalize_special_chars("﹁はい︑そうです︒﹂") == "「はい、そうです。」"

    def test_replacement_character_is_removed(self):
        assert normalize_special_chars("文�字") == "文字"


class TestFormatTextToMarkdown:
    """Markdown整形（pdf_to_markdown.py と PR #7 の修正を移植）"""

    def test_title_and_chapter_heading(self):
        md = format_text_to_markdown("第一章 はじまり\n本文です。", "book.pdf")
        assert md.startswith("# book\n")
        assert "\n## 第一章 はじまり\n" in md
        assert "本文です。" in md

    def test_page_numbers_are_dropped(self):
        md = format_text_to_markdown("本文です。\n12\n続きです。", "book.pdf")
        assert "12" not in md

    def test_trailing_copyright_is_kept(self):
        """文末の著作権表記が消えない（PR #7 の修正）"""
        md = format_text_to_markdown("本文です。\nCopyright © 2020 Author", "book.pdf")
        assert "**原書情報:**" in md
        assert "> Copyright © 2020 Author" in md

    def test_body_with_press_is_not_metadata(self):
        """Press や Agency を含む本文を原書情報に誤分類しない（PR #7 の修正）"""
        md = format_text_to_markdown("The Press reported it.", "book.pdf")
        assert "原書情報" not in md
        assert "The Press reported it." in md


class TestSelectBodyBoxes:
    """柱（ランニングヘッド）とページ番号だけを除外する"""

    # (x0, y0, x1, y1, text)。PDF座標系は下が0、ページの高さは1000

    def test_drops_running_head_and_page_number(self):
        boxes = [
            (0, 950, 100, 990, "吾輩は猫である"),
            (0, 200, 100, 800, "本文"),
            (0, 10, 100, 40, "12"),
        ]
        assert select_body_boxes(boxes, 1000, running_heads={"吾輩は猫である"}) == ["本文"]

    def test_keeps_column_that_starts_in_header_band(self):
        """上端の帯から始まって下へ続く縦書きの列は本文として残す"""
        boxes = [(900, 300, 940, 960, "縦書きの本文の列")]
        assert select_body_boxes(boxes, 1000) == ["縦書きの本文の列"]

    def test_keeps_one_off_heading_in_header_band(self):
        """繰り返し現れない見出しは、上端の帯にあっても残す"""
        boxes = [(0, 950, 100, 990, "第一章　はじまり"), (0, 200, 100, 800, "本文")]
        assert select_body_boxes(boxes, 1000) == ["第一章　はじまり", "本文"]

    def test_custom_ratios(self):
        boxes = [(0, 10, 100, 40, "12")]
        assert select_body_boxes(boxes, 1000, footer_ratio=0.0) == ["12"]


class TestFindRunningHeads:
    """複数ページの同じ帯に繰り返し現れるテキストを柱として検出する"""

    def test_repeated_band_text_is_running_head(self):
        pages = [
            (1000, [(0, 950, 100, 990, "吾輩は猫である 12"), (0, 200, 100, 800, "本文A")]),
            (1000, [(0, 950, 100, 990, "吾輩は猫である 13"), (0, 200, 100, 800, "本文B")]),
        ]
        # ページ番号の数字が違っても同じ柱とみなす
        assert find_running_heads(pages) == {"吾輩は猫である"}

    def test_text_on_single_page_is_not_running_head(self):
        pages = [
            (1000, [(0, 950, 100, 990, "第一章　はじまり")]),
            (1000, [(0, 200, 100, 800, "本文")]),
        ]
        assert find_running_heads(pages) == set()


class TestNeedsOcr:
    """テキスト層がそのまま使えるか（OCRが必要か）の判定"""

    def test_good_japanese_text_does_not_need_ocr(self):
        assert needs_ocr("吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。") is False

    def test_good_english_text_does_not_need_ocr(self):
        assert needs_ocr("The quick brown fox jumps over the lazy dog. " * 2) is False

    def test_empty_or_tiny_text_needs_ocr(self):
        assert needs_ocr("") is True
        assert needs_ocr("  12  ") is True

    def test_cid_glyphs_need_ocr(self):
        """フォントの対応表がなく (cid:NNN) になっているテキスト層"""
        assert needs_ocr("(cid:1234)(cid:567)(cid:89)" * 10) is True

    def test_replacement_characters_need_ocr(self):
        assert needs_ocr("文字化け����" * 5) is True

    def test_garbled_symbols_need_ocr(self):
        """想定外の文字（文字化け）が大半を占めるテキスト層"""
        assert needs_ocr("ÃƒÂ¤Ã‚Â¸Ã‚Â­ÃƒÂ¦Ã‚Â–Ã‚Â‡" * 5) is True


class TestParseNdlJson:
    """NDLOCR-Lite の JSON 出力からページ本文を取り出す"""

    def test_joins_lines_in_order(self, tmp_path):
        data = {"contents": [[
            {"id": 0, "text": "一行目", "isTextline": "true"},
            {"id": 1, "text": "二行目", "isTextline": "true"},
        ]]}
        path = tmp_path / "page.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        assert parse_ndl_json(path) == "一行目\n二行目"

    def test_empty_page(self, tmp_path):
        path = tmp_path / "page.json"
        path.write_text(json.dumps({"contents": [[]]}), encoding="utf-8")
        assert parse_ndl_json(path) == ""


class TestParseYomitokuJson:
    """YomiToku の JSON 出力から読み順どおりに本文を取り出す"""

    def _write(self, tmp_path, paragraphs):
        path = tmp_path / "page.json"
        path.write_text(json.dumps({"paragraphs": paragraphs}, ensure_ascii=False), encoding="utf-8")
        return path

    def test_orders_paragraphs_by_reading_order(self, tmp_path):
        path = self._write(tmp_path, [
            {"order": 1, "contents": "二段落目", "role": None},
            {"order": 0, "contents": "一段落目", "role": None},
        ])
        assert parse_yomitoku_json(path) == "一段落目\n\n二段落目"

    def test_drops_page_header_and_footer(self, tmp_path):
        path = self._write(tmp_path, [
            {"order": 0, "contents": "柱", "role": "page_header"},
            {"order": 1, "contents": "本文", "role": None},
            {"order": 2, "contents": "12", "role": "page_footer"},
        ])
        assert parse_yomitoku_json(path) == "本文"


class TestParseIndraJsonl:
    """Indra OCR の JSON Lines 出力から読み順どおりに本文を取り出す"""

    def _write(self, tmp_path, **page):
        record = {"page": 0, "source": "ocr", "note": "", "lines": [], "blocks": [], **page}
        path = tmp_path / "page.jsonl"
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def test_joins_blocks_and_drops_page_header_and_footer(self, tmp_path):
        path = self._write(tmp_path, blocks=[
            {"label": "Page-Header", "bbox": [0, 0, 1, 1], "text": "吾輩は猫である 12"},
            {"label": "Section-Header", "bbox": [0, 0, 1, 1], "text": "第一章 はじまり"},
            {"label": "Text", "bbox": [0, 0, 1, 1], "text": "一行目\n二行目"},
            {"label": "Page-Footer", "bbox": [0, 0, 1, 1], "text": "12"},
        ])
        assert parse_indra_jsonl(path) == "第一章 はじまり\n\n一行目\n二行目"

    def test_falls_back_to_lines_without_blocks(self, tmp_path):
        path = self._write(tmp_path, lines=[
            {"box": [0, 0, 1, 1], "text": "一行目", "conf": 0.9},
            {"box": [0, 0, 1, 1], "text": "二行目", "conf": 0.9},
        ])
        assert parse_indra_jsonl(path) == "一行目\n二行目"

    def test_failed_page_is_empty(self, tmp_path, capsys):
        path = self._write(tmp_path, source="error", note="decode failed")
        assert parse_indra_jsonl(path) == ""
        assert "decode failed" in capsys.readouterr().err


class TestRunIndra:
    """Indra OCR の CLI 呼び出し"""

    def test_runs_cli_in_ocr_mode_and_reads_each_output(self, tmp_path, monkeypatch):
        from pdfocr import engines
        install = tmp_path / "indra"
        (install / ".venv" / "bin").mkdir(parents=True)
        (install / ".venv" / "bin" / "python").touch()
        images = [tmp_path / "pages" / "page_0001.png", tmp_path / "pages" / "page_0002.png"]
        calls = []

        def fake_run(cmd, check):
            calls.append(cmd)
            out_dir = Path(cmd[cmd.index("--out") + 1])
            out_dir.mkdir(parents=True)
            for image in images:
                record = {"source": "ocr", "blocks": [{"label": "Text", "text": f"本文{image.stem}"}]}
                (out_dir / f"{image.stem}.jsonl").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

        monkeypatch.setattr(engines.subprocess, "run", fake_run)
        results = engines.run_indra(images, tmp_path, path=install)

        cmd = calls[0]
        assert cmd[0] == str(install / ".venv" / "bin" / "indra-ocr")
        assert cmd[1:3] == [str(p) for p in images]
        assert cmd[cmd.index("--mode") + 1] == "ocr"
        assert cmd[cmd.index("--format") + 1] == "json"
        assert results == {images[0]: "本文page_0001", images[1]: "本文page_0002"}

    def test_missing_output_raises(self, tmp_path, monkeypatch):
        from pdfocr import engines
        install = tmp_path / "indra"
        (install / ".venv" / "bin").mkdir(parents=True)
        (install / ".venv" / "bin" / "python").touch()
        monkeypatch.setattr(engines.subprocess, "run", lambda cmd, check: None)
        with pytest.raises(RuntimeError, match="page_0001.png"):
            engines.run_indra([tmp_path / "page_0001.png"], tmp_path, path=install)

    def test_not_installed_raises_with_setup_hint(self, tmp_path):
        from pdfocr import engines
        with pytest.raises(RuntimeError, match="setup indra"):
            engines.run_indra([tmp_path / "page_0001.png"], tmp_path, path=tmp_path / "none")


class TestSectionHeadings:
    """節見出し（一　はじめに / 一．/ 一、）の検出"""

    @pytest.mark.parametrize("heading", ["一　はじめに", "二．方法", "三、結果"])
    def test_kanji_numeral_section_headings(self, heading):
        md = format_text_to_markdown(heading, "book.pdf")
        assert f"\n### {heading}\n" in md

    def test_kanji_numeral_word_is_not_heading(self):
        md = format_text_to_markdown("一人で歩いた。", "book.pdf")
        assert "###" not in md


def test_english_with_some_accents_does_not_need_ocr():
    """アクセント付きの文字を一部に含む英文は文字化けとみなさない"""
    assert needs_ocr("The café served crème brûlée and a fine rosé to the naïve guests.") is False


class TestConvertRouting:
    """ページごとにテキスト層とOCRを振り分ける"""

    GOOD = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"

    def _convert(self, pages, mode, ocr_result="OCRの結果です。"):
        from pdfocr.pipeline import convert
        calls = []

        def fake_render(pdf_path, page_indexes, dpi, out_dir):
            return [out_dir / f"page_{i + 1:04d}.png" for i in page_indexes]

        def fake_ocr(image_paths, work_dir):
            calls.append([p.name for p in image_paths])
            return {p: f"{ocr_result}({p.stem})" for p in image_paths}

        result = convert(
            Path("in.pdf"), ocr_mode=mode,
            extract=lambda _path: pages, render=fake_render, run_ocr=fake_ocr,
        )
        return result, calls

    def test_auto_ocrs_only_pages_without_usable_text(self):
        result, calls = self._convert([self.GOOD, "", self.GOOD], "auto")
        assert calls == [["page_0002.png"]]
        assert result == [self.GOOD, "OCRの結果です。(page_0002)", self.GOOD]

    def test_auto_without_any_bad_page_never_runs_ocr(self):
        _result, calls = self._convert([self.GOOD], "auto")
        assert calls == []

    def test_always_ocrs_every_page(self):
        result, calls = self._convert([self.GOOD, self.GOOD], "always")
        assert calls == [["page_0001.png", "page_0002.png"]]
        assert result[0].startswith("OCRの結果です。")

    def test_never_uses_text_layer_only(self):
        result, calls = self._convert([self.GOOD, ""], "never")
        assert calls == []
        assert result == [self.GOOD, ""]


class TestCli:
    """コマンドラインの動作"""

    GOOD = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"

    @pytest.fixture
    def pdf(self, tmp_path):
        path = tmp_path / "book.pdf"
        path.write_bytes(b"%PDF-1.4")
        return path

    def test_existing_output_is_untouched_before_extraction(self, pdf, monkeypatch):
        from pdfocr import cli
        output = pdf.with_suffix(".txt")
        output.write_text("old", encoding="utf-8")
        monkeypatch.setattr(cli, "convert", lambda *a, **k: pytest.fail("抽出が実行された"))
        assert cli.main([str(pdf)]) == 1
        assert output.read_text(encoding="utf-8") == "old"

    def test_force_overwrites_and_writes_text(self, pdf, monkeypatch):
        from pdfocr import cli
        output = pdf.with_suffix(".txt")
        output.write_text("old", encoding="utf-8")
        monkeypatch.setattr(cli, "convert", lambda *a, **k: [self.GOOD, self.GOOD, "続き"])
        assert cli.main([str(pdf), "--force"]) == 0
        # 連続する重複ページは除去される
        assert output.read_text(encoding="utf-8") == f"{self.GOOD}\n\n続き"

    def test_markdown_output(self, pdf, monkeypatch):
        from pdfocr import cli
        monkeypatch.setattr(cli, "convert", lambda *a, **k: ["第一章　はじまり\n本 文﹁です﹂。"])
        assert cli.main([str(pdf), "--format", "md"]) == 0
        md = pdf.with_suffix(".md").read_text(encoding="utf-8")
        assert "## 第一章　はじまり" in md
        assert "本文「です」。" in md

    def test_empty_result_fails_without_writing(self, pdf, monkeypatch):
        from pdfocr import cli
        monkeypatch.setattr(cli, "convert", lambda *a, **k: ["", "  "])
        assert cli.main([str(pdf)]) == 1
        assert not pdf.with_suffix(".txt").exists()

    def test_setup_dispatches_to_engine(self, monkeypatch):
        from pdfocr import cli
        called = []
        monkeypatch.setattr(cli, "setup_ndl", lambda update=False: called.append(("ndl", update)))
        assert cli.main(["setup", "ndl", "--update"]) == 0
        assert called == [("ndl", True)]


    def test_setup_dispatches_to_indra(self, monkeypatch):
        from pdfocr import cli
        called = []
        monkeypatch.setattr(cli, "setup_indra", lambda update=False: called.append(("indra", update)))
        assert cli.main(["setup", "indra"]) == 0
        assert called == [("indra", False)]

    def test_engine_indra_uses_indra_runner(self, pdf, monkeypatch):
        from pdfocr import cli
        used = []

        def fake_convert(*_a, run_ocr, **_k):
            used.append(run_ocr)
            return [self.GOOD]

        monkeypatch.setattr(cli, "convert", fake_convert)
        assert cli.main([str(pdf), "--engine", "indra"]) == 0
        assert used == [cli.run_indra]


class TestRemoveSpacesBetweenJapanese:
    """1文字ずつ配置されたテキスト層で入る、日本語の文字間の半角スペースを除去する"""

    def test_spaces_between_japanese_chars_are_removed(self):
        from pdfocr.postprocess import remove_spaces_between_japanese
        assert remove_spaces_between_japanese("吾 輩 は 猫 で あ る 。") == "吾輩は猫である。"

    def test_spaces_around_latin_words_are_kept(self):
        from pdfocr.postprocess import remove_spaces_between_japanese
        assert remove_spaces_between_japanese("日本語と English text を含む") == "日本語と English text を含む"

    def test_fullwidth_space_is_kept(self):
        """節見出し（一　はじめに）の判定に使う全角スペースは残す"""
        from pdfocr.postprocess import remove_spaces_between_japanese
        assert remove_spaces_between_japanese("一　はじめに") == "一　はじめに"


class TestMarkdownLineJoin:
    """段落内の行をつなぐとき、日本語どうしは空白を入れない"""

    def test_japanese_lines_are_joined_without_space(self):
        md = format_text_to_markdown("名前はまだ無い。\nどこで生れたか", "book.pdf")
        assert "名前はまだ無い。どこで生れたか" in md

    def test_english_lines_are_joined_with_space(self):
        md = format_text_to_markdown("The quick brown\nfox jumps", "book.pdf")
        assert "The quick brown fox jumps" in md

"""google_tts モジュールのテスト"""

from __future__ import annotations

import sys
from pathlib import Path

# python ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestSplitTextForTts:
    """テキスト分割のテスト"""

    def test_short_text_no_split(self):
        """短いテキストは分割されない"""
        from google_tts import split_text_for_tts

        text = "こんにちは"
        result = split_text_for_tts(text, max_bytes=4800)
        assert result == [text]

    def test_empty_text_returns_empty_list(self):
        """空文字は空リストを返す"""
        from google_tts import split_text_for_tts

        result = split_text_for_tts("", max_bytes=4800)
        assert result == []

    def test_split_at_period(self):
        """句点で分割される"""
        from google_tts import split_text_for_tts

        # 「あ」は UTF-8 で 3 バイト
        # 「。」は UTF-8 で 3 バイト
        # max_bytes=30 で、約10文字まで
        text = "あいうえお。かきくけこ。さしすせそ。"
        result = split_text_for_tts(text, max_bytes=30)

        # 分割されている
        assert len(result) > 1
        # 各チャンクは max_bytes 以下
        for chunk in result:
            assert len(chunk.encode("utf-8")) <= 30
        # 結合すると元のテキストになる
        assert "".join(result) == text

    def test_split_at_newline_priority(self):
        """改行は句点より優先して分割される"""
        from google_tts import split_text_for_tts

        text = "段落1です\n段落2です\n段落3です"
        result = split_text_for_tts(text, max_bytes=30)

        # 改行で分割されている
        assert len(result) > 1
        # 各チャンクは改行を含まない（または末尾のみ）
        for chunk in result:
            assert len(chunk.encode("utf-8")) <= 30

    def test_split_at_exclamation(self):
        """感嘆符で分割される"""
        from google_tts import split_text_for_tts

        text = "すごい！びっくり！やった！"
        result = split_text_for_tts(text, max_bytes=24)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk.encode("utf-8")) <= 24

    def test_split_at_question(self):
        """疑問符で分割される"""
        from google_tts import split_text_for_tts

        text = "なぜ？どうして？いつ？"
        result = split_text_for_tts(text, max_bytes=18)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk.encode("utf-8")) <= 18

    def test_split_at_comma_as_fallback(self):
        """句点がない場合は読点で分割される"""
        from google_tts import split_text_for_tts

        text = "これは長い文で、途中に読点があり、最後まで句点がない文です"
        result = split_text_for_tts(text, max_bytes=60)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk.encode("utf-8")) <= 60

    def test_preserve_text_integrity(self):
        """分割・結合後に元のテキストが復元できる"""
        from google_tts import split_text_for_tts

        text = "こんにちは。さようなら。また会いましょう。元気でね。"
        result = split_text_for_tts(text, max_bytes=30)

        assert "".join(result) == text

    def test_long_single_sentence_raises_error(self):
        """分割不可能な長い文はエラー"""
        from google_tts import split_text_for_tts

        # 分割点のない長い文字列
        text = "あ" * 100  # 300 バイト
        with pytest.raises(ValueError, match="分割できません"):
            split_text_for_tts(text, max_bytes=90)

    def test_mixed_ascii_and_japanese(self):
        """ASCII と日本語が混在するテキストを正しく処理"""
        from google_tts import split_text_for_tts

        text = "Hello World。こんにちは。ABC。日本語。"
        result = split_text_for_tts(text, max_bytes=30)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk.encode("utf-8")) <= 30
        assert "".join(result) == text

    def test_real_world_text_size(self):
        """実際の使用ケース: 5000バイト以上のテキスト"""
        from google_tts import split_text_for_tts

        # 約 6000 バイトのテキスト（日本語 2000 文字）
        sentences = ["これはテスト文です。"] * 200
        text = "".join(sentences)

        result = split_text_for_tts(text, max_bytes=4800)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk.encode("utf-8")) <= 4800
        assert "".join(result) == text


class TestMergeAudioFiles:
    """音声ファイル結合のテスト"""

    def test_merge_two_files(self, tmp_path):
        """2つの音声ファイルが結合される"""
        from pydub import AudioSegment
        from pydub.generators import Sine

        from google_tts import merge_audio_files

        # テスト用の短い音声ファイルを生成
        audio1 = Sine(440).to_audio_segment(duration=100)  # 100ms の 440Hz
        audio2 = Sine(880).to_audio_segment(duration=100)  # 100ms の 880Hz

        path1 = tmp_path / "audio1.mp3"
        path2 = tmp_path / "audio2.mp3"
        output_path = tmp_path / "merged.mp3"

        audio1.export(str(path1), format="mp3")
        audio2.export(str(path2), format="mp3")

        result = merge_audio_files([path1, path2], output_path, "mp3")

        assert result == output_path
        assert output_path.exists()

        # 結合後の長さを確認（約200ms）
        merged = AudioSegment.from_mp3(str(output_path))
        assert 150 < len(merged) < 250  # 許容誤差を含む

    def test_empty_list_raises_error(self, tmp_path):
        """空のリストはエラー"""
        from google_tts import merge_audio_files

        output_path = tmp_path / "output.mp3"
        with pytest.raises(ValueError, match="音声ファイルが指定されていません"):
            merge_audio_files([], output_path, "mp3")

    def test_single_file_copies(self, tmp_path):
        """1ファイルの場合はコピーされる"""
        from pydub.generators import Sine

        from google_tts import merge_audio_files

        audio = Sine(440).to_audio_segment(duration=100)
        path1 = tmp_path / "audio1.mp3"
        output_path = tmp_path / "output.mp3"

        audio.export(str(path1), format="mp3")

        result = merge_audio_files([path1], output_path, "mp3")

        assert result == output_path
        assert output_path.exists()

    def test_merge_wav_files(self, tmp_path):
        """WAV ファイルの結合"""
        from pydub import AudioSegment
        from pydub.generators import Sine

        from google_tts import merge_audio_files

        audio1 = Sine(440).to_audio_segment(duration=100)
        audio2 = Sine(880).to_audio_segment(duration=100)

        path1 = tmp_path / "audio1.wav"
        path2 = tmp_path / "audio2.wav"
        output_path = tmp_path / "merged.wav"

        audio1.export(str(path1), format="wav")
        audio2.export(str(path2), format="wav")

        result = merge_audio_files([path1, path2], output_path, "wav")

        assert result == output_path
        assert output_path.exists()

        merged = AudioSegment.from_wav(str(output_path))
        assert 150 < len(merged) < 250

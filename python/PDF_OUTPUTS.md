# ローカルPDF出力の保存

`pdf_to_text.py`、`pdf_ocr.py`、`pdf_to_markdown.py` は既存の出力を既定では上書きしません。
保存は同じディレクトリ内の一時ファイルから原子的に公開します。入力のハードリンクも出力先として拒否します。
明示的に置き換える場合だけ `--force` を指定します。Markdownの入力PDF自体を出力先には指定できません。

```sh
cd python
uv run python pdf_to_text.py input.pdf
uv run python pdf_ocr.py input.pdf --force
uv run python pdf_to_markdown.py input.pdf -o output.md --force
```

抽出に失敗した場合は終了コード1です。OCRはページの一部が失敗した場合も失敗とし、不完全な新規結果を保存しません。
テキストが全く抽出できなかった場合も既存結果を空ファイルで置き換えません。
API校正を指定する場合も、既存出力の確認はAPI呼び出しより先に行います。

これらの変更はCloud OCRや音声処理には適用していません。

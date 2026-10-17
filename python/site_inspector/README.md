# Site Inspector

企業ドメインを対象にサイト構造とコンテンツを収集し、レポートを生成する CLI ツールです。

## 使い方

```bash
uv sync
uv run site-inspector https://example.com --depth 2 --max-pages 50 --download-images --output-dir reports/latest
```

## 主な機能

- robots.txt と sitemap.xml を考慮したクロール計画の自動生成
- ページタイトル・見出し・メタ情報・本文サマリの抽出
- 画像タグ収集および属性情報の記録（画像ダウンロードはオプション）
- JSON / CSV / Markdown レポートの出力
- マークダウンと `structure.json` による階層的なサイト構造の可視化
- レートリミット・クロール上限・除外URLパターンの設定

詳細は `resources/configs/sample.yaml` を参照してください。

## クロール結果と制限

- `max_pages` は並列処理中・取得失敗を含むページ試行数の上限です。失敗は各ページの `error` に記録し、残りのページを処理します。
- 本文の `word_count` は全文の空白区切り語数、`text_preview` は最大1200文字です。
- `data:` 等のHTTP(S)以外の画像は `src` を保存し、`absolute_url` は空とします。ダウンロード対象にはなりません。
- `playwright` 設定は将来用で、現在はJavaScriptレンダリングに対応していません。

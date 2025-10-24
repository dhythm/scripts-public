# Site Inspector

企業ドメインを対象にサイト構造とコンテンツを収集し、レポートを生成する CLI ツールです。

## 使い方

```bash
uv sync
uv run site-inspector crawl https://example.com --depth 2 --output-dir reports/latest
```

## 主な機能

- robots.txt と sitemap.xml を考慮したクロール計画の自動生成
- ページタイトル・見出し・メタ情報・本文サマリの抽出
- 画像タグ収集および属性情報の記録（画像ダウンロードはオプション）
- JSON / CSV / Markdown レポートの出力
- レートリミット・クロール上限・除外URLパターンの設定

詳細は `resources/configs/sample.yaml` を参照してください。

# Node TypeScript ツール集

`node-ts` ディレクトリには Node.js + TypeScript で動作するユーティリティを配置しています。

## 一次情報ハーベスタ (`primary-source-harvester.ts`)

OpenAI Responses API の `web_search` ツールと自前の `pdf_search` 関数を組み合わせて、指定キーワードに関する一次情報・二次情報を一定数集めるツールです。PDF のみをピンポイントで探す場合は Bing の `filetype:pdf` 検索を行い、公式ドメインを優先します。

### 前提

- Node.js 18 以上
- 依存関係インストール: `npm install`
- 環境変数 `OPENAI_API_KEY` を設定

### 使い方

```bash
# 例: 2 つのキーワードを調査し、結果を JSON に保存
npm run primary-harvest -- \
  --keyword "生成AI ガバナンス" \
  --keyword "EV バッテリー" \
  --primary-min 3 \
  --secondary-min 2 \
  --output reports/primary-sources.json
```

主なオプション:

| オプション | 説明 |
| --- | --- |
| `-k, --keyword <text>` | 調査するキーワード。複数指定可。位置引数でも指定可能 |
| `--primary-min <n>` | 目標とする一次情報件数 (既定: 2) |
| `--secondary-min <n>` | 目標とする二次情報件数 (既定: 2) |
| `--pdf-limit <n>` | 1 回の PDF検索で取得する最大件数 (既定: 5、最大10) |
| `--max-passes <n>` | モデルに許可するツール呼び出しループの上限 (既定: 6) |
| `-o, --output <path>` | 結果を保存するファイルパス。省略時は標準出力に JSON を表示 |
| `--country/--region/--city/--timezone` | `web_search` に渡す利用者位置 (任意) |
| `--debug` | モデル出力やツール呼び出しを詳細ログとして表示し、`reports/debug/` に生レスポンスを保存 |

### 出力フォーマット

実行結果は以下の JSON 形式です。

```json
{
  "generatedAt": "2025-01-31T12:34:56.000Z",
  "reports": [
    {
      "keyword": "生成AI ガバナンス",
      "summary": "...",
      "stats": { "primaryCount": 3, "secondaryCount": 2 },
      "sources": [
        {
          "classification": "primary",
          "title": "...",
          "url": "https://...",
          "summary": "...",
          "whyTrusted": "公式発表",
          "retrievalMethod": "web_search"
        }
      ]
    }
  ]
}
```

`sources` には各リンクの分類・説明・信頼根拠が含まれるため、そのままレポート資料の一次情報メモとして利用できます。

### デバッグログ

`--debug` を指定すると、実行中のツール呼び出し内容を `[debug]` 付きで標準出力に表示し、モデルから返った生テキストを `reports/debug/<slug>-response-<timestamp>.txt` に保存します。JSON 解析に失敗した場合は `--debug` の有無にかかわらず `reports/debug/<slug>-parse-error-...` が残るので、モデル出力を直接確認して原因調査ができます。

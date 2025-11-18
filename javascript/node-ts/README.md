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

## ソース逆引きアナライザー (`source-origin-analyzer.ts`)

特定URLの本文を解析して「一次/二次/三次/その他」を判定するソースチェック機能と、必要に応じて一次・二次情報を逆引きする機能を組み合わせたツールです。三次情報と判断された場合は、OpenAI Responses API の `web_search` ツールを用いて根拠となる公式発表や信頼できる報道を追加で探索し、引用元の概要と信頼理由を整理します。モデルは `gpt-5-mini`（Responses API, `responses.create`）を利用します。

### 前提

- Node.js 18 以上
- `npm install` 済み
- 環境変数 `OPENAI_API_KEY`

### 使い方

```bash
# URLの一次/二次/三次判定（JSON表示 & ファイル保存）
npm run source-origin -- \
  --url "https://example.com/case-study" \
  --json \
  --output reports/source-origin-example.json

# web_search用の推定ロケーションを渡す例
npm run source-origin -- \
  --url "https://news.example.com/article" \
  --country JP --region Tokyo --timezone Asia/Tokyo
```

主なオプション:

| オプション | 説明 |
| --- | --- |
| `--url <URL>` | 判定対象のURL (必須) |
| `--output <path>` | 結果JSONの保存先 |
| `--json` | 判定結果をJSONで標準出力に表示 |
| `--max-chars <n>` | LLMに渡す本文文字数の上限 (既定: 8000) |
| `--debug` | 取得メタデータを表示 |
| `--country/--region/--city/--timezone` | `web_search` へ渡す推定利用者位置 |

標準出力では分類・理由・要約・主要主体を日本語で表示し、三次情報だった場合のみ逆引きした補完ソース一覧を追記します。`--json` を付ければAPIレスポンスをそのままログ保管や後続処理へ回せます。


## Text-to-Speech (`text-to-speech/`)

OpenAI Text-to-Speech API を使用して日本語テキストを音声ファイルに変換するツールです。

### 前提

- Node.js 18 以上
- 依存関係インストール: `npm install`
- 環境変数 `OPENAI_API_KEY` を設定

### 使い方

```bash
# テキストを直接指定して音声生成
npm run tts -- -t "こんにちは、世界" -o output.mp3

# ファイルから読み込み
npm run tts -- -i input.txt -o output.mp3

# 高品質モデルとカスタム音声を使用
npm run tts -- -t "Hello, World" -o output.mp3 -m tts-1-hd -v alloy

# 速度を変更
npm run tts -- -t "こんにちは" -o output.mp3 -s 1.5

# ヘルプを表示
npm run tts -- --help
```

主なオプション:

| オプション | 説明 |
| --- | --- |
| `-t, --text <text>` | 音声に変換するテキスト |
| `-i, --input <file>` | テキストファイルのパス |
| `-o, --output <file>` | 出力音声ファイルのパス (必須) |
| `-v, --voice <voice>` | 音声の種類 (デフォルト: nova)<br>選択肢: alloy, echo, fable, onyx, nova, shimmer |
| `-m, --model <model>` | 使用するモデル (デフォルト: tts-1)<br>選択肢: tts-1, tts-1-hd |
| `-f, --format <format>` | 出力フォーマット (デフォルト: mp3)<br>選択肢: mp3, opus, aac, flac, wav, pcm |
| `-s, --speed <speed>` | 再生速度 (デフォルト: 1.0, 範囲: 0.25-4.0) |
| `-h, --help` | ヘルプを表示 |

### 音声の種類

- `alloy` - 中性的な声
- `echo` - 男性的な声
- `fable` - イギリス英語の男性的な声
- `onyx` - 深みのある男性的な声
- `nova` - 女性的な声（デフォルト、日本語に適している）
- `shimmer` - 明るい女性的な声

### プログラムから使用する

```typescript
import OpenAI from "openai";
import { textToSpeech } from "./text-to-speech/index.js";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const result = await textToSpeech(
  client,
  "こんにちは、世界",
  "output.mp3",
  {
    voice: "nova",
    model: "tts-1",
    format: "mp3",
    speed: 1.0,
  }
);

console.log(`音声ファイルを生成しました: ${result.outputPath}`);
console.log(`ファイルサイズ: ${result.sizeBytes} bytes`);
```

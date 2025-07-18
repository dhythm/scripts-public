# PDF Text Check Tool

PDFファイルにテキストレイヤーが含まれているかを確認するツールです。

## 概要

2つのバージョンがあります：
- `check-pdf-text.ts` - pdf-libライブラリを使用した詳細な解析
- `check-pdf-text-simple.ts` - 軽量版（外部ライブラリ不要）

## 使用方法

### pdf-libバージョン
```bash
deno task check-pdf <PDFファイルパス>
```

### 軽量バージョン
```bash
deno task check-pdf-simple <PDFファイルパス>
```

## 出力例

```
PDFファイル: sample.pdf
ページ数: 10
テキスト層あり
```

または

```
PDFファイル: scanned.pdf
ページ数: 5
画像のみ（OCR が必要）
```

## 用途

- PDFファイルがOCR処理が必要かどうかの事前確認
- テキスト抽出可能なPDFの識別
- PDFファイルの簡易検証
# PDF構造分析ガイド

## 概要

Vision APIとDocument AIは、PDFからテキストを抽出するだけでなく、文書の構造情報も提供します。この構造情報を活用することで、より高度な文書解析が可能になります。

## 利用可能な構造情報

### Vision API (DOCUMENT_TEXT_DETECTION)

Vision APIのfullTextAnnotationには以下の階層構造が含まれます：

```
fullTextAnnotation
├── text (全体のテキスト)
└── pages[]
    ├── property (言語検出など)
    ├── width, height (ページサイズ)
    └── blocks[]
        ├── blockType (TEXT, TABLE, PICTURE, RULER, BARCODE)
        ├── boundingBox (位置情報)
        ├── confidence (信頼度)
        └── paragraphs[]
            ├── boundingBox
            ├── confidence
            └── words[]
                ├── boundingBox
                ├── confidence
                └── symbols[]
                    ├── text (文字)
                    ├── boundingBox
                    └── confidence
```

### Document AI

Document AIはより詳細な構造情報を提供します：

```
document
├── text (全体のテキスト)
├── pages[]
│   ├── pageNumber
│   ├── dimension (幅、高さ、単位)
│   ├── layout (レイアウト情報)
│   ├── detectedLanguages[]
│   ├── blocks[] (ブロック単位)
│   ├── paragraphs[] (段落単位)
│   ├── lines[] (行単位)
│   ├── tokens[] (単語/トークン単位)
│   ├── tables[] (テーブル構造)
│   │   ├── headerRows[]
│   │   └── bodyRows[]
│   └── formFields[] (フォームフィールド)
│       ├── fieldName
│       └── fieldValue
└── entities[] (エンティティ抽出)
    ├── type
    ├── mentionText
    └── confidence
```

## 構造情報の活用例

### 1. 階層的なテキスト抽出

ブロック、段落、単語の階層構造を利用して、文書の論理構造を保持したまま抽出：

- 見出しと本文の区別
- 段落ごとの処理
- リスト項目の識別

### 2. レイアウト分析

バウンディングボックス情報を使用して：

- 2カラムレイアウトの検出
- サイドバーやヘッダー/フッターの識別
- 図表の位置特定

### 3. テーブルデータの抽出

構造化されたテーブル情報から：

- CSVやExcel形式への変換
- データベースへの格納
- 表形式データの分析

### 4. フォーム処理

フォームフィールドの自動抽出：

- 申請書の自動処理
- アンケート結果の集計
- 契約書の重要項目抽出

### 5. 多言語対応

言語検出機能を使用して：

- 言語別の処理
- 翻訳の必要性判断
- 多言語文書の分割

## 使用方法

### 構造分析ツール

PDFの構造を分析してJSON形式で保存：

```bash
# Vision APIを使用
deno run --allow-read --allow-write --allow-env --allow-net structure-analysis.ts ./document.pdf

# Document AIを使用
deno run --allow-read --allow-write --allow-env --allow-net structure-analysis.ts ./document.pdf --document-ai
```

### 構造化データ抽出ツール

保存された構造情報から、整形されたデータを抽出：

```bash
# Markdown形式で出力（デフォルト）
deno run --allow-read --allow-write structured-extraction.ts ./document_vision_api_structure.json

# JSON形式で出力
deno run --allow-read --allow-write structured-extraction.ts ./document_vision_api_structure.json --format=json

# テーブルをCSV形式で出力
deno run --allow-read --allow-write structured-extraction.ts ./document_document_ai_structure.json --format=csv
```

## 実装のポイント

### 1. テキストの再構築

Document AIでは、テキストは全体で1つの文字列として提供され、各要素（段落、単語など）はtextAnchorで位置を指定します：

```typescript
// textAnchorからテキストを取得
function getTextFromLayout(documentText: string, layout: any): string {
  const segments = layout.textAnchor.textSegments.map(segment => {
    const start = parseInt(segment.startIndex);
    const end = parseInt(segment.endIndex);
    return documentText.substring(start, end);
  });
  return segments.join('');
}
```

### 2. バウンディングボックスの正規化

Vision APIとDocument AIでは座標系が異なる場合があります：

- Vision API: ピクセル座標
- Document AI: 正規化座標（0-1の範囲）またはピクセル座標

### 3. 信頼度の活用

各要素には信頼度（confidence）が付与されています。低い信頼度の要素は：

- 人間による確認が必要
- 代替の処理方法を検討
- エラーハンドリングの実装

### 4. ブロックタイプの判定

Vision APIのblockTypeを活用：

- TEXT: 通常のテキストブロック
- TABLE: テーブル（ただし詳細構造は提供されない）
- PICTURE: 画像
- RULER: 罫線
- BARCODE: バーコード

### 5. パフォーマンスの考慮

大きなPDFファイルの場合：

- ページ単位での処理
- 必要な情報のみを抽出
- メモリ使用量の監視

## 高度な活用例

### 1. セマンティック検索

構造情報を使って、より正確な検索を実現：

```typescript
// 見出しのみを検索
const headings = doc.pages.flatMap(page => 
  page.sections.filter(s => s.type === 'heading')
);

// 特定のテーブルのデータを検索
const tableData = doc.pages.flatMap(page => 
  page.tables?.map(table => table.data) || []
);
```

### 2. 文書の要約

構造を理解した要約生成：

- 各セクションの最初の段落を抽出
- 見出しの階層構造を保持
- 重要なテーブルやフォームデータを含める

### 3. データ検証

フォームやテーブルのデータ検証：

- 必須フィールドの確認
- データ型の検証
- 計算値の確認

## まとめ

Vision APIとDocument AIの構造情報を活用することで、単純なテキスト抽出を超えた高度な文書処理が可能になります。用途に応じて適切なAPIを選択し、提供される構造情報を最大限に活用してください。
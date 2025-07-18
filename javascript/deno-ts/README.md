# Deno TypeScript プロジェクト

## PDF テキスト診断スクリプト

PDF ファイルにテキスト層が含まれているかを確認するスクリプトを3種類用意しています。

### 1. pdf-lib 版 (check-pdf-text.ts)

シンプルな実装ですが、精度は限定的です。

```bash
# 直接実行
deno run --allow-read src/check-pdf-text.ts sample.pdf

# タスク経由で実行
deno task check-pdf sample.pdf
```

### 2. PDF.js 版 (check-pdf-text-pdfjs.ts)

より正確にテキストコンテンツを検出します。各ページのテキスト文字数も表示します。
※ 環境によってはキャンバス依存関係のエラーが発生する場合があります。

```bash
# 直接実行
deno run --allow-read --allow-net src/check-pdf-text-pdfjs.ts sample.pdf

# タスク経由で実行
deno task check-pdf-pdfjs sample.pdf
```

### 3. 軽量版 (check-pdf-text-simple.ts)

外部依存なしで PDF の生データを解析します。エラーが発生しない安定版です。

```bash
# 直接実行
deno run --allow-read src/check-pdf-text-simple.ts sample.pdf

# タスク経由で実行
deno task check-pdf-simple sample.pdf
```

## 機能

- コマンドライン引数で PDF ファイルパスを指定
- PDF のページ数を表示
- テキスト層の有無を判定
- PDF.js 版では各ページのテキスト文字数も表示

## エラーハンドリング

- ファイルが存在しない場合のエラー表示
- PDF 以外のファイルを指定した場合のエラー表示
- 引数未指定時の使用方法表示
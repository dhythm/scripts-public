# JavaScript/TypeScript 実行環境

このディレクトリには Node.js と Deno の両方の TypeScript 実行環境が含まれています。

## Node.js TypeScript 環境

### セットアップ
```bash
cd node-ts
npm install
```

### 実行コマンド
- 開発モード（ホットリロード付き）: `npm run dev`
- ビルド: `npm run build`
- プロダクション実行: `npm start`
- 型チェック: `npm run typecheck`

### 特徴
- tsx による高速な TypeScript 実行
- strict モード有効
- ES2022 ターゲット
- Node.js エコシステムとの完全な互換性

## Deno TypeScript 環境

### セットアップ
Deno がインストールされていない場合:
```bash
curl -fsSL https://deno.land/install.sh | sh
```

### 実行コマンド
- 開発モード（ホットリロード付き）: `deno task dev`
- 実行: `deno task start`
- フォーマット: `deno task fmt`
- リント: `deno task lint`
- 型チェック: `deno task check`

### 特徴
- TypeScript をネイティブサポート
- セキュアなデフォルト設定（権限ベース）
- 標準ライブラリ搭載
- URL インポート対応

## 使い分けの指針

### Node.js を選ぶ場合
- npm パッケージを使用したい
- 既存の Node.js プロジェクトとの互換性が必要
- Express.js などのフレームワークを使用する

### Deno を選ぶ場合
- シンプルな設定で始めたい
- セキュリティを重視する
- モダンな TypeScript 開発を行いたい
- 標準ライブラリだけで完結させたい
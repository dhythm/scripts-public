#!/bin/bash

# 使用法: ./copy-env-files.sh <SOURCE_DIR> <DEST_DIR>
# 例: ./copy-env-files.sh ./main-repo ./worktrees/dev-branch

set -e

SRC_DIR="$1"
DEST_DIR="$2"

if [[ -z "$SRC_DIR" || -z "$DEST_DIR" ]]; then
  echo "Usage: $0 <source_dir> <dest_dir>"
  exit 1
fi

# 実体の絶対パスを取得
SRC_DIR="$(cd "$SRC_DIR" && pwd)"
DEST_DIR="$(cd "$DEST_DIR" && pwd)"

echo "Copying .env files from $SRC_DIR to $DEST_DIR ..."

# 再帰的に .env* を探索し、同じパス構造でコピー
find "$SRC_DIR" \( -name "node_modules" -o -name ".git" \) -prune -o -type f -name ".env*" -print | while read -r src_file; do
  # 相対パスを取得
  rel_path="${src_file#$SRC_DIR/}"
  dest_file="$DEST_DIR/$rel_path"

  # 必要なディレクトリを作成
  mkdir -p "$(dirname "$dest_file")"

  # コピー
  cp "$src_file" "$dest_file"

  echo "Copied: $rel_path -> $dest_file"
done

echo "✅ Done."

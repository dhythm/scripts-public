/**
 * PDFファイルのページ数を推定する
 * PDFファイルの内部構造を解析して、/Type/Page オブジェクトの数をカウントする
 */
export function estimatePdfPageCount(pdfData: Uint8Array): number {
  try {
    // PDFデータを文字列に変換（バイナリデータの一部をテキストとして解析）
    const decoder = new TextDecoder("latin1");
    const pdfText = decoder.decode(pdfData);

    // /Type /Page パターンを探す（PDFページオブジェクトのマーカー）
    const pagePattern = /\/Type\s*\/Page(?![s/])/g;
    const matches = pdfText.match(pagePattern);
    
    if (matches && matches.length > 0) {
      console.debug(`PDFページ数推定: ${matches.length}ページ`);
      return matches.length;
    }

    // /Page パターンが見つからない場合は、/Count を探す
    const countPattern = /\/Count\s+(\d+)/g;
    let maxCount = 0;
    let match;
    
    while ((match = countPattern.exec(pdfText)) !== null) {
      const count = parseInt(match[1], 10);
      if (count > maxCount) {
        maxCount = count;
      }
    }
    
    if (maxCount > 0) {
      console.debug(`PDFページ数推定（/Countから）: ${maxCount}ページ`);
      return maxCount;
    }

    // ページ数が特定できない場合は1を返す
    console.warn("PDFページ数を特定できませんでした。デフォルト値1を使用します。");
    return 1;
  } catch (error) {
    console.error("PDFページ数の推定中にエラーが発生しました:", error);
    return 1;
  }
}

/**
 * ページ範囲に基づいてバッチを作成する
 * @param totalPages 総ページ数
 * @param batchSize バッチサイズ（デフォルト: 5）
 * @returns バッチの配列（各バッチはページ番号の配列）
 */
export function createPageBatches(totalPages: number, batchSize: number = 5): number[][] {
  const batches: number[][] = [];
  
  for (let startPage = 1; startPage <= totalPages; startPage += batchSize) {
    const endPage = Math.min(startPage + batchSize - 1, totalPages);
    const batch = Array.from(
      { length: endPage - startPage + 1 }, 
      (_, i) => startPage + i
    );
    batches.push(batch);
  }
  
  return batches;
}
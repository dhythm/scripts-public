// 軽量版 PDF テキスト検出スクリプト
// PDF ファイルの生データを解析してテキストストリームを検出

async function checkPdfText(filePath: string): Promise<void> {
  try {
    const data = await Deno.readFile(filePath);
    const text = new TextDecoder('latin1').decode(data);
    
    console.log(`PDFファイル: ${filePath}`);
    console.log(`ファイルサイズ: ${(data.length / 1024).toFixed(2)} KB`);
    
    // PDF ヘッダーの確認
    if (!text.startsWith('%PDF-')) {
      console.error('エラー: 有効な PDF ファイルではありません');
      Deno.exit(1);
    }
    
    // テキスト描画コマンドの検出
    // BT...ET (テキストブロック)
    // Tj, TJ (テキスト表示)
    // Td, TD, Tm (テキスト位置)
    const textBlockPattern = /BT[\s\S]*?ET/g;
    const textBlocks = text.match(textBlockPattern) || [];
    
    // テキスト表示コマンドのカウント
    let tjCount = 0;
    let tjMatches = 0;
    
    for (const block of textBlocks) {
      // Tj コマンド（単純テキスト表示）
      const tjCommands = block.match(/\([^)]*\)\s*Tj/g) || [];
      tjCount += tjCommands.length;
      
      // TJ コマンド（配列テキスト表示）
      const tjArrayCommands = block.match(/\[[\s\S]*?\]\s*TJ/g) || [];
      tjMatches += tjArrayCommands.length;
    }
    
    // ストリームオブジェクトの検出
    const streamPattern = /stream[\r\n]+([\s\S]*?)[\r\n]+endstream/g;
    const streams = text.match(streamPattern) || [];
    
    // テキストコンテンツの推定
    let hasTextContent = false;
    let extractedText = '';
    
    // 簡易的なテキスト抽出（括弧内の文字列）
    const textPattern = /\(([^)]+)\)\s*Tj/g;
    let match;
    while ((match = textPattern.exec(text)) !== null) {
      const content = match[1];
      // エスケープシーケンスの簡易処理
      const decoded = content
        .replace(/\\n/g, '\n')
        .replace(/\\r/g, '\r')
        .replace(/\\t/g, '\t')
        .replace(/\\\(/g, '(')
        .replace(/\\\)/g, ')');
      
      if (decoded.trim().length > 0) {
        hasTextContent = true;
        extractedText += decoded + ' ';
      }
    }
    
    console.log('---');
    console.log(`テキストブロック数: ${textBlocks.length}`);
    console.log(`Tj コマンド数: ${tjCount}`);
    console.log(`TJ コマンド数: ${tjMatches}`);
    console.log(`ストリーム数: ${streams.length}`);
    
    if (hasTextContent && extractedText.length > 0) {
      console.log(`抽出されたテキストの一部: ${extractedText.substring(0, 100)}...`);
    }
    
    console.log('---');
    console.log(hasTextContent || tjCount > 0 || tjMatches > 0 
      ? 'テキスト層あり' 
      : '画像のみ（OCR が必要）');
    
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) {
      console.error(`エラー: ファイルが見つかりません: ${filePath}`);
    } else if (error instanceof Error) {
      console.error(`エラー: ${error.message}`);
    } else {
      console.error('不明なエラーが発生しました');
    }
    Deno.exit(1);
  }
}

// メイン処理
if (import.meta.main) {
  const args = Deno.args;
  
  if (args.length === 0) {
    console.error('使用方法: deno run --allow-read check-pdf-text-simple.ts <PDFファイルパス>');
    Deno.exit(1);
  }
  
  const filePath = args[0];
  await checkPdfText(filePath);
}
import { PDFDocument } from "https://cdn.skypack.dev/pdf-lib@1.17.1";

async function checkPdfText(filePath: string): Promise<void> {
  try {
    const bytes = await Deno.readFile(filePath);
    const doc = await PDFDocument.load(bytes);
    
    const pageCount = doc.getPageCount();
    console.log(`PDFファイル: ${filePath}`);
    console.log(`ページ数: ${pageCount}`);
    
    let hasText = false;
    
    for (let i = 0; i < pageCount; i++) {
      const page = doc.getPage(i);
      
      // pdf-lib では直接テキストコンテンツを取得できないため、
      // ページのコンテンツストリームを確認
      const operators = page.node.Contents();
      
      if (operators) {
        const contents = operators.toString();
        // テキスト描画オペレータの存在を確認
        // BT (Begin Text), ET (End Text), Tj (Show Text), TJ (Show Text Array)
        if (contents.includes('BT') && contents.includes('ET') && 
            (contents.includes('Tj') || contents.includes('TJ'))) {
          hasText = true;
          break;
        }
      }
    }
    
    console.log(hasText ? 'テキスト層あり' : '画像のみ（OCR が必要）');
    
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
    console.error('使用方法: deno run --allow-read check-pdf-text.ts <PDFファイルパス>');
    Deno.exit(1);
  }
  
  const filePath = args[0];
  await checkPdfText(filePath);
}
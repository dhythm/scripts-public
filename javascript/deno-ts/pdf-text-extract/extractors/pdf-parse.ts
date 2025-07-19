import { TextExtractor, TextExtractionResult } from "../types/index.ts";
import pdf from "npm:pdf-parse@1.1.1";

export class PdfParseExtractor implements TextExtractor {
  async extractText(pdfPath: string): Promise<TextExtractionResult> {
    try {
      const buffer = await Deno.readFile(pdfPath);
      return await this.extractTextFromBuffer(buffer);
    } catch (error) {
      return {
        text: "",
        successful: false,
        error: error instanceof Error ? error : new Error(String(error)),
        hasTextLayer: false,
      };
    }
  }

  async extractTextFromBuffer(buffer: Uint8Array): Promise<TextExtractionResult> {
    try {
      // pdf-parseにバッファを渡して解析
      const data = await pdf(buffer);
      
      // テキストが空でないかチェック
      const hasTextLayer = data.text.trim().length > 0;
      
      if (!hasTextLayer) {
        return {
          text: "",
          successful: true,
          hasTextLayer: false,
          pageCount: data.numpages,
        };
      }

      return {
        text: data.text,
        pageCount: data.numpages,
        successful: true,
        hasTextLayer: true,
      };
    } catch (error) {
      // PDFが破損している場合やテキスト層がない場合
      if (error instanceof Error && 
          (error.message.includes("No text") || 
           error.message.includes("Invalid PDF"))) {
        return {
          text: "",
          successful: true,
          hasTextLayer: false,
        };
      }

      return {
        text: "",
        successful: false,
        error: error instanceof Error ? error : new Error(String(error)),
        hasTextLayer: false,
      };
    }
  }
}
import { TextExtractor, TextExtractionResult } from "../types/index.ts";
import { pdfToText } from "npm:pdf-ts@0.0.2";
import { Buffer } from "npm:buffer@6.0.3";

export class PdfTsExtractor implements TextExtractor {
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
      // pdf-tsはBufferを期待するため、Uint8ArrayをBufferに変換
      const nodeBuffer = Buffer.from(buffer);
      
      // pdf-tsでテキスト抽出
      const text = await pdfToText(nodeBuffer);
      
      // テキストが空でないかチェック
      const hasTextLayer = text.trim().length > 0;
      
      if (!hasTextLayer) {
        return {
          text: "",
          successful: true,
          hasTextLayer: false,
        };
      }

      // ページ数の推定（改ページ文字をカウント）
      const pageBreaks = (text.match(/\f/g) || []).length;
      const pageCount = pageBreaks > 0 ? pageBreaks + 1 : undefined;

      return {
        text,
        pageCount,
        successful: true,
        hasTextLayer: true,
      };
    } catch (error) {
      // PDFが破損している場合やテキスト層がない場合
      if (error instanceof Error && error.message.includes("No text content")) {
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
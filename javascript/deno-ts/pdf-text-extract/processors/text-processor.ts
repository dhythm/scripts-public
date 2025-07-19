import { TextExtractor, PdfFileInfo, TextExtractionResult, Logger } from "../types/index.ts";
import { basename } from "std/path/mod.ts";

export interface ProcessingResult {
  fileInfo: PdfFileInfo;
  extractionResult: TextExtractionResult;
  savedPath?: string;
}

export class PdfTextProcessor {
  constructor(
    private extractor: TextExtractor,
    private logger: Logger
  ) {}

  async processFile(fileInfo: PdfFileInfo, saveToFile: boolean = true): Promise<ProcessingResult> {
    const fileName = basename(fileInfo.path);
    this.logger.info(`処理開始: ${fileName}`);

    try {
      // テキスト抽出
      const extractionResult = await this.extractor.extractText(fileInfo.path);

      if (!extractionResult.successful) {
        this.logger.error(`抽出失敗: ${fileName}`, extractionResult.error);
        return { fileInfo, extractionResult };
      }

      if (!extractionResult.hasTextLayer) {
        this.logger.warn(`テキスト層なし: ${fileName} (画像のみのPDFの可能性があります)`);
        return { fileInfo, extractionResult };
      }

      // ファイルに保存
      if (saveToFile && extractionResult.text) {
        const savedPath = fileInfo.outputPath;
        await this.saveText(extractionResult.text, savedPath);
        
        this.logger.info(
          `完了: ${fileName} → ${basename(savedPath)} ` +
          `(${extractionResult.pageCount ? `${extractionResult.pageCount}ページ` : "ページ数不明"})`
        );

        return { fileInfo, extractionResult, savedPath };
      }

      return { fileInfo, extractionResult };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.logger.error(`処理エラー: ${fileName} - ${errorMessage}`);
      
      return {
        fileInfo,
        extractionResult: {
          text: "",
          successful: false,
          error: error instanceof Error ? error : new Error(errorMessage),
          hasTextLayer: false,
        },
      };
    }
  }

  private async saveText(text: string, outputPath: string): Promise<void> {
    const encoder = new TextEncoder();
    await Deno.writeFile(outputPath, encoder.encode(text));
  }

  cleanupText(text: string): string {
    // テキストのクリーンアップ処理
    return text
      .replace(/\r\n/g, "\n")  // 改行コードの統一
      .replace(/\f/g, "\n\n--- ページ区切り ---\n\n")  // 改ページ文字を見やすく
      .replace(/\n{3,}/g, "\n\n")  // 過剰な改行を削減
      .trim();
  }
}
import { TextExtractionResult, CloudOcrClient } from "../types/index.ts";
import { Logger } from "../types/index.ts";
import { basename } from "std/path/mod.ts";

export class PdfTextProcessor {
  constructor(
    private client: CloudOcrClient,
    private logger: Logger
  ) {}

  async processFile(filePath: string): Promise<TextExtractionResult> {
    const fileName = basename(filePath);
    this.logger.info(`処理開始: ${fileName}`);

    const startTime = performance.now();

    try {
      const text = await this.client.extractTextFromPdf(filePath);
      const pageCount = this.estimatePageCount(text);
      const processingTime = performance.now() - startTime;

      this.logger.info(
        `処理完了: ${fileName} (${processingTime.toFixed(0)}ms, ${pageCount}ページ)`
      );

      return {
        text: this.formatExtractedText(text),
        pageCount,
        successful: true,
      };
    } catch (error) {
      const processingTime = performance.now() - startTime;
      this.logger.error(
        `処理失敗: ${fileName} (${processingTime.toFixed(0)}ms)`,
        error as Error
      );

      return {
        text: "",
        pageCount: 0,
        successful: false,
        error: error as Error,
      };
    }
  }

  private formatExtractedText(text: string): string {
    if (!text) return "";

    return text
      .replace(/\r\n/g, "\n")
      .replace(/\r/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  private estimatePageCount(text: string): number {
    if (!text) return 0;

    const pageBreaks = text.match(/\f/g);
    if (pageBreaks) {
      return pageBreaks.length + 1;
    }

    const lines = text.split("\n").length;
    const estimatedLinesPerPage = 50;
    return Math.max(1, Math.ceil(lines / estimatedLinesPerPage));
  }
}
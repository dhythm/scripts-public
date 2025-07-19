import { AppConfig, ProcessingSummary, ProcessingError, TextExtractor, PdfFileInfo } from "./types/index.ts";
import { ConsoleLogger } from "./utils/logger.ts";
import { DenoFileSystemHandler } from "./handlers/file-system.ts";
import { PdfTsExtractor } from "./extractors/pdf-ts.ts";
import { PdfParseExtractor } from "./extractors/pdf-parse.ts";
import { PdfTextProcessor, ProcessingResult } from "./processors/text-processor.ts";
import { ProgressTracker, ConcurrentProcessor } from "./utils/progress.ts";
import { basename, join } from "std/path/mod.ts";

export class PdfTextExtractorApp {
  private logger: ConsoleLogger;
  private fileHandler: DenoFileSystemHandler;
  private extractor: TextExtractor;
  private processor: PdfTextProcessor;
  private startTime: number = 0;
  private extractedTexts: Map<string, string> = new Map();

  constructor(private config: AppConfig) {
    this.logger = new ConsoleLogger(config.verbose);
    this.fileHandler = new DenoFileSystemHandler();
    this.extractor = this.createExtractor();
    this.processor = new PdfTextProcessor(this.extractor, this.logger);
  }

  private createExtractor(): TextExtractor {
    switch (this.config.libraryType) {
      case "pdf-parse":
        return new PdfParseExtractor();
      case "pdf-ts":
      default:
        return new PdfTsExtractor();
    }
  }

  async run(): Promise<ProcessingSummary> {
    this.startTime = performance.now();
    const summary: ProcessingSummary = {
      totalFiles: 0,
      successfulFiles: 0,
      failedFiles: 0,
      skippedFiles: 0,
      totalTimeMs: 0,
      errors: [],
    };

    try {
      this.logger.info("PDF テキスト抽出ツールを開始します");
      this.logger.info(`入力ディレクトリ: ${this.config.inputDir}`);
      this.logger.info(`出力ディレクトリ: ${this.config.outputDir || "入力ディレクトリと同じ"}`);
      this.logger.info(`使用ライブラリ: ${this.config.libraryType}`);
      this.logger.info(`並行処理数: ${this.config.concurrency || 3}`);
      if (this.config.merge) {
        this.logger.info("マージモード: 有効");
      }

      const pdfFiles = await this.scanForPdfFiles();
      summary.totalFiles = pdfFiles.length;

      if (pdfFiles.length === 0) {
        this.logger.warn("PDFファイルが見つかりませんでした");
        return this.finalizeSummary(summary);
      }

      this.logger.info(`${pdfFiles.length} 個の PDF ファイルが見つかりました`);

      // ファイル情報を取得
      const fileInfos = await this.getFileInfos(pdfFiles);

      // プログレストラッカーを初期化
      const progressTracker = new ProgressTracker(fileInfos.length);

      // 並行処理でテキスト抽出
      const concurrentProcessor = new ConcurrentProcessor<PdfFileInfo, ProcessingResult>(
        this.config.concurrency || 3
      );

      const { results, errors } = await concurrentProcessor.process(
        fileInfos,
        async (fileInfo) => {
          const result = await this.processor.processFile(fileInfo, !this.config.merge);
          
          // マージモードの場合はメモリに保存
          if (this.config.merge && result.extractionResult.text) {
            const cleanedText = this.processor.cleanupText(result.extractionResult.text);
            this.extractedTexts.set(fileInfo.path, cleanedText);
          }
          
          return result;
        },
        (fileInfo, result) => {
          progressTracker.increment();
          
          if (result instanceof Error) {
            summary.failedFiles++;
            summary.errors.push({
              filePath: fileInfo.path,
              error: result.message,
              timestamp: new Date(),
            });
          } else {
            const processingResult = result as ProcessingResult;
            if (processingResult.extractionResult.successful) {
              if (processingResult.extractionResult.hasTextLayer) {
                summary.successfulFiles++;
              } else {
                summary.skippedFiles++;
              }
            } else {
              summary.failedFiles++;
              summary.errors.push({
                filePath: fileInfo.path,
                error: processingResult.extractionResult.error?.message || "不明なエラー",
                timestamp: new Date(),
              });
            }
          }
        }
      );

      // エラーの処理
      for (const { item, error } of errors) {
        summary.errors.push({
          filePath: item.path,
          error: error.message,
          timestamp: new Date(),
        });
      }

      // マージモードの場合、すべてのテキストを1つのファイルに保存
      if (this.config.merge && this.extractedTexts.size > 0) {
        await this.saveMergedOutput();
      }

      return this.finalizeSummary(summary);

    } catch (error) {
      this.logger.error("予期しないエラーが発生しました", error);
      summary.errors.push({
        filePath: "システム",
        error: error instanceof Error ? error.message : String(error),
        timestamp: new Date(),
      });
      return this.finalizeSummary(summary);
    }
  }

  private async scanForPdfFiles(): Promise<string[]> {
    try {
      return await this.fileHandler.scanDirectory(
        this.config.inputDir,
        this.config.filePattern
      );
    } catch (error) {
      this.logger.error("ディレクトリスキャンエラー", error);
      throw error;
    }
  }

  private async getFileInfos(pdfFiles: string[]): Promise<PdfFileInfo[]> {
    const fileInfos: PdfFileInfo[] = [];
    
    for (const filePath of pdfFiles) {
      try {
        const fileInfo = await this.fileHandler.getPdfFileInfo(
          filePath,
          this.config.outputDir
        );
        fileInfos.push(fileInfo);
      } catch (error) {
        this.logger.error(`ファイル情報取得エラー: ${filePath}`, error);
      }
    }
    
    return fileInfos;
  }

  private async saveMergedOutput(): Promise<void> {
    const outputDir = this.config.outputDir || this.config.inputDir;
    const outputPath = join(outputDir, "merged_output.txt");
    
    // ファイルパスでソートしてから結合
    const sortedPaths = Array.from(this.extractedTexts.keys()).sort();
    const mergedTexts: string[] = [];
    
    for (const path of sortedPaths) {
      const text = this.extractedTexts.get(path);
      if (text) {
        mergedTexts.push(`=== ${basename(path)} ===\n\n${text}`);
      }
    }
    
    const finalText = mergedTexts.join(this.config.mergeSeparator || "\n\n========================================\n\n");
    
    try {
      await this.fileHandler.saveTextToFile(finalText, outputPath);
      this.logger.info(`マージされたテキストを保存しました: ${outputPath}`);
    } catch (error) {
      this.logger.error("マージファイルの保存に失敗しました", error);
      throw error;
    }
  }

  private finalizeSummary(summary: ProcessingSummary): ProcessingSummary {
    summary.totalTimeMs = performance.now() - this.startTime;
    
    console.log("\n====================");
    console.log("処理完了");
    console.log("====================");
    console.log(`合計ファイル数: ${summary.totalFiles}`);
    console.log(`成功: ${summary.successfulFiles}`);
    console.log(`スキップ（テキスト層なし）: ${summary.skippedFiles}`);
    console.log(`失敗: ${summary.failedFiles}`);
    console.log(`処理時間: ${(summary.totalTimeMs / 1000).toFixed(2)} 秒`);
    
    if (summary.errors.length > 0) {
      console.log("\nエラー詳細:");
      for (const error of summary.errors) {
        console.log(`- ${basename(error.filePath)}: ${error.error}`);
      }
    }
    
    return summary;
  }
}
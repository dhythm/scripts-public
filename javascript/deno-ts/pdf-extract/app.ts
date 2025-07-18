import { AppConfig, ProcessingSummary, ProcessingError, CloudOcrClient, PdfFileInfo } from "./types/index.ts";
import { ConsoleLogger } from "./utils/logger.ts";
import { DenoFileSystemHandler } from "./handlers/file-system.ts";
import { VisionApiClientV2 } from "./clients/vision-api-v2.ts";
import { DocumentAiClient } from "./clients/document-ai.ts";
import { PdfTextProcessor } from "./processors/pdf-text.ts";
import { ProgressTracker, ConcurrentProcessor } from "./utils/progress.ts";
import { basename, join } from "std/path/mod.ts";

export class PdfTextExtractor {
  private logger: ConsoleLogger;
  private fileHandler: DenoFileSystemHandler;
  private ocrClient: CloudOcrClient;
  private processor: PdfTextProcessor;
  private startTime: number = 0;

  constructor(private config: AppConfig) {
    this.logger = new ConsoleLogger(config.verbose);
    this.fileHandler = new DenoFileSystemHandler();
    this.ocrClient = this.createOcrClient();
    this.processor = new PdfTextProcessor(this.ocrClient, this.logger);
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
      this.logger.info(`使用API: ${this.config.apiType.toUpperCase()}`);
      this.logger.info(`並行処理数: ${this.config.concurrency || 3}`);

      await this.authenticateClient();

      const pdfFiles = await this.scanForPdfFiles();
      summary.totalFiles = pdfFiles.length;

      if (pdfFiles.length === 0) {
        this.logger.warn("PDFファイルが見つかりませんでした");
        return this.finalizeSummary(summary);
      }

      this.logger.info(`${pdfFiles.length} 個のPDFファイルが見つかりました`);

      const results = await this.processFiles(pdfFiles, summary);
      this.updateSummaryFromResults(summary, results);

      // マージオプションが有効な場合、すべてのテキストファイルを結合
      if (this.config.merge && summary.successfulFiles > 0) {
        await this.mergeTextFiles(pdfFiles, results);
      }

    } catch (error) {
      this.logger.error("致命的なエラーが発生しました", error as Error);
      summary.errors.push({
        filePath: "システム",
        error: (error as Error).message,
        timestamp: new Date(),
      });
    }

    return this.finalizeSummary(summary);
  }

  private createOcrClient(): CloudOcrClient {
    switch (this.config.apiType) {
      case "documentai":
        return new DocumentAiClient();
      case "vision":
      default:
        return new VisionApiClientV2();
    }
  }

  private async authenticateClient(): Promise<void> {
    this.logger.info("Google Cloud 認証を開始します");
    try {
      await this.ocrClient.authenticate();
      this.logger.info("認証に成功しました");
    } catch (error) {
      this.logger.error("認証に失敗しました", error as Error);
      throw error;
    }
  }

  private async scanForPdfFiles(): Promise<PdfFileInfo[]> {
    this.logger.debug("PDFファイルをスキャンしています...");
    
    const filePaths = await this.fileHandler.scanDirectory(
      this.config.inputDir,
      this.config.filePattern
    );

    const fileInfos: PdfFileInfo[] = [];
    for (const filePath of filePaths) {
      const info = await this.fileHandler.getPdfFileInfo(
        filePath,
        this.config.outputDir
      );
      fileInfos.push(info);
    }

    return fileInfos;
  }

  private async processFiles(
    files: PdfFileInfo[],
    _summary: ProcessingSummary
  ): Promise<Map<PdfFileInfo, ProcessingError | null>> {
    const progressTracker = new ProgressTracker(files.length);
    const results = new Map<PdfFileInfo, ProcessingError | null>();

    const processor = new ConcurrentProcessor(
      files,
      async (file) => await this.processSingleFile(file, progressTracker),
      this.config.concurrency || 3,
      (completed, total) => {
        if (completed === total) {
          progressTracker.finish();
        }
      }
    );

    const processingResults = await processor.processAll();

    for (const [file, result] of processingResults) {
      results.set(file, result);
    }

    return results;
  }

  private async processSingleFile(
    fileInfo: PdfFileInfo,
    progressTracker: ProgressTracker
  ): Promise<ProcessingError | null> {
    const fileName = basename(fileInfo.path);
    progressTracker.startFile(fileName);

    try {
      const result = await this.processor.processFile(fileInfo.path);

      if (!result.successful) {
        throw result.error || new Error("テキスト抽出に失敗しました");
      }

      if (!result.text || result.text.trim().length === 0) {
        this.logger.warn(`${fileName}: テキストが検出されませんでした（画像のみのPDFの可能性があります）`);
        throw new Error("テキストが検出されませんでした（画像のみのPDFの可能性があります）");
      }

      await this.fileHandler.saveTextToFile(result.text, fileInfo.outputPath);
      
      this.logger.debug(
        `保存完了: ${fileInfo.outputPath} (${result.pageCount}ページ, ${result.text.length}文字)`
      );

      progressTracker.completeFile();
      return null;

    } catch (error) {
      progressTracker.completeFile();
      const errorMessage = (error as Error).message;
      this.logger.error(`処理エラー: ${fileName}`, error as Error);
      
      return {
        filePath: fileInfo.path,
        error: errorMessage,
        timestamp: new Date(),
      };
    }
  }

  private updateSummaryFromResults(
    summary: ProcessingSummary,
    results: Map<PdfFileInfo, ProcessingError | null>
  ): void {
    for (const [_, error] of results) {
      if (error) {
        summary.failedFiles++;
        summary.errors.push(error);
      } else {
        summary.successfulFiles++;
      }
    }
  }

  private finalizeSummary(summary: ProcessingSummary): ProcessingSummary {
    summary.totalTimeMs = performance.now() - this.startTime;
    
    this.logger.info("\n===== 処理サマリー =====");
    this.logger.info(`総ファイル数: ${summary.totalFiles}`);
    this.logger.info(`成功: ${summary.successfulFiles}`);
    this.logger.info(`失敗: ${summary.failedFiles}`);
    this.logger.info(`スキップ: ${summary.skippedFiles}`);
    this.logger.info(`処理時間: ${(summary.totalTimeMs / 1000).toFixed(2)}秒`);

    if (summary.errors.length > 0) {
      this.logger.warn("\n処理中のエラー:");
      for (const error of summary.errors) {
        this.logger.warn(`- ${basename(error.filePath)}: ${error.error}`);
      }
    }

    return summary;
  }

  private async mergeTextFiles(
    files: PdfFileInfo[],
    results: Map<PdfFileInfo, ProcessingError | null>
  ): Promise<void> {
    this.logger.info("\nテキストファイルのマージを開始します...");
    
    const successfulFiles: PdfFileInfo[] = [];
    for (const [file, error] of results) {
      if (!error) {
        successfulFiles.push(file);
      }
    }

    // ファイル名でソート（アルファベット順）
    successfulFiles.sort((a, b) => basename(a.path).localeCompare(basename(b.path)));

    const mergedTexts: string[] = [];
    const separator = this.config.mergeSeparator || "\n\n========================================\n\n";

    for (const file of successfulFiles) {
      try {
        const text = await Deno.readTextFile(file.outputPath);
        const fileName = basename(file.path);
        
        // ファイル名をヘッダーとして追加
        mergedTexts.push(`=== ${fileName} ===\n\n${text}`);
      } catch (error) {
        this.logger.error(`ファイルの読み取りに失敗: ${file.outputPath}`, error as Error);
      }
    }

    if (mergedTexts.length > 0) {
      const mergedContent = mergedTexts.join(separator);
      const outputDir = this.config.outputDir || this.config.inputDir;
      const mergedFilePath = join(outputDir, "merged_output.txt");
      
      try {
        await Deno.writeTextFile(mergedFilePath, mergedContent);
        this.logger.info(`マージファイルを作成しました: ${mergedFilePath}`);
        this.logger.info(`${successfulFiles.length} 個のファイルをマージしました`);
      } catch (error) {
        this.logger.error("マージファイルの作成に失敗しました", error as Error);
      }
    }
  }
}
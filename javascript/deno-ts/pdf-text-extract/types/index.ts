export interface AppConfig {
  inputDir: string;
  outputDir?: string;
  libraryType: "pdf-ts" | "pdf-parse";
  filePattern?: string;
  concurrency?: number;
  verbose?: boolean;
  merge?: boolean;
  mergeSeparator?: string;
}

export interface ProcessingSummary {
  totalFiles: number;
  successfulFiles: number;
  failedFiles: number;
  skippedFiles: number;
  totalTimeMs: number;
  errors: ProcessingError[];
}

export interface ProcessingError {
  filePath: string;
  error: string;
  timestamp: Date;
}

export interface PdfFileInfo {
  path: string;
  size: number;
  lastModified: Date;
  outputPath: string;
}

export interface TextExtractionResult {
  text: string;
  pageCount?: number;
  successful: boolean;
  error?: Error;
  hasTextLayer: boolean;
}

export interface TextExtractor {
  extractText(pdfPath: string): Promise<TextExtractionResult>;
  extractTextFromBuffer(buffer: Uint8Array): Promise<TextExtractionResult>;
}

export interface FileSystemHandler {
  scanDirectory(dir: string, pattern?: string): Promise<string[]>;
  saveTextToFile(text: string, outputPath: string): Promise<void>;
  getPdfFileInfo(filePath: string, outputDir?: string): Promise<PdfFileInfo>;
  readPdfFile(filePath: string): Promise<Uint8Array>;
  formatFileSize(bytes: number): string;
}

export enum LogLevel {
  DEBUG,
  INFO,
  WARN,
  ERROR,
}

export interface Logger {
  log(level: LogLevel, message: string, data?: unknown): void;
  debug(message: string, data?: unknown): void;
  info(message: string, data?: unknown): void;
  warn(message: string, data?: unknown): void;
  error(message: string, data?: unknown): void;
}
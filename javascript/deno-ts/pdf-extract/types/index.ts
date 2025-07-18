export interface AppConfig {
  inputDir: string;
  outputDir?: string;
  apiType: "vision" | "documentai";
  filePattern?: string;
  concurrency?: number;
  verbose?: boolean;
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
  pageCount: number;
  successful: boolean;
  error?: Error;
}

export interface CloudOcrClient {
  authenticate(): Promise<void>;
  extractTextFromPdf(pdfPath: string): Promise<string>;
}

export interface FileSystemHandler {
  scanDirectory(dir: string, pattern?: string): Promise<string[]>;
  saveTextToFile(text: string, outputPath: string): Promise<void>;
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
  error(message: string, error?: Error): void;
}
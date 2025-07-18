import { FileSystemHandler, PdfFileInfo } from "../types/index.ts";
import { walk } from "std/fs/walk.ts";
import { join, dirname, basename } from "std/path/mod.ts";
import { ensureDir } from "std/fs/ensure_dir.ts";

export class DenoFileSystemHandler implements FileSystemHandler {
  async scanDirectory(dir: string, pattern?: string): Promise<string[]> {
    const pdfFiles: string[] = [];
    const globPattern = pattern || "*.pdf";

    try {
      for await (const entry of walk(dir, {
        includeDirs: false,
        match: [new RegExp(globPattern.replace(/\*/g, ".*").replace(/\?/g, "."), "i")],
      })) {
        if (entry.path.toLowerCase().endsWith(".pdf")) {
          pdfFiles.push(entry.path);
        }
      }
    } catch (error) {
      if (error instanceof Deno.errors.NotFound) {
        throw new Error(`ディレクトリが見つかりません: ${dir}`);
      }
      throw error;
    }

    return pdfFiles.sort();
  }

  async saveTextToFile(text: string, outputPath: string): Promise<void> {
    try {
      const dir = dirname(outputPath);
      await ensureDir(dir);
      
      const encoder = new TextEncoder();
      await Deno.writeFile(outputPath, encoder.encode(text));
    } catch (error) {
      if (error instanceof Deno.errors.PermissionDenied) {
        throw new Error(`ファイルの書き込み権限がありません: ${outputPath}`);
      }
      throw error;
    }
  }

  async getPdfFileInfo(filePath: string, outputDir?: string): Promise<PdfFileInfo> {
    const stat = await Deno.stat(filePath);
    
    const baseDir = outputDir || dirname(filePath);
    const fileName = basename(filePath, ".pdf");
    const outputPath = join(baseDir, `${fileName}.txt`);

    return {
      path: filePath,
      size: stat.size,
      lastModified: stat.mtime || new Date(),
      outputPath,
    };
  }

  async readPdfFile(filePath: string): Promise<Uint8Array> {
    try {
      return await Deno.readFile(filePath);
    } catch (error) {
      if (error instanceof Deno.errors.NotFound) {
        throw new Error(`PDFファイルが見つかりません: ${filePath}`);
      }
      if (error instanceof Deno.errors.PermissionDenied) {
        throw new Error(`PDFファイルの読み取り権限がありません: ${filePath}`);
      }
      throw error;
    }
  }

  formatFileSize(bytes: number): string {
    const units = ["B", "KB", "MB", "GB"];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(2)} ${units[unitIndex]}`;
  }
}
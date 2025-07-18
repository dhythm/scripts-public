export class ProgressTracker {
  private startTime: number;
  private processedFiles = 0;
  private totalFiles: number;
  private currentFile?: string;
  private lastUpdate = 0;
  private updateInterval = 100; // ms

  constructor(totalFiles: number) {
    this.totalFiles = totalFiles;
    this.startTime = performance.now();
  }

  startFile(fileName: string): void {
    this.currentFile = fileName;
    this.updateProgress();
  }

  completeFile(): void {
    this.processedFiles++;
    this.currentFile = undefined;
    this.updateProgress(true);
  }

  private updateProgress(force = false): void {
    const now = performance.now();
    if (!force && now - this.lastUpdate < this.updateInterval) {
      return;
    }

    this.lastUpdate = now;
    const percentage = Math.round((this.processedFiles / this.totalFiles) * 100);
    const elapsed = (now - this.startTime) / 1000;
    const filesPerSecond = this.processedFiles / elapsed;
    const remainingFiles = this.totalFiles - this.processedFiles;
    const estimatedRemaining = remainingFiles / filesPerSecond;

    this.clearLine();
    const progressBar = this.createProgressBar(percentage);
    const timeInfo = this.formatTime(estimatedRemaining);
    const fileInfo = this.currentFile ? ` 処理中: ${this.currentFile}` : "";
    
    Deno.stdout.writeSync(
      new TextEncoder().encode(
        `${progressBar} ${this.processedFiles}/${this.totalFiles} (${percentage}%) ETA: ${timeInfo}${fileInfo}`
      )
    );
  }

  finish(): void {
    this.clearLine();
    const totalTime = (performance.now() - this.startTime) / 1000;
    console.log(
      `完了: ${this.processedFiles}/${this.totalFiles} ファイル (${this.formatTime(totalTime)})`
    );
  }

  private createProgressBar(percentage: number): string {
    const width = 30;
    const filled = Math.round((width * percentage) / 100);
    const empty = width - filled;
    return `[${"\u2588".repeat(filled)}${"\u2591".repeat(empty)}]`;
  }

  private formatTime(seconds: number): string {
    if (!isFinite(seconds) || seconds < 0) {
      return "--:--";
    }

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  }

  private clearLine(): void {
    Deno.stdout.writeSync(new TextEncoder().encode("\r\x1b[2K"));
  }
}

export class ConcurrentProcessor<T, R> {
  private queue: T[] = [];
  private active = 0;
  private results: Map<T, R> = new Map();

  constructor(
    private items: T[],
    private processor: (item: T) => Promise<R>,
    private concurrency: number,
    private onProgress?: (completed: number, total: number) => void
  ) {
    this.queue = [...items];
  }

  async processAll(): Promise<Map<T, R>> {
    const promises: Promise<void>[] = [];

    for (let i = 0; i < Math.min(this.concurrency, this.queue.length); i++) {
      promises.push(this.processNext());
    }

    await Promise.all(promises);
    return this.results;
  }

  private async processNext(): Promise<void> {
    while (this.queue.length > 0) {
      const item = this.queue.shift();
      if (!item) break;

      this.active++;
      
      try {
        const result = await this.processor(item);
        this.results.set(item, result);
      } catch (error) {
        this.results.set(item, error as R);
      } finally {
        this.active--;
        if (this.onProgress) {
          this.onProgress(this.results.size, this.items.length);
        }
      }
    }
  }
}
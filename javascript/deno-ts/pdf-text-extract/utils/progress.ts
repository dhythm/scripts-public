export class ProgressTracker {
  private processed = 0;
  private total: number;
  private startTime: number;

  constructor(total: number) {
    this.total = total;
    this.startTime = Date.now();
  }

  increment(): void {
    this.processed++;
    this.displayProgress();
  }

  private displayProgress(): void {
    const percentage = Math.round((this.processed / this.total) * 100);
    const elapsed = Date.now() - this.startTime;
    const avgTime = elapsed / this.processed;
    const remaining = avgTime * (this.total - this.processed);
    
    const elapsedStr = this.formatTime(elapsed);
    const remainingStr = this.formatTime(remaining);
    
    console.log(
      `処理中: ${this.processed}/${this.total} (${percentage}%) ` +
      `経過時間: ${elapsedStr} 残り時間: ${remainingStr}`
    );
  }

  private formatTime(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}時間${minutes % 60}分`;
    } else if (minutes > 0) {
      return `${minutes}分${seconds % 60}秒`;
    } else {
      return `${seconds}秒`;
    }
  }
}

export class ConcurrentProcessor<T, R> {
  private concurrency: number;
  private queue: T[] = [];
  private processing = 0;
  private results: R[] = [];
  private errors: Array<{ item: T; error: Error }> = [];

  constructor(concurrency: number = 3) {
    this.concurrency = Math.max(1, Math.min(10, concurrency));
  }

  async process(
    items: T[],
    processor: (item: T) => Promise<R>,
    onProgress?: (item: T, result: R | Error) => void
  ): Promise<{ results: R[]; errors: Array<{ item: T; error: Error }> }> {
    this.queue = [...items];
    this.results = [];
    this.errors = [];
    this.processing = 0;

    const workers = Array(this.concurrency)
      .fill(null)
      .map(() => this.worker(processor, onProgress));

    await Promise.all(workers);

    return { results: this.results, errors: this.errors };
  }

  private async worker(
    processor: (item: T) => Promise<R>,
    onProgress?: (item: T, result: R | Error) => void
  ): Promise<void> {
    while (this.queue.length > 0) {
      const item = this.queue.shift();
      if (!item) break;

      this.processing++;
      try {
        const result = await processor(item);
        this.results.push(result);
        if (onProgress) onProgress(item, result);
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        this.errors.push({ item, error: err });
        if (onProgress) onProgress(item, err);
      } finally {
        this.processing--;
      }
    }
  }
}
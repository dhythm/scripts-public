export class RetryableError extends Error {
  constructor(message: string, public readonly retryable = true) {
    super(message);
    this.name = "RetryableError";
  }
}

export interface RetryOptions {
  maxRetries: number;
  initialDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
}

export class RetryHandler {
  private readonly defaultOptions: RetryOptions = {
    maxRetries: 3,
    initialDelayMs: 1000,
    maxDelayMs: 30000,
    backoffMultiplier: 2,
  };

  async withRetry<T>(
    operation: () => Promise<T>,
    options?: Partial<RetryOptions>
  ): Promise<T> {
    const opts = { ...this.defaultOptions, ...options };
    let lastError: Error | undefined;
    let delayMs = opts.initialDelayMs;

    for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error as Error;

        if (!this.isRetryable(error) || attempt === opts.maxRetries) {
          throw error;
        }

        console.warn(
          `リトライ ${attempt + 1}/${opts.maxRetries}: ${lastError.message}`
        );

        await this.delay(delayMs);
        delayMs = Math.min(delayMs * opts.backoffMultiplier, opts.maxDelayMs);
      }
    }

    throw lastError || new Error("不明なエラー");
  }

  private isRetryable(error: unknown): boolean {
    if (error instanceof RetryableError) {
      return error.retryable;
    }

    if (error instanceof Error) {
      const message = error.message.toLowerCase();
      
      const retryablePatterns = [
        "rate limit",
        "quota exceeded", 
        "timeout",
        "network",
        "connection",
        "temporary",
        "unavailable",
        "503",
        "429",
        "500",
        "502",
        "504",
      ];

      return retryablePatterns.some(pattern => message.includes(pattern));
    }

    return false;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

export class RateLimiter {
  private queue: Array<() => void> = [];
  private running = 0;

  constructor(
    private readonly maxConcurrent: number,
    private readonly minDelayMs: number = 100
  ) {}

  async acquire(): Promise<void> {
    if (this.running >= this.maxConcurrent) {
      await new Promise<void>(resolve => {
        this.queue.push(resolve);
      });
    }
    this.running++;
    await this.delay(this.minDelayMs);
  }

  release(): void {
    this.running--;
    const next = this.queue.shift();
    if (next) {
      next();
    }
  }

  async withLimit<T>(operation: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await operation();
    } finally {
      this.release();
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
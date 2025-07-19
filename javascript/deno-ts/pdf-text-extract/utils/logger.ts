import { Logger, LogLevel } from "../types/index.ts";

export class ConsoleLogger implements Logger {
  constructor(private verbose: boolean = false) {}

  log(level: LogLevel, message: string, data?: unknown): void {
    if (!this.verbose && level === LogLevel.DEBUG) {
      return;
    }

    const timestamp = new Date().toISOString();
    const levelStr = LogLevel[level];
    const prefix = `[${timestamp}] [${levelStr}]`;

    switch (level) {
      case LogLevel.ERROR:
        console.error(`${prefix} ${message}`, data ?? "");
        break;
      case LogLevel.WARN:
        console.warn(`${prefix} ${message}`, data ?? "");
        break;
      default:
        console.log(`${prefix} ${message}`, data ?? "");
    }
  }

  debug(message: string, data?: unknown): void {
    this.log(LogLevel.DEBUG, message, data);
  }

  info(message: string, data?: unknown): void {
    this.log(LogLevel.INFO, message, data);
  }

  warn(message: string, data?: unknown): void {
    this.log(LogLevel.WARN, message, data);
  }

  error(message: string, data?: unknown): void {
    this.log(LogLevel.ERROR, message, data);
  }
}
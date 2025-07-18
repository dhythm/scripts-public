import { Logger, LogLevel } from "../types/index.ts";

export class ConsoleLogger implements Logger {
  private minLevel: LogLevel;
  private useColors: boolean;

  constructor(verbose = false) {
    this.minLevel = verbose ? LogLevel.DEBUG : LogLevel.INFO;
    this.useColors = !Deno.noColor;
  }

  log(level: LogLevel, message: string, data?: unknown): void {
    if (level < this.minLevel) return;

    const timestamp = new Date().toISOString();
    const levelStr = this.getLevelString(level);
    const coloredLevel = this.colorize(levelStr, level);
    
    const logMessage = `[${timestamp}] ${coloredLevel} ${message}`;
    
    switch (level) {
      case LogLevel.ERROR:
        console.error(logMessage);
        break;
      case LogLevel.WARN:
        console.warn(logMessage);
        break;
      default:
        console.log(logMessage);
    }

    if (data !== undefined) {
      console.log(JSON.stringify(data, null, 2));
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

  error(message: string, error?: Error): void {
    this.log(LogLevel.ERROR, message);
    if (error) {
      console.error(error.stack || error.message);
    }
  }

  private getLevelString(level: LogLevel): string {
    switch (level) {
      case LogLevel.DEBUG:
        return "DEBUG";
      case LogLevel.INFO:
        return "INFO ";
      case LogLevel.WARN:
        return "WARN ";
      case LogLevel.ERROR:
        return "ERROR";
      default:
        return "UNKNOWN";
    }
  }

  private colorize(text: string, level: LogLevel): string {
    if (!this.useColors) return text;

    const colors = {
      [LogLevel.DEBUG]: "\x1b[90m", // Gray
      [LogLevel.INFO]: "\x1b[36m",  // Cyan
      [LogLevel.WARN]: "\x1b[33m",  // Yellow
      [LogLevel.ERROR]: "\x1b[31m", // Red
    };

    const reset = "\x1b[0m";
    return `${colors[level]}${text}${reset}`;
  }
}
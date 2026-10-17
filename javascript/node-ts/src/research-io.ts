import { accessSync, constants, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname } from "node:path";

/** Fail on an unusable destination before spending API calls. Existing report files remain replaceable. */
export function prepareOutputPath(outputPath?: string): void {
  if (!outputPath) return;
  const parent = dirname(outputPath);
  mkdirSync(parent, { recursive: true });
  if (existsSync(outputPath)) {
    if (!statSync(outputPath).isFile()) throw new Error(`出力先は通常ファイルではありません: ${outputPath}`);
    accessSync(outputPath, constants.W_OK);
  }
  accessSync(parent, constants.W_OK);
}

/** Bound the entire request and streamed body, and reject error pages or unsupported binary data. */
export async function fetchResearchText(
  url: string,
  init: Parameters<typeof fetch>[1] = {},
  limits: { timeoutMs?: number; maxBytes?: number } = {},
): Promise<{ response: Awaited<ReturnType<typeof fetch>>; body: string }> {
  const maxBytes = limits.maxBytes ?? 5 * 1024 * 1024;
  const response = await fetch(url, { ...init, signal: AbortSignal.timeout(limits.timeoutMs ?? 30_000) });
  const rejectBody = async (message: string): Promise<never> => {
    await response.body?.cancel();
    throw new Error(message);
  };
  if (!response.ok) return rejectBody(`HTTP取得に失敗しました (status ${response.status}): ${url}`);
  const contentType = (response.headers.get("content-type") ?? "").split(";", 1)[0].trim().toLowerCase();
  if (!["text/html", "application/xhtml+xml", "text/plain"].includes(contentType)) {
    return rejectBody(`未対応の Content-Type です: ${contentType || "不明"}`);
  }
  if (Number(response.headers.get("content-length")) > maxBytes) return rejectBody(`HTTP本文が上限 ${maxBytes} bytes を超えています`);
  if (!response.body) throw new Error("HTTP本文がありません");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const chunks: string[] = [];
  let bytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > maxBytes) {
        await reader.cancel();
        throw new Error(`HTTP本文が上限 ${maxBytes} bytes を超えています`);
      }
      chunks.push(decoder.decode(value, { stream: true }));
    }
    chunks.push(decoder.decode());
  } finally {
    reader.releaseLock();
  }
  const body = chunks.join("");
  if (!body.trim()) throw new Error("HTTP本文が空です");
  return { response, body };
}

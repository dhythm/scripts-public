#!/usr/bin/env -S deno run --allow-read --allow-write --allow-env --allow-net

import { parseArgs, validateArgs } from "./cli/parser.ts";
import { PdfTextExtractor } from "./app.ts";

async function main(): Promise<void> {
  try {
    const args = parseArgs(Deno.args);
    const config = validateArgs(args);

    if (!config) {
      Deno.exit(1);
    }

    const extractor = new PdfTextExtractor(config);
    const summary = await extractor.run();

    Deno.exit(summary.failedFiles > 0 ? 1 : 0);

  } catch (error) {
    console.error("予期しないエラーが発生しました:", error);
    Deno.exit(1);
  }
}

if (import.meta.main) {
  await main();
}
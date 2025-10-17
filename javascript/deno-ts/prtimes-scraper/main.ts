#!/usr/bin/env -S deno run --allow-net --allow-write

import { parseArgs } from "jsr:@std/cli/parse-args";
import { scrapeReleases, saveToFile, fetchCompanyInfo } from "./src/scraper.ts";
import type { ScraperOptions } from "./src/types.ts";

/**
 * メイン処理
 */
async function main() {
  const args = parseArgs(Deno.args, {
    string: ["output", "o"],
    boolean: ["verbose", "v", "help", "h", "company-info", "c"],
    default: {
      output: "prtimes_releases.json",
      verbose: false,
      "company-info": false,
    },
    alias: {
      o: "output",
      v: "verbose",
      h: "help",
      c: "company-info",
    },
  });

  if (args.help) {
    printHelp();
    Deno.exit(0);
  }

  const options: ScraperOptions = {
    outputPath: args.output,
    verbose: args.verbose,
    fetchCompanyInfo: args["company-info"],
  };

  try {
    if (options.verbose) {
      console.log("PRTIMESの新着リリースを取得中...");
    }

    const releases = await scrapeReleases();

    if (options.verbose) {
      console.log(`${releases.length}件のリリースを取得しました`);
    }

    // 企業情報も取得する場合
    if (options.fetchCompanyInfo) {
      if (options.verbose) {
        console.log("企業情報を取得中...");
      }

      for (let i = 0; i < releases.length; i++) {
        const release = releases[i];
        if (release.companyId) {
          try {
            const companyInfo = await fetchCompanyInfo(release.companyId);
            releases[i].companyUrl = companyInfo.url;

            if (options.verbose) {
              console.log(
                `[${i + 1}/${releases.length}] ${release.company}: ${companyInfo.url || "URL不明"}`
              );
            }
          } catch (error) {
            if (options.verbose) {
              console.error(
                `[${i + 1}/${releases.length}] ${release.company}: 企業情報の取得に失敗`
              );
            }
          }
        }
      }
    }

    if (options.verbose) {
      console.log(`出力先: ${options.outputPath}`);
    }

    if (options.outputPath) {
      await saveToFile(releases, options.outputPath);
      console.log(`✅ ${releases.length}件のリリースを取得しました`);
    } else {
      console.log(JSON.stringify(releases, null, 2));
    }
  } catch (error) {
    console.error("❌ エラーが発生しました:");
    if (error instanceof Error) {
      console.error(error.message);
    } else {
      console.error("不明なエラー");
    }
    Deno.exit(1);
  }
}

/**
 * ヘルプメッセージを表示
 */
function printHelp() {
  console.log(`
PRTIMESスクレイパー - PRTIMESの新着リリースを取得します

使い方:
  deno run --allow-net --allow-write --allow-env main.ts [オプション]

オプション:
  -o, --output <path>      出力ファイルのパス (デフォルト: prtimes_releases.json)
  -v, --verbose            詳細ログを表示
  -c, --company-info       企業のウェブサイトURLも取得（時間がかかります）
  -h, --help               このヘルプを表示

例:
  # デフォルト設定で実行
  deno run --allow-net --allow-write --allow-env main.ts

  # 出力ファイル名を指定
  deno run --allow-net --allow-write --allow-env main.ts -o output.json

  # 詳細ログを表示
  deno run --allow-net --allow-write --allow-env main.ts -v

  # 企業情報も取得（時間がかかります）
  deno run --allow-net --allow-write --allow-env main.ts -c -v
  `);
}

if (import.meta.main) {
  main();
}

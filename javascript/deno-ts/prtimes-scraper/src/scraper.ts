import * as cheerio from "npm:cheerio@1.0.0";
import type { PrTimesRelease, CompanyInfo } from "./types.ts";

const TARGET_URL = "https://prtimes.jp/main/html/newarrival";

/**
 * HTMLをパースしてリリース情報を抽出する
 * @param html パース対象のHTML
 * @returns リリース情報の配列
 */
export function parseHtml(html: string): PrTimesRelease[] {
  const $ = cheerio.load(html);

  const items = $(".list-article")
    .map((_, el) => {
      const title = $(el).find(".list-article__title").text().trim();
      const href = $(el).find(".list-article__link").attr("href");
      const url = href ? `https://prtimes.jp${href}` : "";

      // 企業名はリンク付きのものを取得（dummyではない方）
      const companyLink = $(el).find(".list-article__company-name-link");
      const company = companyLink.length > 0
        ? companyLink.text().trim()
        : $(el).find(".list-article__company-name").first().text().trim();

      // 企業IDを取得（企業ページのURLから）
      const companyHref = companyLink.attr("href");
      const companyIdMatch = companyHref?.match(/company_id\/(\d+)/);
      const companyId = companyIdMatch ? companyIdMatch[1] : undefined;

      const timeElement = $(el).find(".list-article__time");
      const date = timeElement.attr("datetime") || timeElement.text().trim();

      return { title, company, url, date, companyId };
    })
    .get();

  return items;
}

/**
 * PRTIMESの新着リリース一覧をスクレイピングする
 * @returns リリース情報の配列
 * @throws ネットワークエラーやパースエラーが発生した場合
 */
export async function scrapeReleases(): Promise<PrTimesRelease[]> {
  try {
    const response = await fetch(TARGET_URL);

    if (!response.ok) {
      throw new Error(
        `HTTPエラー: ${response.status} ${response.statusText}`
      );
    }

    const html = await response.text();
    return parseHtml(html);
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`スクレイピングに失敗しました: ${error.message}`);
    }
    throw error;
  }
}

/**
 * 企業ページから企業情報を取得する
 * @param companyId 企業ID
 * @returns 企業情報
 */
export async function fetchCompanyInfo(
  companyId: string
): Promise<CompanyInfo> {
  const companyUrl = `https://prtimes.jp/main/html/searchrlp/company_id/${companyId}`;

  try {
    const response = await fetch(companyUrl);

    if (!response.ok) {
      throw new Error(
        `HTTPエラー: ${response.status} ${response.statusText}`
      );
    }

    const html = await response.text();
    const $ = cheerio.load(html);

    // 企業情報セクションから情報を抽出
    const info: CompanyInfo = {};

    // 基本情報テーブルから情報を抽出
    $(".company-information__table-row").each((_, row) => {
      const $row = $(row);
      const label = $row.find(".company-information__label").text().trim();
      const value = $row.find(".company-information__value");

      if (label === "URL") {
        const urlLink = value.find("a").attr("href");
        if (urlLink) {
          info.url = urlLink;
        }
      } else if (label === "業種") {
        info.industry = value.text().trim();
      } else if (label === "本社所在地") {
        info.address = value.text().trim();
      } else if (label === "資本金") {
        info.capital = value.text().trim();
      }
    });

    return info;
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(
        `企業情報の取得に失敗しました (ID: ${companyId}): ${error.message}`
      );
    }
    throw error;
  }
}

/**
 * リリース情報をJSON形式でファイルに保存する
 * @param releases リリース情報の配列
 * @param outputPath 出力ファイルパス
 */
export async function saveToFile(
  releases: PrTimesRelease[],
  outputPath: string
): Promise<void> {
  try {
    const jsonContent = JSON.stringify(releases, null, 2);
    await Deno.writeTextFile(outputPath, jsonContent);
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`ファイルの保存に失敗しました: ${error.message}`);
    }
    throw error;
  }
}

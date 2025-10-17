import { chromium } from "npm:playwright";
import * as cheerio from "npm:cheerio@1.0.0";
import type { CompanyInfo, PrTimesRelease } from "./types.ts";

const BASE_URL = "https://prtimes.jp";
const TARGET_URL = `${BASE_URL}/main/html/newarrival`;
const COMPANY_API_URL = `${BASE_URL}/api/companies`;

const DEFAULT_MAX_LOAD_MORE = 10;
const DEFAULT_WAIT_AFTER_CLICK_MS = 800;

export type ReleaseScrapeOptions = {
  /** 「もっと見る」ボタンをクリックする最大回数 */
  maxLoadMore?: number;
  /** クリック後に待機する時間（ms） */
  waitAfterClickMs?: number;
  /** ログ出力を行うか */
  verbose?: boolean;
};

type CompanyApiResponse = {
  url?: string | null;
  industry_kbn_name?: string | null;
  address?: string | null;
  address_2?: string | null;
  capital?: string | null;
  phone?: string | null;
  foundation_date?: { year?: string | null; month?: string | null } | null;
  president_name?: string | null;
  description?: string | null;
  twitter_screen_name?: string | null;
  facebook_page_url?: string | null;
  youtube_channel_url?: string | null;
  follower_num?: number | null;
  og_image_url?: string | null;
  cover_image_url?: string | { pc?: string | null; sp?: string | null } | null;
  logo_image_url?: string | { pc?: string | null; sp?: string | null } | null;
};

const isCompanyApiResponse = (value: unknown): value is CompanyApiResponse => {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  return !Object.prototype.hasOwnProperty.call(value, "status");
};

const toAbsoluteUrl = (href?: string | null): string | undefined => {
  if (!href) return undefined;
  try {
    return new URL(href, BASE_URL).toString();
  } catch {
    return href ?? undefined;
  }
};

/**
 * HTMLをパースしてリリース情報を抽出する
 * @param html パース対象のHTML
 * @returns リリース情報の配列
 */
export function parseHtml(html: string): PrTimesRelease[] {
  const $ = cheerio.load(html);

  return $("article.list-article")
    .map((_, el) => {
      const $el = $(el);

      const linkHref = $el.find(".list-article__link").attr("href");
      const url = toAbsoluteUrl(linkHref) ?? "";
      const title = $el.find(".list-article__title").text().trim();

      const companyLink = $el.find(".list-article__company-name-link").first();
      const companyText = companyLink.text().trim();
      const fallbackCompany = $el
        .find(".list-article__company-name, .list-article__company-name--dummy")
        .first()
        .text()
        .trim();
      const company = companyText || fallbackCompany;

      const companyHref = companyLink.attr("href");
      const companyIdMatch = companyHref?.match(/company_id\/(\d+)/);
      const companyId = companyIdMatch ? companyIdMatch[1] : undefined;

      const timeElement = $el.find(".list-article__time").first();
      const date = timeElement.attr("datetime")?.trim() || timeElement.text().trim();
      const relativeTime = timeElement.text().trim() || undefined;

      const imageEl = $el.find(".list-article__image-img").first();
      const imageSrc = imageEl.attr("data-src") ?? imageEl.attr("src");
      const thumbnailUrl = toAbsoluteUrl(imageSrc);
      const thumbnailAlt = imageEl.attr("alt")?.trim();

      const releaseIdMatch = linkHref?.match(/\/p\/([^/.]+(?:\.[^/.]+)?)\.html/i);
      const releaseId = releaseIdMatch ? releaseIdMatch[1] : undefined;

      const tags = $el
        .find(".list-article__tag, .list-article__label")
        .map((__, tagEl) => $(tagEl).text().trim())
        .get()
        .filter((tag) => tag.length > 0);

      return {
        title,
        company,
        url,
        date,
        relativeTime,
        companyId,
        releaseId,
        thumbnailUrl,
        thumbnailAlt,
        tags: tags.length > 0 ? Array.from(new Set(tags)) : undefined,
      };
    })
    .get();
}

/**
 * PRTIMESの新着リリース一覧をスクレイピングする
 * @returns リリース情報の配列
 * @throws ネットワークエラーやパースエラーが発生した場合
 */
export async function scrapeReleases(
  options: ReleaseScrapeOptions = {}
): Promise<PrTimesRelease[]> {
  const {
    maxLoadMore = DEFAULT_MAX_LOAD_MORE,
    waitAfterClickMs = DEFAULT_WAIT_AFTER_CLICK_MS,
    verbose = false,
  } = options;

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  try {
    await page.goto(TARGET_URL, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await page.waitForSelector("article.list-article", { timeout: 60_000 });

    let iteration = 0;
    while (iteration < maxLoadMore) {
      const loadMoreButton = await page.$(".js-new-arrival-list-article-more-button.active");
      if (!loadMoreButton) {
        if (verbose) {
          console.log("「もっと見る」ボタンが見つからなかったため読み込みを終了します。");
        }
        break;
      }

      const previousCount = await page.locator("article.list-article").count();
      if (verbose) {
        console.log(`もっと見るをクリック (${iteration + 1}回目)。取得済み: ${previousCount}件`);
      }

      await loadMoreButton.click({ delay: 50 });

      let currentCount = previousCount;
      for (let retry = 0; retry < 20; retry++) {
        await page.waitForTimeout(250);
        currentCount = await page.locator("article.list-article").count();
        if (currentCount > previousCount) {
          break;
        }
      }

      await page.waitForTimeout(waitAfterClickMs);

      if (verbose) {
        console.log(`現在の件数: ${currentCount}件`);
      }

      iteration += 1;

      if (currentCount <= previousCount) {
        if (verbose) {
          console.log("件数が増加しなかったため追加読み込みを終了します。");
        }
        break;
      }
    }

    const html = await page.content();
    return parseHtml(html);
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`スクレイピングに失敗しました: ${error.message}`);
    }
    throw error;
  } finally {
    await page.close().catch(() => {});
    await browser.close().catch(() => {});
  }
}

/**
 * 企業APIから企業情報を取得する
 * @param companyId 企業ID
 * @returns 企業情報
 */
export async function fetchCompanyInfo(companyId: string): Promise<CompanyInfo> {
  const apiUrl = `${COMPANY_API_URL}/${companyId}`;

  try {
    const response = await fetch(apiUrl, {
      headers: { accept: "application/json" },
    });

    if (!response.ok) {
      throw new Error(`HTTPエラー: ${response.status} ${response.statusText}`);
    }

    const raw = await response.json();
    if (!isCompanyApiResponse(raw)) {
      throw new Error(`APIエラー: ${JSON.stringify(raw)}`);
    }
    const data = raw;

    const info: CompanyInfo = {
      url: data.url ?? undefined,
      industry: data.industry_kbn_name ?? undefined,
      address: data.address ?? undefined,
      address2: data.address_2 ?? undefined,
      capital: data.capital ?? undefined,
      phone: data.phone ?? undefined,
      representative: data.president_name ?? undefined,
      description: data.description ?? undefined,
      followerCount: typeof data.follower_num === "number" ? data.follower_num : undefined,
      ogImageUrl: toAbsoluteUrl(
        typeof data.og_image_url === "string" ? data.og_image_url : undefined,
      ),
      coverImageUrl: toAbsoluteUrl(
        typeof data.cover_image_url === "string" ? data.cover_image_url : data.cover_image_url?.pc,
      ),
      logoImageUrl: toAbsoluteUrl(
        typeof data.logo_image_url === "string" ? data.logo_image_url : data.logo_image_url?.pc,
      ),
    };

    const foundationYear = data.foundation_date?.year ?? undefined;
    const foundationMonth = data.foundation_date?.month ?? undefined;
    if (foundationYear) {
      info.foundationDate = foundationMonth
        ? `${foundationYear}-${foundationMonth.toString().padStart(2, "0")}`
        : foundationYear;
    }

    const twitterId = data.twitter_screen_name?.replace(/^@/, "");
    if (twitterId) {
      info.xUrl = `https://x.com/${twitterId}`;
    }
    if (data.facebook_page_url) {
      info.facebookUrl = data.facebook_page_url;
    }
    if (data.youtube_channel_url) {
      info.youtubeUrl = data.youtube_channel_url;
    }

    return info;
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`企業情報の取得に失敗しました (ID: ${companyId}): ${error.message}`);
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
  outputPath: string,
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

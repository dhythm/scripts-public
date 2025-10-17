import { assertEquals, assertExists } from "jsr:@std/assert";
import { describe, it } from "jsr:@std/testing/bdd";
import { scrapeReleases, parseHtml } from "../src/scraper.ts";
import type { PrTimesRelease } from "../src/types.ts";

describe("PRTIMESスクレイパー", () => {
  describe("parseHtml", () => {
    it("HTMLをパースしてリリース情報を抽出できる", () => {
      const mockHtml = `
        <article class="list-article">
          <a href="/main/html/rd/p/000000001.000000001.html" class="list-article__link">
            <div class="list-article__content">
              <h3 class="list-article__title">テストリリース</h3>
              <time datetime="2025-01-15T10:00:00+0900" class="list-article__time">2025.01.15 10:00</time>
            </div>
          </a>
          <span class="list-article__company-name">
            <a href="/main/html/searchrlp/company_id/1" class="list-article__company-name-link">
              テスト株式会社
            </a>
          </span>
        </article>
      `;

      const result = parseHtml(mockHtml);

      assertEquals(result.length, 1);
      assertEquals(result[0].title, "テストリリース");
      assertEquals(result[0].company, "テスト株式会社");
      assertEquals(result[0].url, "https://prtimes.jp/main/html/rd/p/000000001.000000001.html");
      assertEquals(result[0].date, "2025-01-15T10:00:00+0900");
    });

    it("複数のリリース情報を抽出できる", () => {
      const mockHtml = `
        <article class="list-article">
          <a href="/main/html/rd/p/000000001.000000001.html" class="list-article__link">
            <div class="list-article__content">
              <h3 class="list-article__title">リリース1</h3>
              <time datetime="2025-01-15T10:00:00+0900" class="list-article__time">2025.01.15 10:00</time>
            </div>
          </a>
          <span class="list-article__company-name">
            <a href="/main/html/searchrlp/company_id/1" class="list-article__company-name-link">
              企業A
            </a>
          </span>
        </article>
        <article class="list-article">
          <a href="/main/html/rd/p/000000002.000000002.html" class="list-article__link">
            <div class="list-article__content">
              <h3 class="list-article__title">リリース2</h3>
              <time datetime="2025-01-15T11:00:00+0900" class="list-article__time">2025.01.15 11:00</time>
            </div>
          </a>
          <span class="list-article__company-name">
            <a href="/main/html/searchrlp/company_id/2" class="list-article__company-name-link">
              企業B
            </a>
          </span>
        </article>
      `;

      const result = parseHtml(mockHtml);

      assertEquals(result.length, 2);
      assertEquals(result[0].title, "リリース1");
      assertEquals(result[1].title, "リリース2");
    });

    it("空のHTMLの場合は空配列を返す", () => {
      const mockHtml = "<html><body></body></html>";
      const result = parseHtml(mockHtml);
      assertEquals(result.length, 0);
    });
  });

  describe("scrapeReleases", () => {
    it("型定義に従ったデータ構造を返す", async () => {
      // 注: このテストは実際のネットワークアクセスを伴うため、
      // 必要に応じてモックを使用することを検討してください
      const releases = await scrapeReleases();

      assertExists(releases);
      assertEquals(Array.isArray(releases), true);

      if (releases.length > 0) {
        const release = releases[0];
        assertExists(release.title);
        assertExists(release.company);
        assertExists(release.url);
        assertExists(release.date);
        assertEquals(typeof release.title, "string");
        assertEquals(typeof release.company, "string");
        assertEquals(typeof release.url, "string");
        assertEquals(typeof release.date, "string");
      }
    });
  });
});

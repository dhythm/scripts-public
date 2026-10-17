import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  mkdir: vi.fn(), access: vi.fn(), exists: vi.fn(), stat: vi.fn(), write: vi.fn(),
  fetch: vi.fn(), create: vi.fn(), retrieve: vi.fn(),
}));
vi.mock("node:fs", () => ({
  mkdirSync: mocks.mkdir, accessSync: mocks.access, existsSync: mocks.exists,
  statSync: mocks.stat, writeFileSync: mocks.write, constants: { W_OK: 2 },
}));
vi.mock("openai", () => ({ default: class {
  responses = { create: mocks.create, retrieve: mocks.retrieve };
} }));

import { fetchResearchText, prepareOutputPath } from "./research-io.js";
import { main as harvest, pdfSearch } from "./primary-source-harvester.js";
import { main as analyze } from "./source-origin-analyzer.js";

const originalArgv = process.argv;
const originalExitCode = process.exitCode;
const article = `<html><title>Fixture</title><article>${"検証用の記事本文です。".repeat(12)}</article></html>`;
const sourcePayload = {
  classification: "primary", reasoning: "公式情報です", summary: "要約です", keyEntities: [], followUpSources: [], pendingNeeds: "なし",
};
const harvestedPayload = {
  keyword: "fixture", summary: "要約です", stats: { primaryCount: 1, secondaryCount: 0 }, pendingGaps: "なし",
  sources: [{ classification: "primary", title: "Fixture", url: "https://example.test/source", excerpt: "検証用の引用です。".repeat(20) }],
};
let stdout: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  vi.clearAllMocks();
  process.exitCode = undefined;
  vi.stubEnv("OPENAI_API_KEY", "mock-key-not-a-credential");
  vi.stubGlobal("fetch", mocks.fetch);
  mocks.fetch.mockImplementation(async () => new Response(article, { headers: { "content-type": "text/html; charset=utf-8" } }));
  mocks.mkdir.mockImplementation(() => {});
  mocks.access.mockImplementation(() => {});
  mocks.exists.mockReturnValue(false);
  mocks.stat.mockReturnValue({ isFile: () => true });
  mocks.create.mockResolvedValue({ id: "mock-id", output_text: JSON.stringify(sourcePayload) });
  mocks.retrieve.mockResolvedValue({ id: "mock-id", status: "completed", output: [], output_text: JSON.stringify(harvestedPayload) });
  stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
  vi.spyOn(console, "error").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "log").mockImplementation(() => {});
});
afterEach(() => {
  process.argv = originalArgv;
  process.exitCode = originalExitCode;
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

function args(...values: string[]): void { process.argv = ["node", "fixture.ts", ...values]; }
function output(): unknown { return JSON.parse(stdout.mock.calls.map((call: unknown[]) => call[0]).join("")); }

describe("research HTTP validation", () => {
  it("accepts text and reconstructs split UTF-8 with a bounded request signal", async () => {
    const bytes = new TextEncoder().encode("日本語");
    mocks.fetch.mockResolvedValueOnce(new Response(new ReadableStream({ start(controller) {
      controller.enqueue(bytes.slice(0, 2)); controller.enqueue(bytes.slice(2)); controller.close();
    } }), { headers: { "content-type": "text/plain" } }));
    await expect(fetchResearchText("https://example.test")).resolves.toMatchObject({ body: "日本語" });
    expect(mocks.fetch).toHaveBeenCalledWith("https://example.test", expect.objectContaining({ signal: expect.any(AbortSignal) }));
  });

  it.each([403, 404, 500])("rejects HTTP %i before calling the model", async (status) => {
    args("--url", "https://example.test/article", "--json");
    mocks.fetch.mockResolvedValueOnce(new Response("Error page", { status, headers: { "content-type": "text/html" } }));
    await analyze();
    expect(process.exitCode).toBe(1);
    expect(mocks.create).not.toHaveBeenCalled();
    expect(stdout).not.toHaveBeenCalled();
  });

  it.each(["application/pdf", "application/octet-stream", "application/json", ""])("rejects unsupported Content-Type %s", async (mime) => {
    mocks.fetch.mockResolvedValueOnce(new Response("binary fixture", { headers: { "content-type": mime } }));
    await expect(fetchResearchText("https://example.test")).rejects.toThrow("Content-Type");
  });

  it("rejects a declared oversized response", async () => {
    mocks.fetch.mockResolvedValueOnce(new Response("small fixture", { headers: { "content-type": "text/plain", "content-length": "100" } }));
    await expect(fetchResearchText("https://example.test", {}, { maxBytes: 20 })).rejects.toThrow("上限");
  });

  it("counts streamed bytes even without Content-Length", async () => {
    mocks.fetch.mockResolvedValueOnce(new Response("日本語", { headers: { "content-type": "text/plain" } }));
    await expect(fetchResearchText("https://example.test", {}, { maxBytes: 6 })).rejects.toThrow("上限");
  });

  it("rejects an empty body", async () => {
    mocks.fetch.mockResolvedValueOnce(new Response("  ", { headers: { "content-type": "text/html" } }));
    await expect(fetchResearchText("https://example.test")).rejects.toThrow("空");
  });

  it("aborts a stalled request using its timeout signal", async () => {
    mocks.fetch.mockImplementationOnce((_url, init) => new Promise((_resolve, reject) => {
      init.signal.addEventListener("abort", () => reject(init.signal.reason), { once: true });
    }));
    await expect(fetchResearchText("https://example.test", {}, { timeoutMs: 5 })).rejects.toThrow();
  });

  it("uses the same HTTP validation for Bing PDF search", async () => {
    mocks.fetch.mockResolvedValueOnce(new Response("blocked", { status: 403 }));
    await expect(pdfSearch("fixture", 1, [])).rejects.toThrow("403");
    expect(mocks.create).not.toHaveBeenCalled();
  });
});

describe("research CLI output and failure status", () => {
  it("writes only parseable JSON to stdout for source-origin --json", async () => {
    args("--url", "https://example.test/article", "--json");
    await analyze();
    expect(output()).toMatchObject({ classification: "primary", url: "https://example.test/article" });
    expect(console.log).not.toHaveBeenCalled();
    expect(console.error).toHaveBeenCalled();
    expect(process.exitCode).toBeUndefined();
  });

  it("creates the source-origin output directory before the model call", async () => {
    args("--url", "https://example.test/article", "--json", "--output", "reports/origin.json");
    await analyze();
    expect(mocks.mkdir).toHaveBeenCalledWith("reports", { recursive: true });
    expect(mocks.mkdir.mock.invocationCallOrder[0]).toBeLessThan(mocks.create.mock.invocationCallOrder[0]);
    expect(mocks.write).toHaveBeenCalledWith("reports/origin.json", expect.any(String), "utf-8");
  });

  it("stops source-origin before HTTP and API when the output path is unusable", async () => {
    args("--url", "https://example.test/article", "--output", "reports/origin.json");
    mocks.mkdir.mockImplementationOnce(() => { throw new Error("permission denied fixture"); });
    await analyze();
    expect(process.exitCode).toBe(1);
    expect(mocks.fetch).not.toHaveBeenCalled();
    expect(mocks.create).not.toHaveBeenCalled();
  });

  it("writes parseable JSON to stdout for primary-harvest", async () => {
    args("--keyword", "fixture", "--pdf-strategy", "never");
    await harvest();
    expect(output()).toMatchObject({ reports: [{ keyword: "fixture", stats: { primaryCount: 1 } }] });
    expect(console.log).not.toHaveBeenCalled();
    expect(process.exitCode).toBeUndefined();
  });

  it("preserves partial results and returns nonzero if any keyword fails", async () => {
    args("--keyword", "fails", "--keyword", "fixture", "--pdf-strategy", "never");
    mocks.create.mockRejectedValueOnce(new Error("fixture API failure"));
    await harvest();
    expect(output()).toMatchObject({ reports: [{ keyword: "fails", error: "fixture API failure" }, { keyword: "fixture" }] });
    expect(process.exitCode).toBe(1);
  });

  it("returns nonzero and an error report when all keywords fail", async () => {
    args("--keyword", "fails", "--pdf-strategy", "never");
    mocks.create.mockRejectedValueOnce(new Error("fixture API failure"));
    await harvest();
    expect(output()).toMatchObject({ reports: [{ error: "fixture API failure" }] });
    expect(process.exitCode).toBe(1);
  });

  it("stops primary-harvest before API when destination preparation fails", async () => {
    args("--keyword", "fixture", "--output", "reports/harvest.json");
    mocks.access.mockImplementationOnce(() => { throw new Error("read only fixture"); });
    await expect(harvest()).rejects.toThrow("read only fixture");
    expect(mocks.create).not.toHaveBeenCalled();
  });

  it("rejects a directory as an output file", () => {
    mocks.exists.mockReturnValue(true);
    mocks.stat.mockReturnValue({ isFile: () => false });
    expect(() => prepareOutputPath("reports")).toThrow("通常ファイル");
  });
});

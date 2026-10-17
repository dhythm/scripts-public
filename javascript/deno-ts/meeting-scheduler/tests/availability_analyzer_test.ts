import { assert, assertEquals } from "jsr:@std/assert@^1.0.15";
import { describe, it } from "jsr:@std/testing@^1.0.16/bdd";
import {
  AvailabilityAnalyzer,
  type ChatCompletionClient,
  OPENAI_MODEL,
} from "../processors/availability-analyzer.ts";
import type { Person, TimeSlot } from "../types/index.ts";

const participants: Person[] = [
  { name: "山田", email: "yamada@example.com", source: "google", sourceId: "yamada" },
];

const slots: TimeSlot[] = [
  { start: new Date("2026-10-20T01:00:00Z"), end: new Date("2026-10-20T02:00:00Z") },
  { start: new Date("2026-10-21T05:00:00Z"), end: new Date("2026-10-21T06:00:00Z") },
];

function fakeClient(
  respond: (params: Record<string, unknown>) => Promise<unknown>,
): { client: ChatCompletionClient; calls: Record<string, unknown>[] } {
  const calls: Record<string, unknown>[] = [];
  const client = {
    chat: {
      completions: {
        create: (params: Record<string, unknown>) => {
          calls.push(params);
          return respond(params);
        },
      },
    },
  } as unknown as ChatCompletionClient;
  return { client, calls };
}

function jsonResponse(body: unknown) {
  return Promise.resolve({ choices: [{ message: { content: JSON.stringify(body) } }] });
}

describe("AvailabilityAnalyzer.optimizeWithOpenAI", () => {
  it("gpt-6-luna を使い、temperature は送らない", async () => {
    const { client, calls } = fakeClient(() => jsonResponse({ recommendations: [] }));
    const analyzer = new AvailabilityAnalyzer({ apiKey: "test" }, client);

    await analyzer.optimizeWithOpenAI(slots, participants);

    assertEquals(OPENAI_MODEL, "gpt-6-luna");
    assertEquals(calls.length, 1);
    assertEquals(calls[0].model, "gpt-6-luna");
    assert(!("temperature" in calls[0]), "temperature を送ってはいけない");
  });

  it("AIの推薦を候補として使う", async () => {
    const { client } = fakeClient(() =>
      jsonResponse({ recommendations: [{ index: 1, score: 99, reasons: ["AIの推薦"] }] })
    );
    const analyzer = new AvailabilityAnalyzer({ apiKey: "test" }, client);

    const candidates = await analyzer.optimizeWithOpenAI(slots, participants);

    assertEquals(candidates[0].score, 99);
    assertEquals(candidates[0].reasons, ["AIの推薦"]);
    assertEquals(candidates[0].slot, slots[1]);
  });

  it("API エラー時は警告を出して簡易スコアリングに切り替える", async () => {
    const { client } = fakeClient(() => Promise.reject(new Error("400 Unsupported value")));
    const analyzer = new AvailabilityAnalyzer({ apiKey: "test" }, client);
    const warnings: string[] = [];
    const originalWarn = console.warn;
    const originalError = console.error;
    console.warn = (...args: unknown[]) => warnings.push(args.map(String).join(" "));
    console.error = () => {};
    try {
      const candidates = await analyzer.optimizeWithOpenAI(slots, participants);
      assertEquals(candidates.length, 2);
    } finally {
      console.warn = originalWarn;
      console.error = originalError;
    }

    assert(
      warnings.some((w) => w.includes("簡易スコアリング")),
      `簡易スコアリングへの切り替えを警告すること: ${warnings}`,
    );
  });
});

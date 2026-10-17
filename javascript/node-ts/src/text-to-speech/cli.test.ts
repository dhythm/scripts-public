import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ client: vi.fn(), tts: vi.fn(), read: vi.fn() }));
vi.mock("openai", () => ({ default: class { constructor() { mocks.client(); } } }));
vi.mock("node:fs", () => ({
  readFileSync: mocks.read, existsSync: vi.fn(), mkdirSync: vi.fn(), writeFileSync: vi.fn(),
}));
vi.mock("./index.js", async (importOriginal) => ({
  ...await importOriginal<typeof import("./index.js")>(), textToSpeech: mocks.tts,
}));

import { main, parseCliOptions } from "./cli.js";
import { AUDIO_FORMATS, MODEL_TYPES, VOICE_TYPES } from "./index.js";

const originalArgv = process.argv;
beforeEach(() => {
  vi.clearAllMocks();
  vi.stubEnv("OPENAI_API_KEY", "mock-key-not-a-credential");
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
  vi.spyOn(process, "exit").mockImplementation((code) => { throw new Error(`exit:${code}`); });
  mocks.tts.mockResolvedValue({ outputPath: "/mock/output.mp3", sizeBytes: 1 });
});
afterEach(() => { process.argv = originalArgv; vi.unstubAllEnvs(); vi.restoreAllMocks(); });
function args(...values: string[]): void {
  process.argv = ["node", "cli.ts", "--text", "テスト", "--output", "/mock/output.mp3", ...values];
}

describe("TTS CLI validation before API or file work", () => {
  it.each(["abc", "NaN", "Infinity", "-Infinity", "1.5junk", "", "0.249", "4.001"])(
    "rejects invalid speed %s before creating a client or generating audio", async (speed) => {
      args(`--speed=${speed}`);
      expect(() => parseCliOptions()).toThrow("有限な数値");
      await expect(main()).rejects.toThrow("exit:1");
      expect(mocks.client).not.toHaveBeenCalled();
      expect(mocks.tts).not.toHaveBeenCalled();
      expect(mocks.read).not.toHaveBeenCalled();
    },
  );

  it.each(["0.25", "1", "1.5", "4"])("accepts boundary or valid speed %s", (speed) => {
    args("--speed", speed);
    expect(parseCliOptions().speed).toBe(Number(speed));
  });

  it("preserves the default values", () => {
    args();
    expect(parseCliOptions()).toMatchObject({ voice: "nova", model: "tts-1", format: "mp3", speed: 1 });
  });

  it.each(["voice", "model", "format"])("rejects an unsupported --%s before API work", async (option) => {
    args(`--${option}`, "unsupported-fixture");
    expect(() => parseCliOptions()).toThrow(`--${option}`);
    await expect(main()).rejects.toThrow("exit:1");
    expect(mocks.client).not.toHaveBeenCalled();
    expect(mocks.tts).not.toHaveBeenCalled();
  });

  it.each(VOICE_TYPES)("accepts declared voice %s", (voice) => {
    args("--voice", voice);
    expect(parseCliOptions().voice).toBe(voice);
  });
  it.each(MODEL_TYPES)("accepts declared model %s", (model) => {
    args("--model", model);
    expect(parseCliOptions().model).toBe(model);
  });
  it.each(AUDIO_FORMATS)("accepts declared format %s", (format) => {
    args("--format", format);
    expect(parseCliOptions().format).toBe(format);
  });

  it("passes valid options to a mocked generator without making a real API call", async () => {
    args("--voice", "alloy", "--model", "tts-1-hd", "--format", "wav", "--speed", "0.25");
    await main();
    expect(mocks.tts).toHaveBeenCalledWith(expect.anything(), "テスト", "/mock/output.mp3", {
      voice: "alloy", model: "tts-1-hd", format: "wav", speed: 0.25,
    });
    expect(process.exit).not.toHaveBeenCalled();
  });
});

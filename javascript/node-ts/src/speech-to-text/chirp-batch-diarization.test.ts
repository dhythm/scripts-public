import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { readdir } from "node:fs/promises";

const mocks = vi.hoisted(() => ({
  lstat: vi.fn(), write: vi.fn(), upload: vi.fn(), download: vi.fn(),
  file: vi.fn(), bucket: vi.fn(), getFiles: vi.fn(), delete: vi.fn(),
  getRecognizer: vi.fn(), batchRecognize: vi.fn(),
}));

vi.mock("node:fs", () => ({
  writeFileSync: mocks.write, lstatSync: mocks.lstat,
  existsSync: vi.fn(() => true), mkdirSync: vi.fn(), readFileSync: vi.fn(),
}));
vi.mock("node:fs/promises", () => ({
  mkdtemp: vi.fn(async () => "/mock/chirp-transcode"), readdir: vi.fn(async () => []),
}));
vi.mock("node:child_process", () => ({
  execFile: vi.fn((_cmd, _args, done) => {
    done(null, "", "");
    return { stderr: { on: vi.fn() } };
  }),
}));
vi.mock("@google-cloud/storage", () => ({
  Storage: class { bucket = mocks.bucket; },
}));
vi.mock("@google-cloud/speech", () => ({
  v2: { SpeechClient: class {
    getRecognizer = mocks.getRecognizer;
    batchRecognize = mocks.batchRecognize;
  } },
}));
vi.mock("google-gax", () => ({ Status: { NOT_FOUND: 5 } }));

import { assertSafeMergedOutput, main, readTranscriptResults, validateOutputGcsUri } from "./chirp-batch-diarization.js";

const transcript = [{ alternatives: [{ transcript: "テスト結果" }] }];
const originalArgv = process.argv;

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubEnv("GOOGLE_CLOUD_PROJECT", "mock-project");
  vi.stubEnv("GOOGLE_APPLICATION_CREDENTIALS", "/mock/credentials.json");
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
  mocks.lstat.mockImplementation(() => { throw Object.assign(new Error("missing"), { code: "ENOENT" }); });
  mocks.write.mockImplementation(() => {});
  mocks.upload.mockResolvedValue([]);
  mocks.download.mockResolvedValue([Buffer.from(JSON.stringify({ results: transcript }))]);
  mocks.file.mockReturnValue({ download: mocks.download, delete: mocks.delete });
  mocks.bucket.mockReturnValue({ upload: mocks.upload, file: mocks.file, getFiles: mocks.getFiles });
  mocks.getRecognizer.mockResolvedValue([]);
  mocks.batchRecognize.mockImplementation(async ({ files }) => [{
    promise: async () => [{ results: { [files[0].uri]: { transcript: { results: transcript } } } }],
  }]);
});
afterEach(() => { process.argv = originalArgv; vi.unstubAllEnvs(); vi.restoreAllMocks(); });

function args(...values: string[]): void {
  process.argv = ["node", "chirp-batch-diarization.ts", ...values];
}

describe("Chirp data protection", () => {
  it("rejects the input itself and existing or dangling-symlink output paths", () => {
    expect(() => assertSafeMergedOutput("/mock/audio", "/mock/audio")).toThrow("同じパス");
    mocks.lstat.mockReturnValue({ isSymbolicLink: () => true });
    expect(() => assertSafeMergedOutput("/mock/input.wav", "/mock/output.txt")).toThrow("既に存在");
  });

  it("validates collisions before uploading or recognizing", async () => {
    args("--file", "/mock/audio.wav", "--bucket", "test", "--merged-output", "/mock/audio.wav");
    await expect(main()).rejects.toThrow("同じパス");
    expect(mocks.upload).not.toHaveBeenCalled();
    expect(mocks.batchRecognize).not.toHaveBeenCalled();
  });

  it.each(["gs://test", "gs://test/", "gs://test///"])("rejects root output %s", (uri) => {
    expect(() => validateOutputGcsUri(uri)).toThrow("専用");
  });

  it("uploads without replacing existing objects and saves exclusively", async () => {
    args("--file", "/mock/audio.wav", "--bucket", "test", "--object", "input.flac", "--no-chunk");
    await main();
    expect(mocks.upload).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({
      destination: "input.flac", preconditionOpts: { ifGenerationMatch: 0 },
    }));
    expect(mocks.write).toHaveBeenCalledWith("/mock/audio.txt", "テスト結果", { encoding: "utf8", flag: "wx" });
    expect(mocks.delete).not.toHaveBeenCalled();
    expect(mocks.getFiles).not.toHaveBeenCalled();
  });

  it("preserves GCS data when exclusive local saving fails", async () => {
    args("--gcs-uri", "gs://test/input.flac", "--output-gcs", "gs://test/shared/", "--merged-output", "/mock/output.txt");
    mocks.write.mockImplementation(() => { throw new Error("EEXIST"); });
    await expect(main()).rejects.toThrow("EEXIST");
    expect(mocks.delete).not.toHaveBeenCalled();
    expect(mocks.getFiles).not.toHaveBeenCalled();
  });

  it("rejects an upload precondition failure without recognizing or deleting", async () => {
    args("--file", "/mock/audio.wav", "--bucket", "test", "--no-chunk");
    mocks.upload.mockRejectedValueOnce(new Error("precondition failed"));
    await expect(main()).rejects.toThrow("precondition failed");
    expect(mocks.batchRecognize).not.toHaveBeenCalled();
    expect(mocks.write).not.toHaveBeenCalled();
    expect(mocks.delete).not.toHaveBeenCalled();
  });

  it("reads only the result URI even with a shared output prefix", async () => {
    args("--gcs-uri", "gs://test/input.flac", "--output-gcs", "gs://test/shared/", "--merged-output", "/mock/output.txt");
    mocks.batchRecognize.mockResolvedValueOnce([{ promise: async () => [{ results: {
      "gs://test/input.flac": { cloudStorageResult: { uri: "gs://test/shared/this-run.json" } },
    } }] }]);
    await main();
    expect(mocks.file).toHaveBeenCalledExactlyOnceWith("shared/this-run.json");
    expect(mocks.write).toHaveBeenCalledWith("/mock/output.txt", "テスト結果", expect.objectContaining({ flag: "wx" }));
    expect(mocks.getFiles).not.toHaveBeenCalled();
    expect(mocks.delete).not.toHaveBeenCalled();
  });

  it("combines mixed cloud and inline chunks in input order", async () => {
    args("--file", "/mock/audio.wav", "--bucket", "test", "--object", "input");
    vi.mocked(readdir).mockResolvedValueOnce(["chunk-000.flac", "chunk-001.flac"] as never);
    mocks.batchRecognize.mockImplementationOnce(async ({ files }) => [{ promise: async () => [{ results: {
      [files[0].uri]: { cloudStorageResult: { uri: "gs://test/first-result.json" } },
    } }] }]);
    mocks.batchRecognize.mockImplementationOnce(async ({ files }) => [{ promise: async () => [{ results: {
      [files[0].uri]: { inlineResult: { transcript: { results: [{ alternatives: [{ transcript: "2番目" }] }] } } },
    } }] }]);
    await main();
    expect(mocks.write).toHaveBeenCalledWith("/mock/audio.txt", "テスト結果\n\n--- チャンク 2 開始 ---\n\n2番目", expect.objectContaining({ flag: "wx" }));
    expect(mocks.download).toHaveBeenCalledTimes(1);
    expect(mocks.getFiles).not.toHaveBeenCalled();
    expect(mocks.delete).not.toHaveBeenCalled();
  });

  it("keeps partial chunk results unsaved when a later chunk fails", async () => {
    args("--file", "/mock/audio.wav", "--bucket", "test", "--object", "input");
    vi.mocked(readdir).mockResolvedValueOnce(["chunk-000.flac", "chunk-001.flac"] as never);
    mocks.batchRecognize.mockImplementationOnce(async ({ files }) => [{ promise: async () => [{ results: {
      [files[0].uri]: { transcript: { results: transcript } },
    } }] }]);
    mocks.batchRecognize.mockImplementationOnce(async ({ files }) => [{ promise: async () => [{ results: {
      [files[0].uri]: { error: { code: 3, message: "later chunk failed" } },
    } }] }]);
    await expect(main()).rejects.toThrow("later chunk failed");
    expect(mocks.batchRecognize).toHaveBeenCalledTimes(2);
    expect(mocks.write).not.toHaveBeenCalled();
    expect(mocks.delete).not.toHaveBeenCalled();
  });

  it("accepts the current inlineResult response shape without downloading", async () => {
    await expect(readTranscriptResults({ inlineResult: { transcript: { results: transcript } } })).resolves.toEqual(transcript);
    expect(mocks.download).not.toHaveBeenCalled();
  });

  it.each(["not json", "{}", '{"results":[]}', '{"results":[{"alternatives":[{"transcript":""}]}]}'])(
    "rejects an unreadable or empty result without saving or deleting: %s", async (json) => {
      args("--gcs-uri", "gs://test/input.flac", "--merged-output", "/mock/output.txt");
      mocks.batchRecognize.mockResolvedValueOnce([{ promise: async () => [{ results: {
        "gs://test/input.flac": { cloudStorageResult: { uri: "gs://test/result.json" } },
      } }] }]);
      mocks.download.mockResolvedValueOnce([Buffer.from(json)]);
      await expect(main()).rejects.toThrow();
      expect(mocks.write).not.toHaveBeenCalled();
      expect(mocks.delete).not.toHaveBeenCalled();
    },
  );

  it("does not save a result when the cloud download fails", async () => {
    args("--gcs-uri", "gs://test/input.flac", "--merged-output", "/mock/output.txt");
    mocks.batchRecognize.mockResolvedValueOnce([{ promise: async () => [{ results: {
      "gs://test/input.flac": { uri: "gs://test/result.json" },
    } }] }]);
    mocks.download.mockRejectedValueOnce(new Error("download denied"));
    await expect(main()).rejects.toThrow("download denied");
    expect(mocks.write).not.toHaveBeenCalled();
    expect(mocks.delete).not.toHaveBeenCalled();
  });

  it("does not report file-level errors as saved results", async () => {
    args("--gcs-uri", "gs://test/input.flac", "--merged-output", "/mock/output.txt");
    mocks.batchRecognize.mockResolvedValueOnce([{ promise: async () => [{ results: {
      "gs://test/input.flac": { error: { code: 3, message: "invalid audio" } },
    } }] }]);
    await expect(main()).rejects.toThrow("invalid audio");
    expect(mocks.write).not.toHaveBeenCalled();
    expect(mocks.delete).not.toHaveBeenCalled();
  });
});

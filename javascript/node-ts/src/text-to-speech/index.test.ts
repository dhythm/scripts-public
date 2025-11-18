import { describe, it, expect, beforeEach, vi } from "vitest";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type OpenAI from "openai";
import { textToSpeech } from "./index.js";

// ファイルシステムのモック
vi.mock("node:fs", () => ({
  existsSync: vi.fn(),
  mkdirSync: vi.fn(),
  writeFileSync: vi.fn(),
}));

vi.mock("node:path", () => ({
  dirname: vi.fn(),
}));

describe("textToSpeech", () => {
  let mockClient: OpenAI;
  let mockCreate: ReturnType<typeof vi.fn>;
  let mockExistsSync: ReturnType<typeof vi.fn>;
  let mockMkdirSync: ReturnType<typeof vi.fn>;
  let mockWriteFileSync: ReturnType<typeof vi.fn>;
  let mockDirname: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockCreate = vi.fn();
    mockExistsSync = vi.mocked(existsSync);
    mockMkdirSync = vi.mocked(mkdirSync);
    mockWriteFileSync = vi.mocked(writeFileSync);
    mockDirname = vi.mocked(dirname);

    mockClient = {
      audio: {
        speech: {
          create: mockCreate,
        },
      },
    } as unknown as OpenAI;
  });

  it("日本語テキストを音声に変換できること", async () => {
    // Arrange
    const text = "こんにちは、テストです。";
    const outputPath = "/path/to/output.mp3";
    mockDirname.mockReturnValue("/path/to");
    mockExistsSync.mockReturnValue(true);

    const mockAudioBuffer = Buffer.from("audio data");
    mockCreate.mockResolvedValue({
      arrayBuffer: async () => mockAudioBuffer.buffer,
    });

    // Act
    const result = await textToSpeech(mockClient, text, outputPath);

    // Assert
    expect(result.outputPath).toBe(outputPath);
    expect(mockCreate).toHaveBeenCalledWith({
      model: "tts-1",
      voice: "nova",
      input: text,
      response_format: "mp3",
      speed: 1.0,
    });
    expect(mockWriteFileSync).toHaveBeenCalledWith(
      outputPath,
      expect.any(Buffer)
    );
  });

  it("カスタムオプションを指定できること", async () => {
    // Arrange
    const text = "Hello, this is a test.";
    const outputPath = "/path/to/output.opus";
    const options = {
      voice: "alloy" as const,
      model: "tts-1-hd" as const,
      format: "opus" as const,
      speed: 1.5,
    };
    mockDirname.mockReturnValue("/path/to");
    mockExistsSync.mockReturnValue(true);

    const mockAudioBuffer = Buffer.from("audio data");
    mockCreate.mockResolvedValue({
      arrayBuffer: async () => mockAudioBuffer.buffer,
    });

    // Act
    const result = await textToSpeech(mockClient, text, outputPath, options);

    // Assert
    expect(result.outputPath).toBe(outputPath);
    expect(mockCreate).toHaveBeenCalledWith({
      model: "tts-1-hd",
      voice: "alloy",
      input: text,
      response_format: "opus",
      speed: 1.5,
    });
  });

  it("出力ディレクトリが存在しない場合は作成すること", async () => {
    // Arrange
    const text = "テストテキスト";
    const outputPath = "/new/path/to/output.mp3";
    mockDirname.mockReturnValue("/new/path/to");
    mockExistsSync.mockReturnValue(false);

    const mockAudioBuffer = Buffer.from("audio data");
    mockCreate.mockResolvedValue({
      arrayBuffer: async () => mockAudioBuffer.buffer,
    });

    // Act
    await textToSpeech(mockClient, text, outputPath);

    // Assert
    expect(mockMkdirSync).toHaveBeenCalledWith("/new/path/to", {
      recursive: true,
    });
  });

  it("テキストが空の場合はエラーをスローすること", async () => {
    // Arrange
    const text = "";
    const outputPath = "/path/to/output.mp3";

    // Act & Assert
    await expect(textToSpeech(mockClient, text, outputPath)).rejects.toThrow(
      "テキストが空です"
    );
  });

  it("API呼び出しが失敗した場合はエラーをスローすること", async () => {
    // Arrange
    const text = "テストテキスト";
    const outputPath = "/path/to/output.mp3";
    mockDirname.mockReturnValue("/path/to");
    mockExistsSync.mockReturnValue(true);
    mockCreate.mockRejectedValue(new Error("API error"));

    // Act & Assert
    await expect(textToSpeech(mockClient, text, outputPath)).rejects.toThrow(
      "音声生成に失敗しました"
    );
  });
});

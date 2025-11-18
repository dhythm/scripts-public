import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type OpenAI from "openai";

export type VoiceType = "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer";
export type ModelType = "tts-1" | "tts-1-hd";
export type AudioFormat = "mp3" | "opus" | "aac" | "flac" | "wav" | "pcm";

export interface TextToSpeechOptions {
  voice?: VoiceType;
  model?: ModelType;
  format?: AudioFormat;
  speed?: number;
}

export interface TextToSpeechResult {
  outputPath: string;
  sizeBytes?: number;
}

/**
 * OpenAI Text-to-Speech API を使用してテキストを音声に変換する
 * @param client OpenAI クライアント
 * @param text 音声に変換するテキスト
 * @param outputPath 出力ファイルのパス
 * @param options 音声生成オプション
 * @returns 生成結果
 * @throws テキストが空の場合やAPI呼び出しが失敗した場合
 */
export async function textToSpeech(
  client: OpenAI,
  text: string,
  outputPath: string,
  options: TextToSpeechOptions = {}
): Promise<TextToSpeechResult> {
  // テキストの検証
  if (!text || text.trim().length === 0) {
    throw new Error("テキストが空です");
  }

  // デフォルトのオプション
  const defaultOptions: Required<TextToSpeechOptions> = {
    voice: "nova",
    model: "tts-1",
    format: "mp3",
    speed: 1.0,
  };

  // オプションをマージ
  const mergedOptions = {
    ...defaultOptions,
    ...options,
  };

  // 出力ディレクトリの確認と作成
  const outputDir = dirname(outputPath);
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }

  try {
    // Text-to-Speech API を呼び出し
    const response = await client.audio.speech.create({
      model: mergedOptions.model,
      voice: mergedOptions.voice,
      input: text,
      response_format: mergedOptions.format,
      speed: mergedOptions.speed,
    });

    // 音声データを取得
    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // ファイルに保存
    writeFileSync(outputPath, buffer);

    return {
      outputPath,
      sizeBytes: buffer.length,
    };
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`音声生成に失敗しました: ${error.message}`);
    }
    throw error;
  }
}

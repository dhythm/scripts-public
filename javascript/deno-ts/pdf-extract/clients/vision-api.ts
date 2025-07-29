import { CloudOcrClient } from "../types/index.ts";
import { SimpleGoogleAuth } from "../utils/auth-simple.ts";
import { RetryHandler, RetryableError } from "../utils/retry.ts";
import { encodeBase64Stream } from "../utils/base64.ts";

interface VisionApiRequest {
  requests: Array<{
    inputConfig: {
      content?: string;
      gcsSource?: {
        uri: string;
      };
      mimeType: string;
    };
    features: Array<{
      type: string;
      maxResults?: number;
    }>;
    pages?: number[];
  }>;
}

interface VisionApiResponse {
  responses: Array<{
    fullTextAnnotation?: {
      text: string;
      pages: Array<{
        blocks: unknown[];
      }>;
    };
    error?: {
      code: number;
      message: string;
    };
  }>;
}

export class VisionApiClient implements CloudOcrClient {
  private authHelper: SimpleGoogleAuth;
  private retryHandler: RetryHandler;
  private apiEndpoint = "https://vision.googleapis.com/v1/files:annotate";

  constructor() {
    this.authHelper = new SimpleGoogleAuth();
    this.retryHandler = new RetryHandler();
  }

  async authenticate(): Promise<void> {
    await this.authHelper.loadCredentials();
  }

  async extractTextFromPdf(pdfPath: string): Promise<string> {
    const pdfData = await Deno.readFile(pdfPath);
    
    // 大きなファイルでもスタックオーバーフローを避けるためにチャンク処理
    const base64Content = encodeBase64Stream(pdfData);

    const request: VisionApiRequest = {
      requests: [{
        inputConfig: {
          content: base64Content,
          mimeType: "application/pdf",
        },
        features: [{
          type: "DOCUMENT_TEXT_DETECTION",
        }],
      }],
    };

    return await this.retryHandler.withRetry(async () => {
      const response = await this.makeRequest(request);
      return this.extractTextFromResponse(response);
    });
  }

  private async makeRequest(request: VisionApiRequest): Promise<VisionApiResponse> {
    const authHeaders = await this.authHelper.getAuthHeaders();
    
    const response = await fetch(this.apiEndpoint, {
      method: "POST",
      headers: {
        ...authHeaders,
        "Content-Type": "application/json",
        "x-goog-user-project": this.authHelper.getProjectId(),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      const statusCode = response.status;

      if (statusCode === 429 || statusCode === 503) {
        throw new RetryableError(
          `Vision API レート制限エラー (${statusCode}): ${errorBody}`
        );
      }

      if (statusCode >= 500) {
        throw new RetryableError(
          `Vision API サーバーエラー (${statusCode}): ${errorBody}`
        );
      }

      throw new Error(
        `Vision API エラー (${statusCode}): ${errorBody}`
      );
    }

    return await response.json() as VisionApiResponse;
  }

  private extractTextFromResponse(response: VisionApiResponse): string {
    console.debug("Vision API Response structure:", JSON.stringify(Object.keys(response), null, 2));
    
    if (!response.responses || response.responses.length === 0) {
      throw new Error("Vision API レスポンスが空です");
    }

    const result = response.responses[0];

    if (result.error) {
      throw new Error(
        `Vision API 処理エラー: ${result.error.message} (コード: ${result.error.code})`
      );
    }

    // レスポンスが入れ子になっている場合の処理
    if ((result as any).responses && Array.isArray((result as any).responses)) {
      const texts: string[] = [];
      
      for (const pageResponse of (result as any).responses) {
        if (pageResponse.error) {
          console.warn("ページ処理エラー:", pageResponse.error);
          continue;
        }
        
        if (pageResponse.fullTextAnnotation?.text) {
          texts.push(pageResponse.fullTextAnnotation.text);
        } else if (pageResponse.textAnnotations && pageResponse.textAnnotations.length > 0) {
          // 最初のtextAnnotationには全テキストが含まれる
          texts.push(pageResponse.textAnnotations[0].description);
        }
      }
      
      const combinedText = texts.join("\n\n");
      if (combinedText.trim().length > 0) {
        return combinedText;
      }
    }

    // 単一ページのレスポンス
    if (result.fullTextAnnotation?.text) {
      return result.fullTextAnnotation.text;
    }

    // textAnnotationsからテキストを取得
    if ((result as any).textAnnotations && (result as any).textAnnotations.length > 0) {
      return (result as any).textAnnotations[0].description;
    }

    console.warn("Vision API: テキストが検出されませんでした");
    return "";
  }
}
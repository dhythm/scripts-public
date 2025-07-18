import { CloudOcrClient } from "../types/index.ts";
import { SimpleGoogleAuth } from "../utils/auth-simple.ts";
import { RetryHandler, RetryableError } from "../utils/retry.ts";
import { encodeBase64Stream } from "../utils/base64.ts";

interface VisionApiRequest {
  requests: Array<{
    inputConfig: {
      gcsSource?: {
        uri: string;
      };
      content?: string;
      mimeType: string;
    };
    outputConfig?: {
      gcsDestination?: {
        uri: string;
      };
      batchSize?: number;
    };
    features: Array<{
      type: string;
      maxResults?: number;
    }>;
  }>;
}

interface VisionApiResponse {
  responses: Array<{
    responses?: Array<{
      fullTextAnnotation?: {
        text: string;
        pages?: Array<{
          blocks?: unknown[];
        }>;
      };
      textAnnotations?: Array<{
        description: string;
        locale?: string;
      }>;
      error?: {
        code: number;
        message: string;
      };
    }>;
    error?: {
      code: number;
      message: string;
    };
  }>;
}

export class VisionApiClientV2 implements CloudOcrClient {
  private authHelper: SimpleGoogleAuth;
  private retryHandler: RetryHandler;
  private apiEndpoint = "https://vision.googleapis.com/v1/files:asyncBatchAnnotate";

  constructor() {
    this.authHelper = new SimpleGoogleAuth();
    this.retryHandler = new RetryHandler();
  }

  async authenticate(): Promise<void> {
    await this.authHelper.loadCredentials();
  }

  async extractTextFromPdf(pdfPath: string): Promise<string> {
    const pdfData = await Deno.readFile(pdfPath);
    
    // PDFをページごとに画像に変換してOCRを実行する必要がある
    // Vision APIの同期バッチAPIを使用
    const base64Content = encodeBase64Stream(pdfData);

    // 同期APIを試す
    const syncEndpoint = "https://vision.googleapis.com/v1/files:annotate";
    const request = {
      requests: [{
        inputConfig: {
          content: base64Content,
          mimeType: "application/pdf",
        },
        features: [{
          type: "DOCUMENT_TEXT_DETECTION",
        }],
        // 言語ヒントをrequestレベルに配置
        imageContext: {
          languageHints: ["ja", "en"],
        },
        // ページ範囲を指定（最初の5ページまで - Vision APIの制限）
        pages: Array.from({length: 5}, (_, i) => i + 1),
      }],
    };

    return await this.retryHandler.withRetry(async () => {
      const response = await this.makeRequest(syncEndpoint, request);
      return this.extractTextFromResponse(response);
    });
  }

  private async makeRequest(endpoint: string, request: any): Promise<any> {
    const authHeaders = await this.authHelper.getAuthHeaders();
    
    
    const response = await fetch(endpoint, {
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

      console.error(`Vision API エラー (${statusCode}):`, errorBody);

      if (statusCode === 429 || statusCode === 503) {
        throw new RetryableError(
          `Vision API レート制限エラー (${statusCode})`
        );
      }

      if (statusCode >= 500) {
        throw new RetryableError(
          `Vision API サーバーエラー (${statusCode})`
        );
      }

      throw new Error(
        `Vision API エラー (${statusCode}): ${errorBody}`
      );
    }

    return await response.json();
  }

  private extractTextFromResponse(response: any): string {
    console.debug("Vision API Response structure:", JSON.stringify(Object.keys(response), null, 2));
    
    if (response.responses && response.responses.length > 0) {
      const result = response.responses[0];
      
      if (result.error) {
        throw new Error(
          `Vision API 処理エラー: ${result.error.message} (コード: ${result.error.code})`
        );
      }

      // 複数ページのレスポンスを処理
      if (result.responses && Array.isArray(result.responses)) {
        const texts: string[] = [];
        
        for (const pageResponse of result.responses) {
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
      
      if (result.textAnnotations && result.textAnnotations.length > 0) {
        return result.textAnnotations[0].description;
      }
    }
    
    console.warn("Vision API: テキストが検出されませんでした");
    return "";
  }
}
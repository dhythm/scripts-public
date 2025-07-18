import { CloudOcrClient } from "../types/index.ts";
import { GoogleAuthHelper } from "../utils/auth.ts";
import { RetryHandler, RetryableError } from "../utils/retry.ts";

interface DocumentAiRequest {
  rawDocument: {
    content: string;
    mimeType: string;
  };
}

interface DocumentAiResponse {
  document?: {
    text: string;
    pages: Array<{
      pageNumber: number;
      layout?: {
        textAnchor?: {
          textSegments: Array<{
            startIndex: string;
            endIndex: string;
          }>;
        };
      };
    }>;
  };
  error?: {
    code: number;
    message: string;
    details?: unknown[];
  };
}

export class DocumentAiClient implements CloudOcrClient {
  private authHelper: GoogleAuthHelper;
  private retryHandler: RetryHandler;
  private processorId?: string;
  private location = "us";

  constructor() {
    this.authHelper = new GoogleAuthHelper();
    this.retryHandler = new RetryHandler();
    this.processorId = Deno.env.get("DOCUMENT_AI_PROCESSOR_ID");
  }

  async authenticate(): Promise<void> {
    await this.authHelper.authenticate();

    if (!this.processorId) {
      throw new Error(
        "環境変数 DOCUMENT_AI_PROCESSOR_ID が設定されていません。\n" +
        "Document AI プロセッサIDを設定してください。\n" +
        "プロセッサIDの取得方法は Google Cloud Console の Document AI セクションをご確認ください。"
      );
    }
  }

  async extractTextFromPdf(pdfPath: string): Promise<string> {
    const pdfData = await Deno.readFile(pdfPath);
    const base64Content = btoa(String.fromCharCode(...pdfData));

    const request: DocumentAiRequest = {
      rawDocument: {
        content: base64Content,
        mimeType: "application/pdf",
      },
    };

    return await this.retryHandler.withRetry(async () => {
      const response = await this.makeRequest(request);
      return this.extractTextFromResponse(response);
    });
  }

  private async makeRequest(request: DocumentAiRequest): Promise<DocumentAiResponse> {
    const accessToken = await this.authHelper.getAccessToken();
    const projectId = this.authHelper.getProjectId();
    
    const endpoint = `https://${this.location}-documentai.googleapis.com/v1/projects/${projectId}/locations/${this.location}/processors/${this.processorId}:process`;

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      const statusCode = response.status;

      if (statusCode === 429 || statusCode === 503) {
        throw new RetryableError(
          `Document AI レート制限エラー (${statusCode}): ${errorBody}`
        );
      }

      if (statusCode >= 500) {
        throw new RetryableError(
          `Document AI サーバーエラー (${statusCode}): ${errorBody}`
        );
      }

      throw new Error(
        `Document AI エラー (${statusCode}): ${errorBody}`
      );
    }

    return await response.json() as DocumentAiResponse;
  }

  private extractTextFromResponse(response: DocumentAiResponse): string {
    if (response.error) {
      throw new Error(
        `Document AI 処理エラー: ${response.error.message} (コード: ${response.error.code})`
      );
    }

    if (!response.document || !response.document.text) {
      return "";
    }

    return response.document.text;
  }
}
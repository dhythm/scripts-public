#!/usr/bin/env -S deno run --allow-env --allow-read --allow-net --allow-run

// ===== 型定義 =====
interface DocumentAiRequest {
  rawDocument: {
    content: string;
    mimeType: string;
  };
  // skipHumanReview と imagelessMode を両方設定する必要がある
  skipHumanReview?: boolean;
  imagelessMode?: boolean;
}

interface DocumentAiResponse {
  document?: {
    text: string;
    pages?: Array<{
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

// ===== 定数定義 =====
const DOCUMENT_AI_LOCATION = 'us';

// ===== ユーティリティ関数 =====

/**
 * Base64エンコード
 */
function encodeBase64(data: Uint8Array): string {
  const binary = Array.from(data, (byte) => String.fromCharCode(byte)).join('');
  return btoa(binary);
}

/**
 * 遅延処理
 */
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * リトライ可能なエラーかどうか判定
 */
function isRetryableError(error: Error): boolean {
  const message = error.message.toLowerCase();
  const retryablePatterns = [
    'rate limit',
    'quota exceeded',
    'timeout',
    'network',
    'connection',
    'temporary',
    'unavailable',
    '503',
    '429',
    '500',
    '502',
    '504'
  ];
  
  return retryablePatterns.some(pattern => message.includes(pattern));
}

/**
 * バックオフ付きリトライ処理
 */
async function retryWithBackoff<T>(
  operation: () => Promise<T>,
  maxRetries = 3
): Promise<T> {
  let lastError: Error | undefined;
  let delayMs = 1000;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error as Error;
      
      if (!isRetryableError(error as Error) || attempt === maxRetries) {
        throw error;
      }
      
      console.warn(`リトライ ${attempt + 1}/${maxRetries}: ${lastError.message}`);
      
      await delay(delayMs);
      delayMs = Math.min(delayMs * 2, 30000);
    }
  }
  
  throw lastError || new Error('不明なエラー');
}

// ===== 認証関数 =====

/**
 * gcloud CLIを使用してアクセストークンを取得
 */
async function getAccessTokenFromGcloud(): Promise<string> {
  console.log('gcloud CLIを使用してアクセストークンを取得しています...');
  
  const command = new Deno.Command('gcloud', {
    args: ['auth', 'application-default', 'print-access-token'],
    stdout: 'piped',
    stderr: 'piped',
  });
  
  const { code, stdout, stderr } = await command.output();
  
  if (code !== 0) {
    const errorText = new TextDecoder().decode(stderr);
    throw new Error(`gcloud認証エラー: ${errorText}`);
  }
  
  const accessToken = new TextDecoder().decode(stdout).trim();
  
  if (Deno.env.get("DEBUG")) {
    console.log('アクセストークンを取得しました（最初の20文字）:', accessToken.substring(0, 20) + '...');
  }
  
  return accessToken;
}

/**
 * プロジェクトIDを取得
 */
async function getProjectId(): Promise<string> {
  // 環境変数から取得を試みる
  let projectId = Deno.env.get("GOOGLE_CLOUD_PROJECT");
  
  if (!projectId) {
    // gcloud から取得を試みる
    console.log('gcloud CLIからプロジェクトIDを取得しています...');
    
    const command = new Deno.Command('gcloud', {
      args: ['config', 'get-value', 'project'],
      stdout: 'piped',
      stderr: 'piped',
    });
    
    const { code, stdout } = await command.output();
    
    if (code === 0) {
      projectId = new TextDecoder().decode(stdout).trim();
    }
  }
  
  if (!projectId) {
    throw new Error(
      'プロジェクトIDが見つかりません。\n' +
      '環境変数 GOOGLE_CLOUD_PROJECT を設定するか、\n' +
      'gcloud config set project PROJECT_ID を実行してください。'
    );
  }
  
  return projectId;
}

// ===== Document AI 関数 =====

/**
 * Document AI APIにリクエストを送信
 */
async function makeDocumentAiRequest(params: {
  pdfContent: string;
  accessToken: string;
  projectId: string;
  processorId: string;
  location?: string;
}): Promise<DocumentAiResponse> {
  const {
    pdfContent,
    accessToken,
    projectId,
    processorId,
    location = DOCUMENT_AI_LOCATION
  } = params;
  
  const endpoint = `https://${location}-documentai.googleapis.com/v1/projects/${projectId}/locations/${location}/processors/${processorId}:process`;
  
  const request: DocumentAiRequest = {
    rawDocument: {
      content: pdfContent,
      mimeType: 'application/pdf'
    },
    // imageless mode を有効化（30ページまで処理可能）
    skipHumanReview: true,
    imagelessMode: true  // REST APIではキャメルケースを使用
  };
  
  if (Deno.env.get("DEBUG")) {
    console.log('Document AI エンドポイント:', endpoint);
    console.log('Imageless mode 有効（最大30ページ）');
    console.log('リクエスト:', JSON.stringify(request, null, 2));
  }
  
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(request)
  });
  
  if (!response.ok) {
    const errorBody = await response.text();
    const statusCode = response.status;
    
    if (statusCode === 429 || statusCode === 503 || statusCode >= 500) {
      throw new Error(`Document AI エラー (${statusCode}): ${errorBody}`);
    }
    
    throw new Error(`Document AI エラー (${statusCode}): ${errorBody}`);
  }
  
  return await response.json() as DocumentAiResponse;
}

/**
 * Document AI レスポンスからテキストを抽出
 */
function processDocumentAiResponse(response: DocumentAiResponse): string {
  if (response.error) {
    throw new Error(
      `Document AI 処理エラー: ${response.error.message} (コード: ${response.error.code})`
    );
  }
  
  if (!response.document || !response.document.text) {
    return '';
  }
  
  return response.document.text;
}

// ===== メイン関数 =====

/**
 * PDFファイルからテキストを抽出
 */
async function extractTextFromPdf(pdfPath: string): Promise<string> {
  // 環境変数の確認
  const processorId = Deno.env.get("DOCUMENT_AI_PROCESSOR_ID");
  if (!processorId) {
    throw new Error(
      '環境変数 DOCUMENT_AI_PROCESSOR_ID が設定されていません。\n' +
      'Document AI プロセッサIDを設定してください。\n' +
      'プロセッサIDの取得方法は Google Cloud Console の Document AI セクションをご確認ください。'
    );
  }
  
  // gcloud 認証でアクセストークンを取得
  const accessToken = await getAccessTokenFromGcloud();
  
  // プロジェクトIDの取得
  const projectId = await getProjectId();
  console.log(`プロジェクトID: ${projectId}`);
  
  // PDFファイルの読み込み
  console.log(`PDFファイルを読み込んでいます: ${pdfPath}`);
  const pdfData = await Deno.readFile(pdfPath);
  const pdfContent = encodeBase64(pdfData);
  
  // Document AI APIへのリクエスト
  console.log('Document AI APIでPDFを処理しています...');
  const response = await retryWithBackoff(async () => {
    return await makeDocumentAiRequest({
      pdfContent,
      accessToken,
      projectId,
      processorId
    });
  });
  
  // テキストの抽出
  const extractedText = processDocumentAiResponse(response);
  console.log('テキストの抽出が完了しました。');
  
  return extractedText;
}

/**
 * メインエントリポイント
 */
async function main(): Promise<void> {
  try {
    // コマンドライン引数の処理
    const args = Deno.args;
    
    if (args.length === 0) {
      console.error('使用方法: ./document-ai-pdf-extract-gcloud.ts <PDFファイルパス>');
      console.error('\n前提条件:');
      console.error('  1. gcloud CLI がインストールされている');
      console.error('  2. gcloud auth application-default login を実行済み');
      console.error('  3. 環境変数 DOCUMENT_AI_PROCESSOR_ID が設定されている');
      Deno.exit(1);
    }
    
    const pdfPath = args[0];
    
    // ファイルの存在確認
    try {
      await Deno.stat(pdfPath);
    } catch {
      console.error(`エラー: PDFファイルが見つかりません: ${pdfPath}`);
      Deno.exit(1);
    }
    
    // gcloud の存在確認
    try {
      const command = new Deno.Command('gcloud', {
        args: ['--version'],
        stdout: 'null',
        stderr: 'null',
      });
      await command.output();
    } catch {
      console.error('エラー: gcloud CLI がインストールされていません。');
      console.error('インストール方法: https://cloud.google.com/sdk/docs/install');
      Deno.exit(1);
    }
    
    // テキスト抽出の実行
    const extractedText = await extractTextFromPdf(pdfPath);
    
    // 結果の出力
    console.log('\n===== 抽出されたテキスト =====\n');
    console.log(extractedText);
    
  } catch (error) {
    console.error('エラーが発生しました:', (error as Error).message);
    if (Deno.env.get("DEBUG")) {
      console.error((error as Error).stack);
    }
    Deno.exit(1);
  }
}

// プログラムの実行
if (import.meta.main) {
  main();
}
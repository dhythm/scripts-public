#!/usr/bin/env -S deno run --allow-env --allow-read --allow-net --allow-run

// OpenAI SDKのインポート
import OpenAI from "npm:openai@4.28.0";

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
    text?: string;
    // Document OCR形式 (従来形式)
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
      paragraphs?: Array<{
        layout?: {
          textAnchor?: {
            textSegments: Array<{
              startIndex: string;
              endIndex: string;
            }>;
          };
        };
      }>;
      tables?: Array<{
        layout?: {
          textAnchor?: {
            textSegments: Array<{
              startIndex: string;
              endIndex: string;
            }>;
          };
        };
        headerRows?: Array<{
          cells: Array<{
            layout?: {
              textAnchor?: {
                textSegments: Array<{
                  startIndex: string;
                  endIndex: string;
                }>;
              };
            };
          }>;
        }>;
        bodyRows?: Array<{
          cells: Array<{
            layout?: {
              textAnchor?: {
                textSegments: Array<{
                  startIndex: string;
                  endIndex: string;
                }>;
              };
            };
          }>;
        }>;
      }>;
      lines?: Array<{
        layout?: {
          textAnchor?: {
            textSegments: Array<{
              startIndex: string;
              endIndex: string;
            }>;
          };
        };
      }>;
      blocks?: Array<{
        layout?: {
          textAnchor?: {
            textSegments: Array<{
              startIndex: string;
              endIndex: string;
            }>;
          };
        };
      }>;
    }>;
    // Layout Parser形式 (新形式)
    documentLayout?: {
      blocks: Array<{
        blockId: string;
        pageSpan?: {
          pageStart: number;
          pageEnd: number;
        };
        textBlock?: {
          text?: string;
          type?: string;
        };
        tableBlock?: {
          headerRows?: Array<{
            cells: Array<{
              blocks: Array<{
                blockId: string;
                textBlock?: {
                  text?: string;
                  type?: string;
                };
                pageSpan?: {
                  pageStart: number;
                  pageEnd: number;
                };
              }>;
              rowSpan?: number;
              colSpan?: number;
            }>;
          }>;
          bodyRows?: Array<{
            cells: Array<{
              blocks: Array<{
                blockId: string;
                textBlock?: {
                  text?: string;
                  type?: string;
                };
                pageSpan?: {
                  pageStart: number;
                  pageEnd: number;
                };
              }>;
              rowSpan?: number;
              colSpan?: number;
            }>;
          }>;
          caption?: string;
        };
      }>;
    };
    entities?: Array<{
      type: string;
      mentionText?: string;
      textAnchor?: {
        textSegments: Array<{
          startIndex: string;
          endIndex: string;
        }>;
      };
      properties?: Array<{
        type: string;
        mentionText?: string;
      }>;
    }>;
  };
  error?: {
    code: number;
    message: string;
    details?: unknown[];
  };
}

// 構造化された出力のインターフェース
interface StructuredOutput {
  fullText: string;
  pages: Array<{
    pageNumber: number;
    text: string;
    elements: Array<{
      type: "paragraph" | "table" | "line" | "block";
      content: string;
      metadata?: {
        tableData?: {
          headers: string[][];
          rows: string[][];
        };
      };
    }>;
  }>;
  entities?: Array<{
    type: string;
    text: string;
    properties?: Record<string, string>;
  }>;
}

// ===== 定数定義 =====
const DOCUMENT_AI_LOCATION = "us";

// ===== ユーティリティ関数 =====

/**
 * Base64エンコード
 */
function encodeBase64(data: Uint8Array): string {
  const binary = Array.from(data, (byte) => String.fromCharCode(byte)).join("");
  return btoa(binary);
}

/**
 * 遅延処理
 */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * テキストアンカーからテキストを抽出
 */
function extractTextFromAnchor(
  fullText: string,
  textAnchor?: {
    textSegments: Array<{
      startIndex: string;
      endIndex: string;
    }>;
  }
): string {
  if (
    !textAnchor ||
    !textAnchor.textSegments ||
    textAnchor.textSegments.length === 0
  ) {
    return "";
  }

  return textAnchor.textSegments
    .map((segment) => {
      const start = parseInt(segment.startIndex, 10);
      const end = parseInt(segment.endIndex, 10);
      return fullText.substring(start, end);
    })
    .join("");
}

/**
 * リトライ可能なエラーかどうか判定
 */
function isRetryableError(error: Error): boolean {
  const message = error.message.toLowerCase();
  const retryablePatterns = [
    "rate limit",
    "quota exceeded",
    "timeout",
    "network",
    "connection",
    "temporary",
    "unavailable",
    "503",
    "429",
    "500",
    "502",
    "504",
  ];

  return retryablePatterns.some((pattern) => message.includes(pattern));
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

      console.warn(
        `リトライ ${attempt + 1}/${maxRetries}: ${lastError.message}`
      );

      await delay(delayMs);
      delayMs = Math.min(delayMs * 2, 30000);
    }
  }

  throw lastError || new Error("不明なエラー");
}

// ===== 認証関数 =====

/**
 * gcloud CLIを使用してアクセストークンを取得
 */
async function getAccessTokenFromGcloud(): Promise<string> {
  console.log("gcloud CLIを使用してアクセストークンを取得しています...");

  const command = new Deno.Command("gcloud", {
    args: ["auth", "application-default", "print-access-token"],
    stdout: "piped",
    stderr: "piped",
  });

  const { code, stdout, stderr } = await command.output();

  if (code !== 0) {
    const errorText = new TextDecoder().decode(stderr);
    throw new Error(`gcloud認証エラー: ${errorText}`);
  }

  const accessToken = new TextDecoder().decode(stdout).trim();

  if (Deno.env.get("DEBUG")) {
    console.log(
      "アクセストークンを取得しました（最初の20文字）:",
      accessToken.substring(0, 20) + "..."
    );
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
    console.log("gcloud CLIからプロジェクトIDを取得しています...");

    const command = new Deno.Command("gcloud", {
      args: ["config", "get-value", "project"],
      stdout: "piped",
      stderr: "piped",
    });

    const { code, stdout } = await command.output();

    if (code === 0) {
      projectId = new TextDecoder().decode(stdout).trim();
    }
  }

  if (!projectId) {
    throw new Error(
      "プロジェクトIDが見つかりません。\n" +
        "環境変数 GOOGLE_CLOUD_PROJECT を設定するか、\n" +
        "gcloud config set project PROJECT_ID を実行してください。"
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
    location = DOCUMENT_AI_LOCATION,
  } = params;

  const endpoint = `https://${location}-documentai.googleapis.com/v1/projects/${projectId}/locations/${location}/processors/${processorId}:process`;

  const request: DocumentAiRequest = {
    rawDocument: {
      content: pdfContent,
      mimeType: "application/pdf",
    },
    // imageless mode を有効化（30ページまで処理可能）
    skipHumanReview: true,
    imagelessMode: true, // REST APIではキャメルケースを使用
  };

  if (Deno.env.get("DEBUG")) {
    console.log("Document AI エンドポイント:", endpoint);
    console.log("Imageless mode 有効（最大30ページ）");
    console.log("リクエスト:", JSON.stringify(request, null, 2));
  }

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    const statusCode = response.status;

    if (statusCode === 429 || statusCode === 503 || statusCode >= 500) {
      throw new Error(`Document AI エラー (${statusCode}): ${errorBody}`);
    }

    throw new Error(`Document AI エラー (${statusCode}): ${errorBody}`);
  }

  return (await response.json()) as DocumentAiResponse;
}

/**
 * テーブルデータを抽出
 */
function extractTableData(
  fullText: string,
  table: any
): { headers: string[][]; rows: string[][] } {
  const headers: string[][] = [];
  const rows: string[][] = [];

  // ヘッダー行の抽出
  if (table.headerRows) {
    table.headerRows.forEach((row: any) => {
      const cells: string[] = [];
      if (row.cells) {
        row.cells.forEach((cell: any) => {
          cells.push(extractTextFromAnchor(fullText, cell.layout?.textAnchor));
        });
      }
      headers.push(cells);
    });
  }

  // ボディ行の抽出
  if (table.bodyRows) {
    table.bodyRows.forEach((row: any) => {
      const cells: string[] = [];
      if (row.cells) {
        row.cells.forEach((cell: any) => {
          cells.push(extractTextFromAnchor(fullText, cell.layout?.textAnchor));
        });
      }
      rows.push(cells);
    });
  }

  return { headers, rows };
}

/**
 * テーブルの内容が重複しているかチェック
 */
function isTableDuplicate(
  table1: { headers: string[][]; rows: string[][] },
  table2: { headers: string[][]; rows: string[][] }
): boolean {
  // ヘッダーの比較
  if (table1.headers.length !== table2.headers.length) return false;
  for (let i = 0; i < table1.headers.length; i++) {
    if (table1.headers[i].join("|") !== table2.headers[i].join("|"))
      return false;
  }

  // 行数が大きく異なる場合は別のテーブルと判断
  if (Math.abs(table1.rows.length - table2.rows.length) > 2) return false;

  // 最初の数行を比較して同じかチェック
  const compareRows = Math.min(3, table1.rows.length, table2.rows.length);
  for (let i = 0; i < compareRows; i++) {
    if (table1.rows[i].join("|") !== table2.rows[i].join("|")) return false;
  }

  return true;
}

/**
 * Document AI レスポンスから構造化されたデータを抽出
 */
function processDocumentAiResponse(
  response: DocumentAiResponse
): StructuredOutput {
  if (response.error) {
    throw new Error(
      `Document AI 処理エラー: ${response.error.message} (コード: ${response.error.code})`
    );
  }

  if (!response.document) {
    return {
      fullText: "",
      pages: [],
      entities: [],
    };
  }

  // 形式を判定して適切な処理関数を呼び出す
  if (response.document.documentLayout) {
    console.log("Layout Parser形式を検出しました。");
    return processLayoutParserResponse(response);
  } else if (response.document.pages) {
    console.log("Document OCR形式を検出しました。");
    return processDocumentOcrResponse(response);
  } else {
    console.warn("不明なDocument AI形式です。基本的な処理を行います。");
    // 基本的な処理（テキストとエンティティのみ）
    return {
      fullText: response.document.text || "",
      pages: [],
      entities: response.document.entities?.map((entity) => ({
        type: entity.type,
        text: entity.mentionText || "",
        properties: entity.properties?.reduce((acc, prop) => {
          acc[prop.type] = prop.mentionText || "";
          return acc;
        }, {} as Record<string, string>),
      })) || [],
    };
  }
}

/**
 * Document OCR形式（従来形式）のレスポンスを処理
 */
function processDocumentOcrResponse(
  response: DocumentAiResponse
): StructuredOutput {
  if (!response.document) {
    return {
      fullText: "",
      pages: [],
      entities: [],
    };
  }

  const fullText = response.document.text || "";
  const output: StructuredOutput = {
    fullText,
    pages: [],
    entities: [],
  };

  // エンティティの抽出
  if (response.document.entities) {
    output.entities = response.document.entities.map((entity) => ({
      type: entity.type,
      text:
        entity.mentionText ||
        extractTextFromAnchor(fullText, entity.textAnchor),
      properties: entity.properties?.reduce((acc, prop) => {
        acc[prop.type] = prop.mentionText || "";
        return acc;
      }, {} as Record<string, string>),
    }));
  }

  // 処理済みテーブルを追跡
  const processedTables: Array<{ headers: string[][]; rows: string[][] }> = [];

  // ページごとの構造化データの抽出
  if (response.document.pages) {
    response.document.pages.forEach((page, pageIndex) => {
      const pageData = {
        pageNumber: pageIndex + 1,
        text: "",
        elements: [] as any[],
      };

      // ページテキストの抽出
      if (page.layout?.textAnchor) {
        pageData.text = extractTextFromAnchor(fullText, page.layout.textAnchor);
      }

      // 段落の抽出
      if (page.paragraphs) {
        page.paragraphs.forEach((paragraph) => {
          const text = extractTextFromAnchor(
            fullText,
            paragraph.layout?.textAnchor
          );
          if (text) {
            pageData.elements.push({
              type: "paragraph",
              content: text,
            });
          }
        });
      }

      // テーブルの抽出（重複チェック付き）
      if (page.tables) {
        page.tables.forEach((table) => {
          const tableText = extractTextFromAnchor(
            fullText,
            table.layout?.textAnchor
          );
          const tableData = extractTableData(fullText, table);

          // 重複チェック
          const isDuplicate = processedTables.some((processed) =>
            isTableDuplicate(processed, tableData)
          );

          if (!isDuplicate) {
            processedTables.push(tableData);
            pageData.elements.push({
              type: "table",
              content: tableText,
              metadata: {
                tableData,
              },
            });
          }
        });
      }

      // 行の抽出（段落がない場合のフォールバック）
      if (!page.paragraphs && page.lines) {
        page.lines.forEach((line) => {
          const text = extractTextFromAnchor(fullText, line.layout?.textAnchor);
          if (text) {
            pageData.elements.push({
              type: "line",
              content: text,
            });
          }
        });
      }

      // ブロックの抽出（段落も行もない場合のフォールバック）
      if (!page.paragraphs && !page.lines && page.blocks) {
        page.blocks.forEach((block) => {
          const text = extractTextFromAnchor(
            fullText,
            block.layout?.textAnchor
          );
          if (text) {
            pageData.elements.push({
              type: "block",
              content: text,
            });
          }
        });
      }

      output.pages.push(pageData);
    });
  }

  return output;
}

/**
 * Layout Parser形式のテーブルセルからテキストを抽出
 */
function extractTextFromLayoutTableCell(
  cell: any
): string {
  if (!cell.blocks || cell.blocks.length === 0) {
    return "";
  }
  
  return cell.blocks
    .map((block: any) => {
      if (block.textBlock && block.textBlock.text) {
        return block.textBlock.text;
      }
      return "";
    })
    .filter((text: string) => text)
    .join(" ");
}

/**
 * Layout Parser形式のテーブルを処理
 */
function processLayoutTableBlock(
  tableBlock: any
): { headers: string[][]; rows: string[][] } {
  const headers: string[][] = [];
  const rows: string[][] = [];

  // ヘッダー行の処理
  if (tableBlock.headerRows) {
    tableBlock.headerRows.forEach((row: any) => {
      const headerRow: string[] = [];
      if (row.cells) {
        row.cells.forEach((cell: any) => {
          headerRow.push(extractTextFromLayoutTableCell(cell));
        });
      }
      headers.push(headerRow);
    });
  }

  // ボディ行の処理
  if (tableBlock.bodyRows) {
    tableBlock.bodyRows.forEach((row: any) => {
      const bodyRow: string[] = [];
      if (row.cells) {
        row.cells.forEach((cell: any) => {
          bodyRow.push(extractTextFromLayoutTableCell(cell));
        });
      }
      rows.push(bodyRow);
    });
  }

  return { headers, rows };
}

/**
 * Layout Parser形式のレスポンスを処理
 */
function processLayoutParserResponse(
  response: DocumentAiResponse
): StructuredOutput {
  if (!response.document || !response.document.documentLayout) {
    return {
      fullText: "",
      pages: [],
      entities: [],
    };
  }

  const output: StructuredOutput = {
    fullText: "",
    pages: [],
    entities: [],
  };

  // ページごとのブロックを収集
  const pageBlocks: Map<number, any[]> = new Map();
  
  response.document.documentLayout.blocks.forEach((block) => {
    const pageNum = block.pageSpan?.pageStart || 1;
    if (!pageBlocks.has(pageNum)) {
      pageBlocks.set(pageNum, []);
    }
    pageBlocks.get(pageNum)!.push(block);
  });

  // 全テキストを収集（ページ順）
  const allTexts: string[] = [];
  
  // ページごとに処理
  const sortedPageNumbers = Array.from(pageBlocks.keys()).sort((a, b) => a - b);
  
  sortedPageNumbers.forEach((pageNum) => {
    const blocks = pageBlocks.get(pageNum) || [];
    const pageData = {
      pageNumber: pageNum,
      text: "",
      elements: [] as any[],
    };

    const pageTexts: string[] = [];

    blocks.forEach((block) => {
      if (block.textBlock) {
        const text = block.textBlock.text || "";
        if (text) {
          pageTexts.push(text);
          allTexts.push(text);
          
          pageData.elements.push({
            type: block.textBlock.type === "paragraph" ? "paragraph" : "block",
            content: text,
          });
        }
      } else if (block.tableBlock) {
        const tableData = processLayoutTableBlock(block.tableBlock);
        
        // テーブルのテキスト表現を生成
        const tableText = [
          ...tableData.headers.map(row => row.join(" | ")),
          ...tableData.rows.map(row => row.join(" | "))
        ].join("\n");
        
        if (tableText) {
          pageTexts.push(tableText);
          allTexts.push(tableText);
        }
        
        pageData.elements.push({
          type: "table",
          content: tableText,
          metadata: {
            tableData,
          },
        });
      }
    });

    pageData.text = pageTexts.join("\n");
    output.pages.push(pageData);
  });

  output.fullText = allTexts.join("\n");

  // エンティティの処理（もし存在する場合）
  if (response.document.entities) {
    output.entities = response.document.entities.map((entity) => ({
      type: entity.type,
      text: entity.mentionText || "",
      properties: entity.properties?.reduce((acc, prop) => {
        acc[prop.type] = prop.mentionText || "";
        return acc;
      }, {} as Record<string, string>),
    }));
  }

  return output;
}

/**
 * 構造化データをマークダウン変換用の形式で出力
 */
function formatStructuredOutput(data: StructuredOutput): string {
  const output: string[] = [];

  // メタデータ
  output.push("=== Document AI 構造化出力 ===");
  output.push(`総ページ数: ${data.pages.length}`);
  output.push(`エンティティ数: ${data.entities?.length || 0}`);
  output.push("");

  // ページごとの構造化データ
  data.pages.forEach((page) => {
    output.push(`\n--- ページ ${page.pageNumber} ---`);

    if (page.elements.length === 0) {
      output.push("[構造要素が検出されませんでした]");
    } else {
      page.elements.forEach((element, index) => {
        switch (element.type) {
          case "table":
            output.push(`\n[表 ${index + 1}]`);
            if (element.metadata?.tableData) {
              const { headers, rows } = element.metadata.tableData;

              // ヘッダー
              if (headers.length > 0) {
                output.push("ヘッダー:");
                headers.forEach((row) => {
                  output.push("  " + row.join(" | "));
                });
              }

              // 行
              if (rows.length > 0) {
                output.push("データ:");
                rows.forEach((row) => {
                  output.push("  " + row.join(" | "));
                });
              }
            } else {
              output.push(element.content);
            }
            break;

          case "paragraph":
            output.push(`\n[段落 ${index + 1}]`);
            output.push(element.content);
            break;

          case "line":
            output.push(element.content);
            break;

          case "block":
            output.push(`\n[ブロック ${index + 1}]`);
            output.push(element.content);
            break;
        }
      });
    }
  });

  // エンティティ
  if (data.entities && data.entities.length > 0) {
    output.push("\n\n=== 検出されたエンティティ ===");
    data.entities.forEach((entity) => {
      output.push(`\n種類: ${entity.type}`);
      output.push(`テキスト: ${entity.text}`);
      if (entity.properties && Object.keys(entity.properties).length > 0) {
        output.push("プロパティ:");
        Object.entries(entity.properties).forEach(([key, value]) => {
          output.push(`  ${key}: ${value}`);
        });
      }
    });
  }

  return output.join("\n");
}

// ===== OpenAI 関連関数 =====

/**
 * マークダウン変換用のプロンプトを生成
 */
function createMarkdownPrompt(data: StructuredOutput): string {
  const prompt = `以下は日本語PDFドキュメントをOCRで構造化したデータです。このデータを元の文書の構造と内容を正確に再現したマークダウン形式に変換してください。

極めて重要な指示：
- **与えられた構造化データに含まれる内容のみを出力し、データに存在しない要素は一切追加しない**
- **各ページの内容は構造化データに記載されている通りに1回だけ出力する**
- **構造化データのページ順序と要素順序を厳密に守る**
- **同じ表や内容を複数回出力しない**

階層構造の指示（必ず従ってください）：
- 「新医薬品一覧表」のような主要タイトルは # （レベル1）
- 「新医薬品の薬価算定について」のような大セクションは ## （レベル2）
- 「整理番号 XX-X-X-X」は ### （レベル3）
- 「薬効分類」「成分名」「算定方式」「主な用法・用量」などは #### （レベル4）
- 「有用性加算」「補正加算」「市場規模予測」などのサブセクションも #### （レベル4）
- さらに詳細な項目（「イ. 効能・効果」「ロ. 薬理作用」など）は ##### （レベル5）

要件：
1. 表は適切なマークダウンテーブル形式で表現
2. 段落は適切に改行で区切る
3. 上記の階層構造ルールに従って見出しレベルを設定
4. リストと思われる部分は適切なリスト形式を使用
5. 元の文書の階層構造と情報の関係性を維持
6. OCRの誤認識と思われる部分は文脈から修正

テーブル処理の特別な指示：
- 同じ医薬品の複数規格（例：50mg錠、100mg錠）は同じ行にまとめる
- 空のセルは適切に処理し、行の構造を維持
- 整理番号（例：整理番号 22-4-内-1）の直前に個別の表を挿入しない
- 整理番号から始まるセクションは、その整理番号から直接開始する
- PDFで同じセル内に複数行ある場合は、マークダウンでも同じセルにまとめる（必要に応じて<br>タグを使用）

情報の整理：
- 各医薬品のセクションは整理番号から始める
- 薬効分類、成分名、新薬収載希望者、販売名などは明確に区別して記載
- 算定方式とその詳細（比較薬、補正加算など）は階層的に整理
- 「薬価算定組織における検討結果のまとめ」は独立したセクションとして扱う

禁止事項：
- ファイル名やファイルパスは一切出力しない
- コードブロック記号（\`\`\`markdown等）は使用しない
- 構造化データに存在しない表や内容を追加しない
- 同じタイトル（例：「新医薬品一覧表」）を持つセクションを複数回出力しない

構造化データ：
${JSON.stringify(data, null, 2)}

純粋なマークダウンコンテンツのみを出力してください：`;

  return prompt;
}

/**
 * 大きなドキュメントを分割処理
 */
async function processLargeDocument(
  openai: OpenAI,
  data: StructuredOutput,
  model: string
): Promise<string> {
  const results: string[] = [];
  const maxPagesPerRequest = 5; // 一度に処理するページ数

  for (let i = 0; i < data.pages.length; i += maxPagesPerRequest) {
    const pageChunk = data.pages.slice(i, i + maxPagesPerRequest);
    const chunkData: StructuredOutput = {
      ...data,
      pages: pageChunk,
    };

    console.error(
      `処理中: ページ ${i + 1}-${Math.min(
        i + maxPagesPerRequest,
        data.pages.length
      )} / ${data.pages.length}`
    );

    const prompt = createMarkdownPrompt(chunkData);
    const response = await openai.chat.completions.create({
      model: model,
      messages: [
        {
          role: "system",
          content:
            "あなたは日本語文書のOCR結果を正確なマークダウンに変換する専門家です。極めて重要：与えられた構造化データに含まれる内容のみを出力し、データに存在しない要素は一切追加しないでください。各ページの内容は1回だけ出力し、同じ表や内容を複数回出力しないでください。階層構造は必ず指示に従ってください（#: 主要タイトル、##: 大セクション、###: 整理番号、####: サブセクション、#####: 詳細項目）。ファイル名やコードブロック記号は使用せず、純粋なマークダウンコンテンツのみを出力してください。",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 0.1,
      max_tokens: 4000,
    });

    const content = response.choices[0]?.message?.content || "";
    results.push(cleanupMarkdownOutput(content));
  }

  return results.join("\n\n---\n\n"); // ページ区切りを追加
}

/**
 * マークダウン出力をクリーンアップ
 */
function cleanupMarkdownOutput(content: string): string {
  // ファイル名参照を削除（例: @javascript/000928313.md）
  let cleaned = content.replace(/^@[^\s]+\s*/gm, "");

  // コードブロック記号を削除
  cleaned = cleaned.replace(/^```markdown\s*$/gm, "");
  cleaned = cleaned.replace(/^```\s*$/gm, "");

  // 重複した「新医薬品一覧表」セクションを削除（最初のものだけ残す）
  // 最初の出現位置を特定
  const firstTableMatch = cleaned.match(/^# 新医薬品一覧表.*$/m);
  if (firstTableMatch) {
    const firstTableIndex = cleaned.indexOf(firstTableMatch[0]);
    const beforeFirstTable = cleaned.substring(
      0,
      firstTableIndex + firstTableMatch[0].length
    );
    let afterFirstTable = cleaned.substring(
      firstTableIndex + firstTableMatch[0].length
    );

    // 2回目以降の「新医薬品一覧表」セクション全体を削除
    // セクションは次の「#」レベルの見出しまたは「---」まで続く
    afterFirstTable = afterFirstTable.replace(
      /\n# 新医薬品一覧表[\s\S]*?(?=\n#{1,3}\s|^---$|\n## 新医薬品の薬価算定について|$)/gm,
      ""
    );

    cleaned = beforeFirstTable + afterFirstTable;
  }

  // 整理番号の前に現れる不要な表を検出して削除
  // パターン: 表の後に整理番号が続く場合
  cleaned = cleaned.replace(
    /(\|[^\n]+\|\n\|[-\s|]+\|\n(?:\|[^\n]+\|\n)*)\n+(整理番号\s+\d{2}-\d+-[内外注]-\d+)/gm,
    "\n\n$2"
  );

  // 整理番号の前に現れる薬効分類の表を削除
  // パターン: | 薬効分類 | 成分名 | ... で始まる表
  cleaned = cleaned.replace(
    /\n\|[\s]*薬効分類[\s]*\|[^\n]+\|\n\|[-\s|]+\|\n(?:\|[^\n]+\|\n)*\n+(整理番号\s+\d{2}-\d+-[内外注]-\d+)/gm,
    "\n\n$2"
  );

  // 連続する改行を正規化（3つ以上の改行を2つに）
  cleaned = cleaned.replace(/\n{3,}/g, "\n\n");

  // 先頭と末尾の空白を削除
  cleaned = cleaned.trim();

  return cleaned;
}

/**
 * OpenAI APIを使用してマークダウンに変換
 */
async function convertToMarkdownWithOpenAI(
  data: StructuredOutput,
  model: string = "gpt-4o-mini"
): Promise<string> {
  // APIキーの確認
  const apiKey = Deno.env.get("OPENAI_API_KEY");
  if (!apiKey) {
    throw new Error(
      "環境変数 OPENAI_API_KEY が設定されていません。\n" +
        "OpenAI APIキーを設定してください。"
    );
  }

  const openai = new OpenAI({ apiKey });

  console.error("OpenAI APIを使用してマークダウンに変換中...");
  console.error(`使用モデル: ${model}`);

  try {
    let rawContent: string;

    // ドキュメントが大きい場合は分割処理
    if (data.pages.length > 10) {
      console.error("大きなドキュメントのため、分割処理を実行します。");
      rawContent = await processLargeDocument(openai, data, model);
    } else {
      // 通常の処理
      const prompt = createMarkdownPrompt(data);

      const response = await openai.chat.completions.create({
        model: model,
        messages: [
          {
            role: "system",
            content:
              "あなたは日本語文書のOCR結果を正確なマークダウンに変換する専門家です。極めて重要：与えられた構造化データに含まれる内容のみを出力し、データに存在しない要素は一切追加しないでください。各ページの内容は1回だけ出力し、同じ表や内容を複数回出力しないでください。ファイル名やコードブロック記号は使用せず、純粋なマークダウンコンテンツのみを出力してください。",
          },
          {
            role: "user",
            content: prompt,
          },
        ],
        temperature: 0.1, // 低い温度で一貫性を保つ
        max_tokens: 4000,
      });

      const content = response.choices[0]?.message?.content;
      if (!content) {
        throw new Error("OpenAI APIからの応答が空です");
      }

      rawContent = content;
    }

    // 出力をクリーンアップ
    return cleanupMarkdownOutput(rawContent);
  } catch (error) {
    if (error instanceof Error) {
      // レート制限エラーの場合
      if (error.message.includes("rate limit")) {
        throw new Error(
          "OpenAI APIのレート制限に達しました。しばらく待ってから再試行してください。"
        );
      }
      // その他のエラー
      throw new Error(`OpenAI API エラー: ${error.message}`);
    }
    throw error;
  }
}

// ===== メイン関数 =====

/**
 * PDFファイルから構造化されたデータを抽出
 */
async function extractTextFromPdf(pdfPath: string): Promise<{
  structuredData: StructuredOutput;
  rawResponse?: DocumentAiResponse;
}> {
  // 環境変数の確認
  const processorId = Deno.env.get("DOCUMENT_AI_PROCESSOR_ID");
  if (!processorId) {
    throw new Error(
      "環境変数 DOCUMENT_AI_PROCESSOR_ID が設定されていません。\n" +
        "Document AI プロセッサIDを設定してください。\n" +
        "プロセッサIDの取得方法は Google Cloud Console の Document AI セクションをご確認ください。"
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
  console.log("Document AI APIでPDFを処理しています...");
  const response = await retryWithBackoff(async () => {
    return await makeDocumentAiRequest({
      pdfContent,
      accessToken,
      projectId,
      processorId,
    });
  });

  // 構造化データの抽出
  const structuredData = processDocumentAiResponse(response);
  console.log("構造化データの抽出が完了しました。");

  // デバッグ情報を出力
  if (Deno.env.get("DEBUG")) {
    console.log(`総ページ数: ${structuredData.pages.length}`);
    console.log(`総テキスト長: ${structuredData.fullText.length} 文字`);
    
    structuredData.pages.forEach((page) => {
      console.log(`ページ ${page.pageNumber}: ${page.elements.length} 要素`);
      const elementTypes = page.elements.reduce((acc, elem) => {
        acc[elem.type] = (acc[elem.type] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      
      Object.entries(elementTypes).forEach(([type, count]) => {
        console.log(`  - ${type}: ${count}`);
      });
    });
    
    if (structuredData.entities && structuredData.entities.length > 0) {
      console.log(`エンティティ数: ${structuredData.entities.length}`);
    }
    
    // 生のレスポンスの形式も出力
    if (response.document?.documentLayout) {
      console.log(`Layout Parserブロック数: ${response.document.documentLayout.blocks.length}`);
    } else if (response.document?.pages) {
      console.log(`Document OCRページ数: ${response.document.pages.length}`);
    }
  }

  return {
    structuredData,
    rawResponse: response,
  };
}

/**
 * 出力ファイル名を生成
 */
function generateOutputFilename(pdfPath: string, extension: string): string {
  // パスの区切り文字を判定
  const lastSlashIndex = pdfPath.lastIndexOf("/");
  const lastBackslashIndex = pdfPath.lastIndexOf("\\");
  const separatorIndex = Math.max(lastSlashIndex, lastBackslashIndex);

  let dir = "";
  let filename = pdfPath;

  if (separatorIndex !== -1) {
    dir = pdfPath.substring(0, separatorIndex);
    filename = pdfPath.substring(separatorIndex + 1);
  }

  const nameWithoutExt = filename.replace(/\.pdf$/i, "");

  if (dir) {
    return `${dir}/${nameWithoutExt}.${extension}`;
  } else {
    return `${nameWithoutExt}.${extension}`;
  }
}

/**
 * メインエントリポイント
 */
async function main(): Promise<void> {
  try {
    // コマンドライン引数の処理
    const args = Deno.args;

    if (args.length === 0 || args.includes("--help")) {
      console.error(
        "使用方法: ./document-ai-pdf-extract.ts <PDFファイルパス> [オプション]"
      );
      console.error("\nオプション:");
      console.error("  --json       JSON形式で保存（構造化データ）");
      console.error("  --raw        Document AI OCR結果をそのまま保存（生データ）");
      console.error(
        "  --markdown   OpenAI APIを使用してマークダウンに変換して保存"
      );
      console.error(
        "  --model      使用するOpenAIモデル（デフォルト: gpt-4o-mini）"
      );
      console.error(
        "  --output     出力ファイル名を指定（省略時は入力ファイル名を使用）"
      );
      console.error("  --help       このヘルプを表示");
      console.error("\n環境変数:");
      console.error(
        "  DOCUMENT_AI_PROCESSOR_ID    Document AI プロセッサID（必須）"
      );
      console.error(
        "  OPENAI_API_KEY              OpenAI APIキー（--markdown使用時に必須）"
      );
      console.error("  DEBUG                       デバッグ情報を表示");
      console.error("\n前提条件:");
      console.error("  1. gcloud CLI がインストールされている");
      console.error("  2. gcloud auth application-default login を実行済み");
      console.error("  3. Document AI API が有効化されている");
      console.error("\n使用例:");
      console.error("  # 構造化テキストをtxtファイルに保存");
      console.error("  ./document-ai-pdf-extract.ts input.pdf");
      console.error("  # → input.txt が生成される");
      console.error("");
      console.error("  # マークダウンに変換してmdファイルに保存");
      console.error("  ./document-ai-pdf-extract.ts input.pdf --markdown");
      console.error("  # → input.md が生成される");
      console.error("");
      console.error("  # JSON形式でjsonファイルに保存");
      console.error("  ./document-ai-pdf-extract.ts input.pdf --json");
      console.error("  # → input.json が生成される");
      console.error("");
      console.error("  # Document AI OCR結果をそのまま保存");
      console.error("  ./document-ai-pdf-extract.ts input.pdf --raw");
      console.error("  # → input.raw.json が生成される");
      console.error("");
      console.error("  # 出力ファイル名を指定");
      console.error(
        "  ./document-ai-pdf-extract.ts input.pdf --markdown --output output.md"
      );
      console.error("\n注意事項:");
      console.error("  - Imageless mode により最大30ページまで処理可能");
      console.error("  - 大きなPDFファイルは処理に時間がかかります");
      console.error("  - OpenAI APIの使用には料金が発生します");
      console.error("  - 既存のファイルは上書きされます");
      Deno.exit(1);
    }

    const pdfPath = args.find((arg: string) => !arg.startsWith("--")) || "";

    // ファイルの存在確認
    try {
      await Deno.stat(pdfPath);
    } catch {
      console.error(`エラー: PDFファイルが見つかりません: ${pdfPath}`);
      Deno.exit(1);
    }

    // gcloud の存在確認
    try {
      const command = new Deno.Command("gcloud", {
        args: ["--version"],
        stdout: "null",
        stderr: "null",
      });
      await command.output();
    } catch {
      console.error("エラー: gcloud CLI がインストールされていません。");
      console.error(
        "インストール方法: https://cloud.google.com/sdk/docs/install"
      );
      Deno.exit(1);
    }

    // 構造化データ抽出の実行
    const { structuredData, rawResponse } = await extractTextFromPdf(pdfPath);

    // 出力ファイル名の決定
    const outputIndex = args.indexOf("--output");
    let outputPath: string;
    let content: string;

    // 出力形式の判定と内容の生成
    if (args.includes("--raw")) {
      // Document AI の生のレスポンスを保存
      content = JSON.stringify(rawResponse, null, 2);
      outputPath =
        outputIndex !== -1 && args[outputIndex + 1]
          ? args[outputIndex + 1]
          : generateOutputFilename(pdfPath, "raw.json");
    } else if (args.includes("--markdown")) {
      // マークダウン形式で保存
      try {
        // モデルの取得
        const modelIndex = args.indexOf("--model");
        const model =
          modelIndex !== -1 && args[modelIndex + 1]
            ? args[modelIndex + 1]
            : "gpt-4o-mini";

        content = await convertToMarkdownWithOpenAI(structuredData, model);
        outputPath =
          outputIndex !== -1 && args[outputIndex + 1]
            ? args[outputIndex + 1]
            : generateOutputFilename(pdfPath, "md");
      } catch (error) {
        console.error("マークダウン変換エラー:", (error as Error).message);
        Deno.exit(1);
      }
    } else if (args.includes("--json")) {
      // JSON形式で保存
      content = JSON.stringify(structuredData, null, 2);
      outputPath =
        outputIndex !== -1 && args[outputIndex + 1]
          ? args[outputIndex + 1]
          : generateOutputFilename(pdfPath, "json");
    } else {
      // デフォルト：構造化テキスト形式で保存
      content = formatStructuredOutput(structuredData);
      outputPath =
        outputIndex !== -1 && args[outputIndex + 1]
          ? args[outputIndex + 1]
          : generateOutputFilename(pdfPath, "txt");
    }

    // ファイルに保存
    try {
      await Deno.writeTextFile(outputPath, content);
      console.log(`結果を保存しました: ${outputPath}`);

      // ファイルサイズを表示
      const fileInfo = await Deno.stat(outputPath);
      const fileSizeKB = (fileInfo.size / 1024).toFixed(2);
      console.log(`ファイルサイズ: ${fileSizeKB} KB`);
    } catch (error) {
      console.error(
        `ファイルの保存に失敗しました: ${(error as Error).message}`
      );
      Deno.exit(1);
    }
  } catch (error) {
    console.error("エラーが発生しました:", (error as Error).message);
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

#!/usr/bin/env -S deno run --allow-read --allow-write --allow-env --allow-net --allow-run

import { ConsoleLogger } from "./utils/logger.ts";
import { SimpleGoogleAuth } from "./utils/auth-simple.ts";
import { encodeBase64Stream } from "./utils/base64.ts";
import { RetryHandler, RetryableError } from "./utils/retry.ts";
import { basename, dirname, join } from "std/path/mod.ts";

// ===== 型定義 =====

interface MedicineInfo {
  no: string;
  name: string;
  company: string;
  ingredient: string;
  approvalCategory: string;
  specifications: Array<{
    spec: string;
    price: string;
  }>;
  calculationMethod: string;
  additionalBenefits: string[];
  category: string;
  categoryDetail: string;
}

interface ProcessingOptions {
  outputDir?: string;
  keepStructureJson?: boolean;
  verbose?: boolean;
}

// ===== 日本語テキスト整形クラス =====

class JapaneseTextCleaner {
  // 余計なスペースを除去
  removeExtraSpaces(text: string): string {
    // 日本語文字間のスペースを削除
    let cleaned = text.replace(/([ぁ-んァ-ヶー一-龠々])\s+([ぁ-んァ-ヶー一-龠々])/g, '$1$2');
    
    // 日本語と数字の間のスペースを削除
    cleaned = cleaned.replace(/([ぁ-んァ-ヶー一-龠々])\s+(\d)/g, '$1$2');
    cleaned = cleaned.replace(/(\d)\s+([ぁ-んァ-ヶー一-龠々])/g, '$1$2');
    
    // 括弧の前後の余計なスペースを削除
    cleaned = cleaned.replace(/\s*([（(])\s*/g, '$1');
    cleaned = cleaned.replace(/\s*([）)])\s*/g, '$1');
    
    // パイプ記号を削除
    cleaned = cleaned.replace(/\|/g, '');
    
    // 複数の連続するスペースを1つに
    cleaned = cleaned.replace(/\s+/g, ' ');
    
    return cleaned.trim();
  }

  // 特定のパターンを整形
  formatSpecificPatterns(text: string): string {
    // 薬効分類の整形
    text = text.replace(/(内|注|外)\s+(\d+)/g, '$1$2');
    
    // 価格の整形
    text = text.replace(/(\d+\.?\d*)\s*円/g, '$1円');
    
    // パーセントの整形
    text = text.replace(/([A-Z])\s*=\s*(\d+)\s*%/g, '$1=$2%');
    
    // mg錠の整形
    text = text.replace(/(\d+mg)\s*(\d+)\s*(錠|瓶|包|キット|筒)/g, '$1$2$3');
    
    return text;
  }

  clean(text: string): string {
    let cleaned = this.removeExtraSpaces(text);
    cleaned = this.formatSpecificPatterns(cleaned);
    return cleaned;
  }
}

// ===== Vision API クライアント =====

class VisionApiClient {
  private authHelper: SimpleGoogleAuth;
  private retryHandler: RetryHandler;
  private logger: ConsoleLogger;
  private textCleaner: JapaneseTextCleaner;

  constructor(verbose: boolean = false) {
    this.authHelper = new SimpleGoogleAuth(verbose);
    this.retryHandler = new RetryHandler();
    this.logger = new ConsoleLogger(verbose);
    this.textCleaner = new JapaneseTextCleaner();
  }

  async authenticate(): Promise<void> {
    await this.authHelper.loadCredentials();
  }

  async analyzeDocument(pdfPath: string): Promise<any> {
    const pdfData = await Deno.readFile(pdfPath);
    const base64Content = encodeBase64Stream(pdfData);

    const request = {
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
      const authHeaders = await this.authHelper.getAuthHeaders();
      
      const response = await fetch("https://vision.googleapis.com/v1/files:annotate", {
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
          throw new RetryableError(`Vision API レート制限エラー (${statusCode})`);
        }

        throw new Error(`Vision API エラー (${statusCode}): ${errorBody}`);
      }

      return await response.json();
    });
  }
}

// ===== 医薬品情報抽出クラス =====

class MedicineExtractor {
  private logger: ConsoleLogger;
  private textCleaner: JapaneseTextCleaner;

  constructor(verbose: boolean = false) {
    this.logger = new ConsoleLogger(verbose);
    this.textCleaner = new JapaneseTextCleaner();
  }

  extractFromVisionResponse(response: any): MedicineInfo[] {
    const medicines: MedicineInfo[] = [];
    
    // レスポンスから fullTextAnnotation を取得
    let annotation;
    if (response.responses?.[0]?.responses?.[0]?.fullTextAnnotation) {
      annotation = response.responses[0].responses[0].fullTextAnnotation;
      this.logger.info("レスポンス構造: responses[0].responses[0].fullTextAnnotation");
    } else if (response.responses?.[0]?.fullTextAnnotation) {
      annotation = response.responses[0].fullTextAnnotation;
      this.logger.info("レスポンス構造: responses[0].fullTextAnnotation");
    } else {
      this.logger.warn("構造データが見つかりません");
      return medicines;
    }

    // テキスト全体を取得
    const fullText = annotation.text || '';
    this.logger.info(`取得したテキストの長さ: ${fullText.length}文字`);
    const lines = fullText.split('\n').filter(line => line.trim());
    this.logger.info(`総行数: ${lines.length}行`);

    let currentMedicine: Partial<MedicineInfo> | null = null;
    let currentSpecs: Array<{ spec: string; price: string }> = [];
    let collectingPrice = false;
    let lastSpec = '';

    // 医薬品パターンを含む行をデバッグ
    this.logger.info("\n=== 医薬品パターンの検証 ===");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      const cleanLine = this.textCleaner.clean(line);
      
      // パターン1: 番号と名前が同じ行にある場合
      const medicineMatch = cleanLine.match(/^(\d+)\s*(.*(錠|注|剤|液|ペン|キット|シリンジ|ワイプ).*)$/);
      if (medicineMatch && parseInt(medicineMatch[1]) <= 20 && !medicineMatch[2].match(/^mg\d+/)) { // 番号は20以下、規格ではない
        // 前の医薬品を保存
        if (currentMedicine && currentMedicine.name) {
          currentMedicine.specifications = currentSpecs;
          medicines.push(this.completeMedicine(currentMedicine));
        }
        
        // 新しい医薬品を開始
        currentMedicine = {
          no: medicineMatch[1],
          name: medicineMatch[2],
        };
        currentSpecs = [];
        collectingPrice = false;
        this.logger.info(`医薬品検出(同一行): ${currentMedicine.no}. ${currentMedicine.name}`);
        continue;
      }
      
      // パターン2: 番号が単独の行にある場合
      if (i < lines.length - 1) {
        const nextLine = lines[i + 1]?.trim() || '';
        const cleanNextLine = this.textCleaner.clean(nextLine);
        
        if (cleanLine.match(/^\d+$/) && parseInt(cleanLine) <= 20 && cleanNextLine.match(/^\d*.*(錠|注|剤|液|ペン|キット|シリンジ|ワイプ)/)) {
          // 番号が薬品名に含まれている場合を処理
          let medicineName = cleanNextLine;
          const leadingNumberMatch = cleanNextLine.match(/^(\d+)(.*)/);
          if (leadingNumberMatch) {
            medicineName = leadingNumberMatch[2].trim();
          }
          
          // 真の医薬品パターンか確認（フィルタリング）
          if (!medicineName.match(/注射薬|薬|外\d+|内\d+/) && medicineName.length > 5) {
            // 前の医薬品を保存
            if (currentMedicine && currentMedicine.name) {
              currentMedicine.specifications = currentSpecs;
              medicines.push(this.completeMedicine(currentMedicine));
            }
            
            // 新しい医薬品を開始
            currentMedicine = {
              no: cleanLine,
              name: medicineName,
            };
            currentSpecs = [];
            collectingPrice = false;
            this.logger.info(`医薬品検出(別行): ${currentMedicine.no}. ${currentMedicine.name}`);
            i++; // 次の行をスキップ
            continue;
          }
        }
      }

      if (currentMedicine) {
        // 「同」の処理
        if (line === '同' && i < lines.length - 1) {
          const nextLine = lines[i + 1].trim();
          if (nextLine.match(/錠|mg/)) {
            // 同じ薬品の別規格
            this.logger.info(`同じ薬品の別規格: ${nextLine}`);
            i++; // 次の行をスキップ
            continue;
          }
        }
        
        // 会社名
        if (cleanLine.includes('株式会社') || cleanLine.includes('(株)')) {
          // 既存のデータがある場合は結合
          if (currentMedicine.company) {
            currentMedicine.company += ' ' + cleanLine;
          } else {
            currentMedicine.company = cleanLine;
          }
        }
        // 成分名（会社名の後の行で、特定のキーワードを含まない）
        else if (i > 0 && 
                 (lines[i-1].includes('(株)') || lines[i-1].includes('株式会社')) &&
                 !cleanLine.includes('新有効成分') && 
                 !cleanLine.includes('mg') &&
                 cleanLine.length > 3) {
          currentMedicine.ingredient = cleanLine;
        }
        // 承認区分
        else if (cleanLine.includes('新有効成分含有医')) {
          currentMedicine.approvalCategory = '新有効成分含有医薬品';
        }
        // 規格
        else if (cleanLine.match(/^\d+mg\d+(錠|瓶|包|キット|筒)/) || 
                 cleanLine.match(/^\d+%[\d.]+g\d+包/)) {
          lastSpec = cleanLine;
          collectingPrice = true;
        }
        // 価格
        else if (collectingPrice && cleanLine.match(/^[\d,]+\.?\d*円/)) {
          currentSpecs.push({
            spec: lastSpec,
            price: cleanLine.split(' ')[0] // 最初の価格部分のみ
          });
          collectingPrice = false;
        }
        // 算定方式
        else if (cleanLine.includes('類似薬効比較方式') || cleanLine.includes('原価計算方式')) {
          currentMedicine.calculationMethod = cleanLine.split(' ')[0]; // 最初の部分のみ
        }
        // 加算情報
        else if ((cleanLine.includes('加算') || cleanLine.includes('新薬創出')) && 
                 !cleanLine.includes('算定')) {
          if (!currentMedicine.additionalBenefits) {
            currentMedicine.additionalBenefits = [];
          }
          currentMedicine.additionalBenefits.push(cleanLine);
        }
        // 薬効分類
        else if (cleanLine.match(/(内|注|外)\d+/)) {
          const match = cleanLine.match(/((内|注|外)\d+)\s*(.*)/);
          if (match) {
            currentMedicine.category = match[1];
            currentMedicine.categoryDetail = match[3] || '';
            
            // 次の行も詳細の一部かチェック
            if (i < lines.length - 1) {
              const nextLine = this.textCleaner.clean(lines[i + 1]);
              if (!nextLine.match(/^\d+\s+/) && 
                  !nextLine.includes('(株)') && 
                  nextLine.length > 5 &&
                  !nextLine.match(/^\d+$/)) {
                currentMedicine.categoryDetail += ' ' + nextLine;
                i++;
              }
            }
          }
        }
      }
    }

    // 最後の医薬品を保存
    if (currentMedicine && currentMedicine.name) {
      currentMedicine.specifications = currentSpecs;
      medicines.push(this.completeMedicine(currentMedicine));
    }

    return medicines;
  }

  private completeMedicine(partial: Partial<MedicineInfo>): MedicineInfo {
    return {
      no: partial.no || '',
      name: partial.name || '',
      company: this.textCleaner.clean(partial.company || ''),
      ingredient: this.textCleaner.clean(partial.ingredient || ''),
      approvalCategory: partial.approvalCategory || '',
      specifications: partial.specifications || [],
      calculationMethod: partial.calculationMethod || '',
      additionalBenefits: (partial.additionalBenefits || []).map(b => this.textCleaner.clean(b)),
      category: partial.category || '',
      categoryDetail: this.textCleaner.clean(partial.categoryDetail || ''),
    };
  }
}

// ===== マークダウンフォーマッタ =====

class MarkdownFormatter {
  // 詳細形式のマークダウンを生成
  formatDetailed(medicines: MedicineInfo[], title: string): string {
    const lines: string[] = [];
    
    lines.push(`# ${title}`);
    lines.push('');
    
    medicines.forEach((med, index) => {
      lines.push(`## ${med.no}. ${med.name}`);
      lines.push('');
      
      // 基本情報
      lines.push('### 基本情報');
      if (med.company) {
        lines.push(`- **会社名**: ${med.company}`);
      }
      if (med.ingredient) {
        lines.push(`- **成分名**: ${med.ingredient}`);
      }
      if (med.approvalCategory) {
        lines.push(`- **承認区分**: ${med.approvalCategory}`);
      }
      lines.push('');
      
      // 規格・価格
      if (med.specifications.length > 0) {
        lines.push('### 規格・価格');
        lines.push('| 規格 | 算定薬価 |');
        lines.push('|------|----------|');
        med.specifications.forEach(spec => {
          lines.push(`| ${spec.spec} | ${spec.price || '-'} |`);
        });
        lines.push('');
      }
      
      // 薬効分類
      if (med.category || med.categoryDetail) {
        lines.push('### 薬効分類');
        lines.push(`${med.category} ${med.categoryDetail}`.trim());
        lines.push('');
      }
      
      // 算定
      if (med.calculationMethod || med.additionalBenefits.length > 0) {
        lines.push('### 算定');
        if (med.calculationMethod) {
          lines.push('#### 算定方式');
          lines.push(med.calculationMethod);
          lines.push('');
        }
        if (med.additionalBenefits.length > 0) {
          lines.push('#### 加算等');
          med.additionalBenefits.forEach(benefit => {
            lines.push(`- ${benefit}`);
          });
          lines.push('');
        }
      }
      
      if (index < medicines.length - 1) {
        lines.push('---');
        lines.push('');
      }
    });
    
    return lines.join('\n');
  }

  // 概要テーブル形式のマークダウンを生成
  formatSummary(medicines: MedicineInfo[], title: string): string {
    const lines: string[] = [];
    
    lines.push(`# ${title}`);
    lines.push('');
    lines.push('## 医薬品一覧');
    lines.push('');
    lines.push('| No. | 銘柄名 | 会社名 | 成分名 | 薬効分類 | 算定方式 |');
    lines.push('|-----|--------|--------|--------|----------|----------|');
    
    medicines.forEach(med => {
      const category = med.category && med.categoryDetail 
        ? `${med.category} ${med.categoryDetail.substring(0, 20)}...` 
        : med.category || med.categoryDetail || '';
      
      const row = [
        med.no,
        med.name,
        med.company || '-',
        med.ingredient || '-',
        category,
        med.calculationMethod || '-'
      ].map(cell => cell.replace(/\|/g, '\\|'));
      
      lines.push('| ' + row.join(' | ') + ' |');
    });
    
    // 集計
    lines.push('');
    lines.push('## 集計');
    lines.push('');
    lines.push('| 区分 | 品目数 |');
    lines.push('|------|--------|');
    
    const internalCount = medicines.filter(m => m.category.startsWith('内')).length;
    const injectionCount = medicines.filter(m => m.category.startsWith('注')).length;
    const externalCount = medicines.filter(m => m.category.startsWith('外')).length;
    
    if (internalCount > 0) lines.push(`| 内用薬 | ${internalCount} |`);
    if (injectionCount > 0) lines.push(`| 注射薬 | ${injectionCount} |`);
    if (externalCount > 0) lines.push(`| 外用薬 | ${externalCount} |`);
    lines.push(`| **計** | **${medicines.length}** |`);
    
    return lines.join('\n');
  }
}

// ===== メインプロセッサ =====

class PdfMedicineExtractor {
  private logger: ConsoleLogger;
  private visionClient: VisionApiClient;
  private extractor: MedicineExtractor;
  private formatter: MarkdownFormatter;

  constructor(verbose: boolean = false) {
    this.logger = new ConsoleLogger(verbose);
    this.visionClient = new VisionApiClient(verbose);
    this.extractor = new MedicineExtractor(verbose);
    this.formatter = new MarkdownFormatter();
  }

  async process(pdfPath: string, options: ProcessingOptions = {}): Promise<void> {
    try {
      // ファイルの存在確認
      const fileInfo = await Deno.stat(pdfPath);
      if (!fileInfo.isFile) {
        throw new Error(`${pdfPath} はファイルではありません`);
      }
      
      this.logger.info(`PDFファイル: ${pdfPath}`);
      this.logger.info(`ファイルサイズ: ${(fileInfo.size / 1024 / 1024).toFixed(2)} MB`);
      
      // 出力ディレクトリの設定
      const outputDir = options.outputDir || dirname(pdfPath);
      const baseName = basename(pdfPath, '.pdf');
      
      // Vision APIで分析
      this.logger.info('\nVision APIで分析中...');
      await this.visionClient.authenticate();
      const visionResponse = await this.visionClient.analyzeDocument(pdfPath);
      
      // 構造JSONを保存（オプション）
      if (options.keepStructureJson) {
        const jsonPath = join(outputDir, `${baseName}_structure.json`);
        await Deno.writeTextFile(jsonPath, JSON.stringify(visionResponse, null, 2));
        this.logger.info(`構造データを保存: ${jsonPath}`);
      }
      
      // 医薬品情報を抽出
      this.logger.info('\n医薬品情報を抽出中...');
      const medicines = this.extractor.extractFromVisionResponse(visionResponse);
      this.logger.info(`抽出された医薬品数: ${medicines.length}`);
      
      // タイトルの抽出
      let title = '新医薬品一覧表';
      let fullText = '';
      
      // レスポンスからテキストを取得（入れ子構造に対応）
      if (visionResponse.responses?.[0]?.responses?.[0]?.fullTextAnnotation?.text) {
        fullText = visionResponse.responses[0].responses[0].fullTextAnnotation.text;
      } else if (visionResponse.responses?.[0]?.fullTextAnnotation?.text) {
        fullText = visionResponse.responses[0].fullTextAnnotation.text;
      }
      
      if (fullText) {
        const firstLine = fullText.split('\n')[0];
        if (firstLine.includes('新医薬品')) {
          title = firstLine.replace(/\s+/g, ' ').trim();
        }
      }
      
      // マークダウンファイルを生成
      this.logger.info('\nマークダウンファイルを生成中...');
      
      // 詳細形式
      const detailedPath = join(outputDir, `${baseName}_detailed.md`);
      const detailedContent = this.formatter.formatDetailed(medicines, title);
      await Deno.writeTextFile(detailedPath, detailedContent);
      this.logger.info(`詳細形式を保存: ${detailedPath}`);
      
      // 概要形式
      const summaryPath = join(outputDir, `${baseName}_summary.md`);
      const summaryContent = this.formatter.formatSummary(medicines, title);
      await Deno.writeTextFile(summaryPath, summaryContent);
      this.logger.info(`概要形式を保存: ${summaryPath}`);
      
      // 結果のプレビュー
      this.logger.info('\n=== 抽出結果 ===');
      medicines.slice(0, 3).forEach(med => {
        this.logger.info(`${med.no}. ${med.name}`);
        if (med.company) this.logger.info(`   会社: ${med.company}`);
        if (med.ingredient) this.logger.info(`   成分: ${med.ingredient}`);
        if (med.specifications.length > 0) {
          this.logger.info(`   価格: ${med.specifications[0].price || '-'}`);
        }
      });
      if (medicines.length > 3) {
        this.logger.info('   ...');
      }
      
    } catch (error) {
      this.logger.error('エラーが発生しました', error as Error);
      
      if ((error as Error).message.includes('GOOGLE_APPLICATION_CREDENTIALS')) {
        this.logger.error('\n環境変数の設定が必要です:');
        this.logger.error('export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"');
      }
      
      throw error;
    }
  }
}

// ===== メイン処理 =====

async function main() {
  const args = Deno.args;
  
  // ヘルプ表示
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
PDF医薬品データ抽出ツール

使用方法:
  deno run --allow-all pdf-medicine-extractor.ts <PDFファイル> [オプション]

オプション:
  --output-dir <ディレクトリ>  出力先ディレクトリを指定
  --keep-json                構造データJSONファイルも保存
  --verbose                  詳細ログを表示
  --help, -h                 このヘルプを表示

例:
  deno run --allow-all pdf-medicine-extractor.ts medicine.pdf
  deno run --allow-all pdf-medicine-extractor.ts medicine.pdf --output-dir ./output --keep-json

必要な環境変数:
  GOOGLE_APPLICATION_CREDENTIALS  Google Cloud サービスアカウントキーのパス

出力ファイル:
  {basename}_detailed.md   詳細形式のマークダウン
  {basename}_summary.md    概要テーブル形式のマークダウン
  {basename}_structure.json 構造データ（--keep-json指定時のみ）
`);
    Deno.exit(0);
  }
  
  // 引数の解析
  const pdfPath = args[0];
  const options: ProcessingOptions = {
    outputDir: undefined,
    keepStructureJson: args.includes('--keep-json'),
    verbose: args.includes('--verbose'),
  };
  
  // 出力ディレクトリの取得
  const outputDirIndex = args.indexOf('--output-dir');
  if (outputDirIndex !== -1 && outputDirIndex < args.length - 1) {
    options.outputDir = args[outputDirIndex + 1];
  }
  
  // 処理実行
  const extractor = new PdfMedicineExtractor(options.verbose);
  
  try {
    await extractor.process(pdfPath, options);
    console.log('\n処理が完了しました。');
  } catch (error) {
    console.error('\n処理中にエラーが発生しました。');
    Deno.exit(1);
  }
}

// エントリーポイント
if (import.meta.main) {
  await main();
}
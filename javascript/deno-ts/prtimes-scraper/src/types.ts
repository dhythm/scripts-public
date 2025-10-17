/**
 * PRTIMESのリリース情報の型定義
 */
export type PrTimesRelease = {
  /** リリースのタイトル */
  title: string;
  /** 企業名 */
  company: string;
  /** リリースのURL */
  url: string;
  /** 公開日時 */
  date: string;
  /** 企業のウェブサイトURL（オプション） */
  companyUrl?: string;
  /** 企業ID */
  companyId?: string;
};

/**
 * 企業情報の型定義
 */
export type CompanyInfo = {
  /** 企業のウェブサイトURL */
  url?: string;
  /** 業種 */
  industry?: string;
  /** 本社所在地 */
  address?: string;
  /** 資本金 */
  capital?: string;
};

/**
 * スクレイピングのオプション
 */
export type ScraperOptions = {
  /** 出力ファイルパス（デフォルト: prtimes_releases.json） */
  outputPath?: string;
  /** 詳細ログを表示するか */
  verbose?: boolean;
  /** 企業情報も取得するか */
  fetchCompanyInfo?: boolean;
};

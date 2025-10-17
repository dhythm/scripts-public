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
};

/**
 * スクレイピングのオプション
 */
export type ScraperOptions = {
  /** 出力ファイルパス（デフォルト: prtimes_releases.json） */
  outputPath?: string;
  /** 詳細ログを表示するか */
  verbose?: boolean;
};

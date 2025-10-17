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
  /** 相対的な時間表記（例: "5分前"） */
  relativeTime?: string;
  /** 企業のウェブサイトURL（オプション） */
  companyUrl?: string;
  /** 企業ID */
  companyId?: string;
  /** リリースID（例: 000000001.000000001） */
  releaseId?: string;
  /** サムネイル画像URL */
  thumbnailUrl?: string;
  /** サムネイル画像の代替テキスト */
  thumbnailAlt?: string;
  /** 記事に付随するタグ */
  tags?: string[];
  /** 企業詳細情報 */
  companyInfo?: CompanyInfo;
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
  /** 本社所在地（建物名などの補足） */
  address2?: string;
  /** 資本金 */
  capital?: string;
  /** 電話番号 */
  phone?: string;
  /** 設立年月 */
  foundationDate?: string;
  /** 代表者名 */
  representative?: string;
  /** 企業概要 */
  description?: string;
  /** X（旧Twitter）アカウントURL */
  xUrl?: string;
  /** FacebookページURL */
  facebookUrl?: string;
  /** YouTubeチャンネルURL */
  youtubeUrl?: string;
  /** InstagramアカウントURL */
  instagramUrl?: string;
  /** LinkedInページURL */
  linkedinUrl?: string;
  /** TikTokアカウントURL */
  tiktokUrl?: string;
  /** SNS等におけるフォロワー数 */
  followerCount?: number;
  /** ロゴ画像URL */
  logoImageUrl?: string;
  /** カバー画像URL */
  coverImageUrl?: string;
  /** OGP画像URL */
  ogImageUrl?: string;
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

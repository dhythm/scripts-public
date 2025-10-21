export interface CalendarEvent {
  id: string;
  summary: string;
  start: Date;
  end: Date;
  attendees?: string[];
}

export interface TimeSlot {
  start: Date;
  end: Date;
}

export interface Person {
  name: string;
  email: string;
  source: "google" | "hubspot";
  sourceId: string; // Google: calendarId, HubSpot: userId
}

export interface AvailabilityResult {
  person: Person;
  busySlots: TimeSlot[];
  availableSlots?: TimeSlot[];
}

export interface MeetingCandidate {
  slot: TimeSlot;
  score: number;
  reasons: string[];
}

export interface CliOptions {
  startDate: Date;
  endDate: Date;
  duration: number; // 分単位
  participants: Person[];
  businessHoursOnly: boolean;
  timezone: string;
  outputFormat: "json" | "markdown" | "text";
  useOpenAI: boolean;
  verbose: boolean;
  showAll: boolean; // 全ての候補を表示
  limit?: number; // 表示する候補数の上限
  rawSlots: boolean; // 連続した空き時間ブロックを表示
  minDuration?: number; // 最小会議時間（分）
  textNoIndex?: boolean; // テキスト出力で番号を表示しない
  skipHolidays: boolean; // 日本の祝日を除外
}

export interface GoogleCalendarConfig {
  clientId: string;
  clientSecret: string;
  refreshToken: string;
}

export interface HubSpotConfig {
  apiKey: string;
}

export interface OpenAIConfig {
  apiKey: string;
  model?: string;
}

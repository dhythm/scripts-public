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
  calendarId?: string;
  hubspotUserId?: string;
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
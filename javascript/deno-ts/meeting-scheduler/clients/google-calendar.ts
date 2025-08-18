import { CalendarEvent, TimeSlot, Person } from "../types/index.ts";
import { GoogleAuth } from "../utils/auth-google.ts";

export class GoogleCalendarClient {
  private auth: GoogleAuth;
  private baseUrl = "https://www.googleapis.com/calendar/v3";

  constructor(auth: GoogleAuth) {
    this.auth = auth;
  }

  async getEvents(
    calendarId: string,
    timeMin: Date,
    timeMax: Date
  ): Promise<CalendarEvent[]> {
    const accessToken = await this.auth.getAccessToken();
    
    const params = new URLSearchParams({
      timeMin: timeMin.toISOString(),
      timeMax: timeMax.toISOString(),
      singleEvents: "true",
      orderBy: "startTime",
    });

    const response = await fetch(
      `${this.baseUrl}/calendars/${encodeURIComponent(calendarId)}/events?${params}`,
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Google Calendar API エラー: ${error}`);
    }

    const data = await response.json();
    
    return (data.items || []).map((item: any) => ({
      id: item.id,
      summary: item.summary || "（タイトルなし）",
      start: new Date(item.start?.dateTime || item.start?.date),
      end: new Date(item.end?.dateTime || item.end?.date),
      attendees: item.attendees?.map((a: any) => a.email) || [],
    }));
  }

  async getFreeBusy(
    calendarIds: string[],
    timeMin: Date,
    timeMax: Date
  ): Promise<Map<string, TimeSlot[]>> {
    const accessToken = await this.auth.getAccessToken();
    
    const requestBody = {
      timeMin: timeMin.toISOString(),
      timeMax: timeMax.toISOString(),
      items: calendarIds.map(id => ({ id })),
    };

    if (Deno.env.get("DEBUG") === "true") {
      console.log("\n📅 Google Calendar FreeBusy API リクエスト:");
      console.log(`  期間: ${timeMin.toISOString()} - ${timeMax.toISOString()}`);
      console.log(`  カレンダー: ${calendarIds.join(", ")}`);
    }

    const response = await fetch(
      `${this.baseUrl}/freeBusy`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      }
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Google Calendar FreeBusy API エラー: ${error}`);
    }

    const data = await response.json();
    
    if (Deno.env.get("DEBUG") === "true") {
      console.log("\n📅 Google Calendar FreeBusy API レスポンス:");
      console.log(JSON.stringify(data, null, 2));
    }
    
    const result = new Map<string, TimeSlot[]>();

    for (const [calendarId, calendar] of Object.entries(data.calendars || {})) {
      const busySlots = ((calendar as any).busy || []).map((busy: any) => ({
        start: new Date(busy.start),
        end: new Date(busy.end),
      }));
      result.set(calendarId, busySlots);
      
      if (Deno.env.get("DEBUG") === "true") {
        console.log(`\n  ${calendarId}:`);
        if (busySlots.length === 0) {
          console.log("    予定なし");
        } else {
          busySlots.forEach((slot: TimeSlot) => {
            console.log(`    - ${slot.start.toISOString()} ~ ${slot.end.toISOString()}`);
          });
        }
      }
    }

    return result;
  }

  async getMultipleCalendarsBusy(
    people: Person[],
    timeMin: Date,
    timeMax: Date
  ): Promise<Map<string, TimeSlot[]>> {
    const googlePeople = people.filter(p => p.source === "google");
    const calendarIds = googlePeople.map(p => p.sourceId);
    
    if (calendarIds.length === 0) {
      return new Map();
    }

    return this.getFreeBusy(calendarIds, timeMin, timeMax);
  }
}
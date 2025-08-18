import { CalendarEvent, TimeSlot, Person } from "../types/index.ts";
import { HubSpotAuth } from "../utils/auth-hubspot.ts";

export class HubSpotClient {
  private auth: HubSpotAuth;
  private baseUrl = "https://api.hubapi.com";

  constructor(auth: HubSpotAuth) {
    this.auth = auth;
  }

  async getMeetings(
    userId: string,
    startDate: Date,
    endDate: Date
  ): Promise<CalendarEvent[]> {
    const headers = this.auth.getAuthHeader();
    
    // HubSpot Meetings API v3
    const response = await fetch(
      `${this.baseUrl}/crm/v3/objects/meetings`,
      {
        headers,
      }
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`HubSpot API エラー: ${error}`);
    }

    const data = await response.json();
    const meetings: CalendarEvent[] = [];

    // HubSpotのミーティングオブジェクトをフィルタリング
    for (const meeting of data.results || []) {
      const startTime = meeting.properties?.hs_meeting_start_time;
      const endTime = meeting.properties?.hs_meeting_end_time;
      
      if (startTime && endTime) {
        const meetingStart = new Date(startTime);
        const meetingEnd = new Date(endTime);
        
        // 指定期間内のミーティングのみ
        if (meetingStart >= startDate && meetingEnd <= endDate) {
          meetings.push({
            id: meeting.id,
            summary: meeting.properties?.hs_meeting_title || "HubSpotミーティング",
            start: meetingStart,
            end: meetingEnd,
            attendees: this.extractAttendees(meeting),
          });
        }
      }
    }

    return meetings;
  }

  private extractAttendees(meeting: any): string[] {
    const attendees: string[] = [];
    
    // 参加者のメールアドレスを抽出
    if (meeting.properties?.hs_meeting_external_url) {
      // URLから参加者情報を推測（実際のAPIレスポンスに応じて調整）
      const externalAttendees = meeting.properties?.hs_attendee_owner_ids;
      if (externalAttendees) {
        attendees.push(...externalAttendees.split(";"));
      }
    }

    return attendees;
  }

  async getUserAvailability(
    userId: string,
    startDate: Date,
    endDate: Date
  ): Promise<TimeSlot[]> {
    // HubSpot Meetings Availability API を使用
    const headers = this.auth.getAuthHeader();
    
    // ユーザーのカレンダー設定を取得
    const response = await fetch(
      `${this.baseUrl}/meetings/v1/settings/users/${userId}`,
      {
        headers,
      }
    );

    if (response.ok) {
      const data = await response.json();
      
      // ユーザーの利用可能時間から予約済み時間を取得
      const meetings = await this.getMeetings(userId, startDate, endDate);
      
      return meetings.map(meeting => ({
        start: meeting.start,
        end: meeting.end,
      }));
    }

    // エラーの場合は空配列を返す
    console.warn(`HubSpotユーザー ${userId} の予定取得に失敗しました`);
    return [];
  }

  async getSchedulerAvailability(
    meetingLink: string,
    timezone: string = "Asia/Tokyo"
  ): Promise<TimeSlot[]> {
    // HubSpot Scheduler API v3 を使用
    const headers = this.auth.getAuthHeader();
    
    if (Deno.env.get("DEBUG") === "true") {
      console.log("\n📅 HubSpot Scheduler API リクエスト:");
      console.log(`  Meeting Link: ${meetingLink}`);
      console.log(`  Timezone: ${timezone}`);
    }
    
    const response = await fetch(
      `${this.baseUrl}/scheduler/v3/meetings/meeting-links/book/availability-page/${meetingLink}?timezone=${timezone}`,
      {
        headers,
      }
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`HubSpot Scheduler API エラー: ${error}`);
    }

    const data = await response.json();
    
    if (Deno.env.get("DEBUG") === "true") {
      console.log("\n📅 HubSpot Scheduler API レスポンス:");
      console.log(`  busyTimes数: ${data.allUsersBusyTimes?.[0]?.busyTimes?.length || 0}`);
    }
    
    const busySlots: TimeSlot[] = [];
    
    // allUsersBusyTimesから予約済み時間を抽出
    for (const userBusy of data.allUsersBusyTimes || []) {
      if (Deno.env.get("DEBUG") === "true") {
        console.log(`\n  ユーザー: ${userBusy.meetingsUser?.userProfile?.fullName || "不明"}`);
        console.log(`  Email: ${userBusy.meetingsUser?.userProfile?.email || "不明"}`);
      }
      
      for (const busy of userBusy.busyTimes || []) {
        const slot = {
          start: new Date(busy.start),
          end: new Date(busy.end),
        };
        busySlots.push(slot);
        
        if (Deno.env.get("DEBUG") === "true") {
          console.log(`    - ${slot.start.toISOString()} ~ ${slot.end.toISOString()}`);
        }
      }
    }
    
    return busySlots;
  }

  async getMultipleUsersBusy(
    people: Person[],
    startDate: Date,
    endDate: Date
  ): Promise<Map<string, TimeSlot[]>> {
    const result = new Map<string, TimeSlot[]>();
    const hubspotPeople = people.filter(p => p.source === "hubspot");

    for (const person of hubspotPeople) {
      try {
        // Scheduler APIを使用（meetingLinkとして扱う）
        const busySlots = await this.getSchedulerAvailability(
          person.sourceId,
          Deno.env.get("DEFAULT_TIMEZONE") || "Asia/Tokyo"
        );
        
        // 期間でフィルタリング
        const filteredSlots = busySlots.filter(
          slot => slot.start >= startDate && slot.end <= endDate
        );
        
        result.set(person.sourceId, filteredSlots);
        
        if (Deno.env.get("DEBUG") === "true") {
          console.log(`\n${person.name}: ${filteredSlots.length}件の予定（フィルタ後）`);
        }
      } catch (error) {
        console.error(`${person.name}のHubSpot予定取得エラー:`, error);
        result.set(person.sourceId, []);
      }
    }

    return result;
  }
}
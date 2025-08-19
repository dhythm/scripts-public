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
    timezone: string = "Asia/Tokyo",
    useAvailabilities: boolean = false
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
      
      // availabilitiesの情報も表示
      const availCount = data.linkAvailability?.linkAvailabilityByDuration?.["3600000"]?.availabilities?.length || 0;
      console.log(`  availabilities数 (60分枠): ${availCount}`);
      console.log("  注意: APIからはUTC時刻で返されます");
    }
    
    // availabilitiesを使用する場合（Webページと同じ空き時間）
    if (useAvailabilities && data.linkAvailability?.linkAvailabilityByDuration) {
      const availableSlots: TimeSlot[] = [];
      const duration60 = data.linkAvailability.linkAvailabilityByDuration["3600000"];
      
      if (duration60?.availabilities) {
        if (Deno.env.get("DEBUG") === "true") {
          console.log("\n  📆 Availabilities (空き時間) を使用:");
        }
        
        for (const avail of duration60.availabilities) {
          const slot = {
            start: new Date(avail.startMillisUtc),
            end: new Date(avail.endMillisUtc),
          };
          
          if (Deno.env.get("DEBUG") === "true") {
            const startJST = slot.start.toLocaleString("ja-JP", { timeZone: "Asia/Tokyo", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
            const endJST = slot.end.toLocaleTimeString("ja-JP", { timeZone: "Asia/Tokyo", hour: "2-digit", minute: "2-digit" });
            console.log(`    空き: ${startJST} - ${endJST}`);
          }
          
          availableSlots.push(slot);
        }
      }
      
      // availabilitiesをbusySlotsに変換（逆転）
      // これは他のロジックとの互換性のため
      return this.convertAvailabilitiesToBusySlots(availableSlots, new Date("2025-08-25T00:00:00+09:00"), new Date("2025-08-29T23:59:59+09:00"));
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
          const startJST = slot.start.toLocaleString("ja-JP", { timeZone: "Asia/Tokyo", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
          const endJST = slot.end.toLocaleString("ja-JP", { timeZone: "Asia/Tokyo", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
          console.log(`    - UTC: ${slot.start.toISOString()} ~ ${slot.end.toISOString()}`);
          console.log(`      JST: ${startJST} ~ ${endJST}`);
        }
      }
    }
    
    return busySlots;
  }
  
  private convertAvailabilitiesToBusySlots(
    availableSlots: TimeSlot[],
    startDate: Date,
    endDate: Date
  ): TimeSlot[] {
    // 空き時間からbusySlotsを逆算
    // これにより、他のロジックと互換性を保つ
    const busySlots: TimeSlot[] = [];
    let currentTime = new Date(startDate);
    
    // 空き時間を時間順にソート
    const sortedAvailable = [...availableSlots].sort((a, b) => a.start.getTime() - b.start.getTime());
    
    for (const avail of sortedAvailable) {
      // 現在時刻から空き時間の開始までがbusy
      if (currentTime < avail.start) {
        busySlots.push({
          start: new Date(currentTime),
          end: new Date(avail.start)
        });
      }
      currentTime = new Date(avail.end);
    }
    
    // 最後の空き時間から終了時刻までがbusy
    if (currentTime < endDate) {
      busySlots.push({
        start: new Date(currentTime),
        end: new Date(endDate)
      });
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
        // HUBSPOT_USE_AVAILABILITIES環境変数で切り替え
        const useAvailabilities = Deno.env.get("HUBSPOT_USE_AVAILABILITIES") === "true";
        const busySlots = await this.getSchedulerAvailability(
          person.sourceId,
          Deno.env.get("DEFAULT_TIMEZONE") || "Asia/Tokyo",
          useAvailabilities
        );
        
        // 期間でフィルタリング（UTCで比較）
        const filteredSlots = busySlots.filter(slot => {
          // startDateとendDateはJSTで指定されているが、DateオブジェクトはUTCで比較可能
          return slot.start <= endDate && slot.end >= startDate;
        });
        
        result.set(person.sourceId, filteredSlots);
        
        if (Deno.env.get("DEBUG") === "true") {
          console.log(`\n${person.name}: ${busySlots.length}件の予定を取得`);
          console.log(`  期間内: ${filteredSlots.length}件の予定`);
        }
      } catch (error) {
        console.error(`${person.name}のHubSpot予定取得エラー:`, error);
        result.set(person.sourceId, []);
      }
    }

    return result;
  }
}
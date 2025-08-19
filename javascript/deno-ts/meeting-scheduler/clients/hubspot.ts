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
    useAvailabilities: boolean = false,
    startDate?: Date,
    endDate?: Date
  ): Promise<TimeSlot[]> {
    // HubSpot Scheduler API v3 を使用
    const headers = this.auth.getAuthHeader();
    
    // 月のオフセットを計算
    const monthOffsets: number[] = [];
    if (startDate && endDate) {
      const currentDate = new Date();
      const currentMonth = currentDate.getMonth();
      const currentYear = currentDate.getFullYear();
      
      const startMonth = startDate.getMonth();
      const startYear = startDate.getFullYear();
      const endMonth = endDate.getMonth();
      const endYear = endDate.getFullYear();
      
      // 開始月から終了月までの各月のオフセットを計算
      for (let year = startYear; year <= endYear; year++) {
        const monthStart = (year === startYear) ? startMonth : 0;
        const monthEnd = (year === endYear) ? endMonth : 11;
        
        for (let month = monthStart; month <= monthEnd; month++) {
          const offset = (year - currentYear) * 12 + (month - currentMonth);
          if (!monthOffsets.includes(offset)) {
            monthOffsets.push(offset);
          }
        }
      }
    } else {
      // 日付範囲が指定されていない場合は現在の月のみ
      monthOffsets.push(0);
    }
    
    if (Deno.env.get("DEBUG") === "true") {
      console.log("\n📅 HubSpot Scheduler API リクエスト:");
      console.log(`  Meeting Link: ${meetingLink}`);
      console.log(`  Timezone: ${timezone}`);
      if (startDate && endDate) {
        console.log(`  期間: ${startDate.toISOString().split('T')[0]} - ${endDate.toISOString().split('T')[0]}`);
        console.log(`  月オフセット: ${monthOffsets.join(', ')}`);
      }
    }
    
    // 各月のデータを取得
    const allAvailabilities: any[] = [];
    const allBusyTimes: any[] = [];
    
    for (const monthOffset of monthOffsets) {
      const url = `${this.baseUrl}/scheduler/v3/meetings/meeting-links/book/availability-page/${meetingLink}?timezone=${timezone}&monthOffset=${monthOffset}`;
      
      if (Deno.env.get("DEBUG") === "true") {
        console.log(`  月オフセット ${monthOffset} のデータを取得中...`);
      }
      
      const response = await fetch(url, { headers });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`HubSpot Scheduler API エラー: ${error}`);
      }

      const data = await response.json();
      
      // このmonthOffsetのavailabilitiesを収集
      if (data.linkAvailability?.linkAvailabilityByDuration?.["3600000"]?.availabilities) {
        allAvailabilities.push(...data.linkAvailability.linkAvailabilityByDuration["3600000"].availabilities);
      }
      
      // busyTimesも収集
      if (data.allUsersBusyTimes) {
        for (const userBusy of data.allUsersBusyTimes) {
          if (userBusy.busyTimes) {
            allBusyTimes.push(...userBusy.busyTimes);
          }
        }
      }
    }
    
    // 収集したデータを処理
    const data = {
      linkAvailability: {
        linkAvailabilityByDuration: {
          "3600000": {
            availabilities: allAvailabilities
          }
        }
      },
      allUsersBusyTimes: allBusyTimes.length > 0 ? [{
        busyTimes: allBusyTimes
      }] : []
    };
    
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
      const duration60 = data.linkAvailability.linkAvailabilityByDuration["3600000"];
      
      if (duration60?.availabilities) {
        if (Deno.env.get("DEBUG") === "true") {
          console.log("\n  📆 Availabilities (空き時間) を使用:");
          console.log(`  注意: availabilitiesに含まれる時間のみが予約可能です`);
        }
        
        // availabilitiesは空き時間なので、そのまま返す
        // これをbusySlotsとして扱うのは間違い
        // 代わりに、availableSlots というフラグを立てる
        const availableSlots: TimeSlot[] = [];
        
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
        
        // availableSlots を特別なマーカーとして返す
        // busySlotsではなくavailableSlots として扱うため、マーカーを付ける
        return availableSlots;
      }
      
      // availabilitiesがない場合は、全て予約不可
      if (Deno.env.get("DEBUG") === "true") {
        console.log("  ⚠️ availabilitiesがありません - 全時間帯が予約不可");
      }
      return [];
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
        // デフォルトでavailabilitiesを使用（Webインターフェースと同じ結果）
        // HUBSPOT_USE_BUSYTIMES=trueで旧動作に切り替え可能
        const useAvailabilities = Deno.env.get("HUBSPOT_USE_BUSYTIMES") !== "true";
        const slots = await this.getSchedulerAvailability(
          person.sourceId,
          Deno.env.get("DEFAULT_TIMEZONE") || "Asia/Tokyo",
          useAvailabilities,
          startDate,
          endDate
        );
        
        if (useAvailabilities) {
          // availabilitiesモードの場合、slotsは空き時間
          // これを特別に扱うために、availableSlotsマーカーを付ける
          // ただし、現在のアーキテクチャではbusySlotsとして扱う必要があるため
          // 全期間から空き時間を引いたものをbusySlotsとして返す
          const availableSlots = slots.filter(slot => {
            return slot.start <= endDate && slot.end >= startDate;
          });
          
          // 空き時間から逆にbusySlotsを計算
          const busySlots: TimeSlot[] = [];
          let currentTime = new Date(startDate);
          
          // 空き時間をソート
          const sortedAvailable = [...availableSlots].sort((a, b) => a.start.getTime() - b.start.getTime());
          
          for (const avail of sortedAvailable) {
            if (currentTime < avail.start) {
              busySlots.push({
                start: new Date(currentTime),
                end: new Date(avail.start)
              });
            }
            currentTime = new Date(Math.max(currentTime.getTime(), avail.end.getTime()));
          }
          
          // 最後の空き時間から終了時刻まで
          if (currentTime < endDate) {
            busySlots.push({
              start: new Date(currentTime),
              end: new Date(endDate)
            });
          }
          
          result.set(person.sourceId, busySlots);
          
          if (Deno.env.get("DEBUG") === "true") {
            console.log(`\n${person.name}: ${availableSlots.length}件の空き時間から変換`);
            console.log(`  busySlots: ${busySlots.length}件`);
            
            // busySlotsの詳細を表示
            for (const busy of busySlots) {
              const startStr = busy.start.toLocaleString("ja-JP", { timeZone: "Asia/Tokyo", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
              const endStr = busy.end.toLocaleString("ja-JP", { timeZone: "Asia/Tokyo", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
              console.log(`    Busy: ${startStr} - ${endStr}`);
            }
          }
        } else {
          // 通常のbusyTimesモード
          const filteredSlots = slots.filter(slot => {
            return slot.start <= endDate && slot.end >= startDate;
          });
          
          result.set(person.sourceId, filteredSlots);
          
          if (Deno.env.get("DEBUG") === "true") {
            console.log(`\n${person.name}: ${slots.length}件の予定を取得`);
            console.log(`  期間内: ${filteredSlots.length}件の予定`);
          }
        }
      } catch (error) {
        console.error(`${person.name}のHubSpot予定取得エラー:`, error);
        result.set(person.sourceId, []);
      }
    }

    return result;
  }
}
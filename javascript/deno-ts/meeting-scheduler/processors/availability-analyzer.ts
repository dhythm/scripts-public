import OpenAI from "npm:openai@4.28.0";
import { 
  TimeSlot, 
  Person, 
  AvailabilityResult, 
  MeetingCandidate,
  OpenAIConfig 
} from "../types/index.ts";
import { 
  findCommonAvailableSlots, 
  formatDateTime,
  mergeTimeSlots 
} from "../utils/date-utils.ts";

export class AvailabilityAnalyzer {
  private openai?: OpenAI;

  constructor(openaiConfig?: OpenAIConfig) {
    if (openaiConfig?.apiKey) {
      this.openai = new OpenAI({ apiKey: openaiConfig.apiKey });
    }
  }

  analyzeAvailability(
    availabilityResults: AvailabilityResult[],
    startDate: Date,
    endDate: Date,
    durationMinutes: number,
    businessHoursOnly: boolean
  ): TimeSlot[] {
    return findCommonAvailableSlots(
      availabilityResults,
      startDate,
      endDate,
      durationMinutes,
      businessHoursOnly
    );
  }

  async optimizeWithOpenAI(
    availableSlots: TimeSlot[],
    participants: Person[],
    preferences?: string
  ): Promise<MeetingCandidate[]> {
    if (!this.openai || availableSlots.length === 0) {
      // OpenAIが使えない場合は単純なスコアリング
      return this.simpleScoring(availableSlots);
    }

    const prompt = this.createOptimizationPrompt(
      availableSlots,
      participants,
      preferences
    );

    try {
      const response = await this.openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content: "あなたは会議スケジュール最適化の専門家です。参加者全員にとって最適な会議時間を提案してください。"
          },
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: 0.3,
        response_format: { type: "json_object" }
      });

      const content = response.choices[0]?.message?.content;
      if (!content) {
        return this.simpleScoring(availableSlots);
      }

      const result = JSON.parse(content);
      return this.parseOpenAIResponse(result, availableSlots);

    } catch (error) {
      console.error("OpenAI API エラー:", error);
      return this.simpleScoring(availableSlots);
    }
  }

  private createOptimizationPrompt(
    slots: TimeSlot[],
    participants: Person[],
    preferences?: string
  ): string {
    const slotsInfo = slots.slice(0, 20).map((slot, index) => ({
      index,
      start: formatDateTime(slot.start),
      end: formatDateTime(slot.end),
      dayOfWeek: ["日", "月", "火", "水", "木", "金", "土"][slot.start.getDay()],
      hour: slot.start.getHours()
    }));

    return `
以下の空き時間から、${participants.length}人の参加者にとって最適な会議時間を5つ選んでください。

参加者:
${participants.map(p => `- ${p.name} (${p.email})`).join("\n")}

利用可能な時間帯:
${JSON.stringify(slotsInfo, null, 2)}

評価基準:
1. 平日の業務時間（9-18時）を優先
2. 午前中（10-12時）または午後早め（14-16時）を推奨
3. 月曜朝や金曜夕方は避ける
4. 連続した会議を避けるため、前後に余裕がある時間を優先

${preferences ? `追加の要望: ${preferences}` : ""}

以下のJSON形式で回答してください:
{
  "recommendations": [
    {
      "index": 0,
      "score": 95,
      "reasons": ["午前中の集中しやすい時間", "全員の通常業務時間内"]
    }
  ]
}
`;
  }

  private parseOpenAIResponse(
    response: any,
    availableSlots: TimeSlot[]
  ): MeetingCandidate[] {
    const candidates: MeetingCandidate[] = [];

    if (response.recommendations && Array.isArray(response.recommendations)) {
      for (const rec of response.recommendations) {
        if (rec.index < availableSlots.length) {
          candidates.push({
            slot: availableSlots[rec.index],
            score: rec.score || 50,
            reasons: rec.reasons || []
          });
        }
      }
    }

    // 候補が少ない場合は追加
    if (candidates.length < 5) {
      const additionalCandidates = this.simpleScoring(availableSlots)
        .filter(c => !candidates.some(existing => 
          existing.slot.start.getTime() === c.slot.start.getTime()
        ));
      
      candidates.push(...additionalCandidates.slice(0, 5 - candidates.length));
    }

    return candidates.sort((a, b) => b.score - a.score);
  }

  private simpleScoring(slots: TimeSlot[]): MeetingCandidate[] {
    return slots.map(slot => {
      let score = 50;
      const reasons: string[] = [];
      const hour = slot.start.getHours();
      const day = slot.start.getDay();

      // 業務時間内
      if (hour >= 9 && hour < 18) {
        score += 20;
        reasons.push("業務時間内");
      }

      // 午前中のゴールデンタイム
      if (hour >= 10 && hour < 12) {
        score += 15;
        reasons.push("午前中の集中しやすい時間");
      }

      // 午後の良い時間
      if (hour >= 14 && hour < 16) {
        score += 10;
        reasons.push("午後の生産的な時間");
      }

      // 平日
      if (day >= 1 && day <= 5) {
        score += 10;
        reasons.push("平日");
      }

      // 避けるべき時間
      if (day === 1 && hour < 10) {
        score -= 10;
        reasons.push("月曜朝は避けた方が良い");
      }
      if (day === 5 && hour >= 16) {
        score -= 10;
        reasons.push("金曜夕方は避けた方が良い");
      }

      return {
        slot,
        score: Math.max(0, Math.min(100, score)),
        reasons
      };
    }).sort((a, b) => b.score - a.score).slice(0, 10);
  }

  combineAvailabilityResults(
    googleBusy: Map<string, TimeSlot[]>,
    hubspotBusy: Map<string, TimeSlot[]>,
    participants: Person[]
  ): AvailabilityResult[] {
    const results: AvailabilityResult[] = [];

    for (const person of participants) {
      let busySlots: TimeSlot[] = [];
      
      // 各参加者のソースに応じて予定を取得
      if (person.source === "google") {
        busySlots = googleBusy.get(person.sourceId) || [];
      } else if (person.source === "hubspot") {
        busySlots = hubspotBusy.get(person.sourceId) || [];
      }

      results.push({
        person,
        busySlots
      });
    }

    return results;
  }
}
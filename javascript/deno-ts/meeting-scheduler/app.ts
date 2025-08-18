#!/usr/bin/env -S deno run --allow-net --allow-env --allow-read

import { parseCliArgs } from "./cli/parser.ts";
import { GoogleCalendarClient } from "./clients/google-calendar.ts";
import { HubSpotClient } from "./clients/hubspot.ts";
import { AvailabilityAnalyzer } from "./processors/availability-analyzer.ts";
import { GoogleAuth } from "./utils/auth-google.ts";
import { HubSpotAuth } from "./utils/auth-hubspot.ts";
import { formatDateTime } from "./utils/date-utils.ts";
import { MeetingCandidate, TimeSlot, OpenAIConfig } from "./types/index.ts";

async function main() {
  try {
    // CLIオプションを解析
    const options = parseCliArgs(Deno.args);

    if (options.verbose) {
      console.log("設定:");
      console.log(`  期間: ${formatDateTime(options.startDate)} - ${formatDateTime(options.endDate)}`);
      console.log(`  会議時間: ${options.duration}分`);
      console.log(`  参加者: ${options.participants.length}人`);
      console.log(`  営業時間のみ: ${options.businessHoursOnly}`);
      console.log(`  OpenAI使用: ${options.useOpenAI}`);
    }

    // 認証の初期化
    let googleAuth: GoogleAuth | undefined;
    let hubspotAuth: HubSpotAuth | undefined;

    // Google Calendar認証（必要な場合）
    const hasGoogleCalendar = options.participants.some(p => p.source === "google");
    if (hasGoogleCalendar) {
      try {
        googleAuth = GoogleAuth.fromEnv();
        if (options.verbose) {
          console.log("✅ Google Calendar API 認証成功");
        }
      } catch (error) {
        console.warn("⚠️ Google Calendar API 認証スキップ:", error);
      }
    }

    // HubSpot認証（必要な場合）
    const hasHubSpot = options.participants.some(p => p.source === "hubspot");
    if (hasHubSpot) {
      try {
        hubspotAuth = HubSpotAuth.fromEnv();
        if (options.verbose) {
          console.log("✅ HubSpot API 認証成功");
        }
      } catch (error) {
        console.warn("⚠️ HubSpot API 認証スキップ:", error);
      }
    }

    // 各参加者の予定を取得
    const googleBusy = new Map<string, TimeSlot[]>();
    const hubspotBusy = new Map<string, TimeSlot[]>();

    // Google Calendarから予定を取得
    if (googleAuth) {
      const googleClient = new GoogleCalendarClient(googleAuth);
      
      if (options.verbose) {
        console.log("\nGoogle Calendarから予定を取得中...");
      }

      try {
        const busyData = await googleClient.getMultipleCalendarsBusy(
          options.participants,
          options.startDate,
          options.endDate
        );

        for (const [sourceId, slots] of busyData) {
          googleBusy.set(sourceId, slots);
          if (options.verbose) {
            const person = options.participants.find(p => p.sourceId === sourceId);
            console.log(`  ${person?.name || sourceId}: ${slots.length}件の予定`);
          }
        }
      } catch (error) {
        console.error("Google Calendar エラー:", error);
      }
    }

    // HubSpotから予定を取得
    if (hubspotAuth) {
      const hubspotClient = new HubSpotClient(hubspotAuth);
      
      if (options.verbose) {
        console.log("\nHubSpotから予定を取得中...");
      }

      try {
        const busyData = await hubspotClient.getMultipleUsersBusy(
          options.participants,
          options.startDate,
          options.endDate
        );

        for (const [sourceId, slots] of busyData) {
          hubspotBusy.set(sourceId, slots);
          if (options.verbose) {
            const person = options.participants.find(p => p.sourceId === sourceId);
            console.log(`  ${person?.name || sourceId}: ${slots.length}件の予定`);
          }
        }
      } catch (error) {
        console.error("HubSpot エラー:", error);
      }
    }

    // 空き時間を分析
    const analyzer = new AvailabilityAnalyzer(
      options.useOpenAI ? {
        apiKey: Deno.env.get("OPENAI_API_KEY") || "",
      } as OpenAIConfig : undefined
    );

    const availabilityResults = analyzer.combineAvailabilityResults(
      googleBusy,
      hubspotBusy,
      options.participants
    );

    const commonSlots = analyzer.analyzeAvailability(
      availabilityResults,
      options.startDate,
      options.endDate,
      options.duration,
      options.businessHoursOnly
    );

    if (options.verbose) {
      console.log(`\n共通の空き時間: ${commonSlots.length}件`);
    }

    // 最適な時間を選定
    let candidates: MeetingCandidate[];
    
    if (options.useOpenAI) {
      if (options.verbose) {
        console.log("OpenAI APIで最適化中...");
      }
      candidates = await analyzer.optimizeWithOpenAI(
        commonSlots,
        options.participants
      );
    } else {
      // OpenAIを使わない場合は単純なスコアリング
      const simpleAnalyzer = new AvailabilityAnalyzer();
      candidates = simpleAnalyzer["simpleScoring"](commonSlots);
    }

    // 結果を出力
    outputResults(candidates, options.outputFormat);

  } catch (error) {
    console.error("エラー:", error);
    Deno.exit(1);
  }
}

function outputResults(
  candidates: MeetingCandidate[],
  format: "text" | "json" | "markdown"
): void {
  if (candidates.length === 0) {
    console.log("\n指定された条件で利用可能な時間が見つかりませんでした。");
    return;
  }

  switch (format) {
    case "json":
      console.log(JSON.stringify({
        totalCandidates: candidates.length,
        topCandidates: candidates.slice(0, 5).map(c => ({
          start: c.slot.start.toISOString(),
          end: c.slot.end.toISOString(),
          score: c.score,
          reasons: c.reasons
        }))
      }, null, 2));
      break;

    case "markdown":
      console.log("# 会議候補時間\n");
      console.log(`合計 ${candidates.length} 件の候補が見つかりました。\n`);
      console.log("## おすすめトップ5\n");
      
      candidates.slice(0, 5).forEach((candidate, index) => {
        console.log(`### ${index + 1}. ${formatDateTime(candidate.slot.start)}`);
        console.log(`- **時間**: ${formatDateTime(candidate.slot.start)} - ${formatDateTime(candidate.slot.end)}`);
        console.log(`- **スコア**: ${candidate.score}/100`);
        if (candidate.reasons.length > 0) {
          console.log(`- **理由**:`);
          candidate.reasons.forEach(reason => {
            console.log(`  - ${reason}`);
          });
        }
        console.log("");
      });
      break;

    case "text":
    default:
      console.log("\n=== 会議候補時間 ===");
      console.log(`合計 ${candidates.length} 件の候補が見つかりました。\n`);
      console.log("おすすめトップ5:");
      
      candidates.slice(0, 5).forEach((candidate, index) => {
        console.log(`\n${index + 1}. ${formatDateTime(candidate.slot.start)} - ${formatDateTime(candidate.slot.end)}`);
        console.log(`   スコア: ${candidate.score}/100`);
        if (candidate.reasons.length > 0) {
          console.log(`   理由: ${candidate.reasons.join(", ")}`);
        }
      });
      
      if (candidates.length > 5) {
        console.log(`\n他 ${candidates.length - 5} 件の候補があります。`);
      }
      break;
  }
}

// メイン関数を実行
if (import.meta.main) {
  main();
}
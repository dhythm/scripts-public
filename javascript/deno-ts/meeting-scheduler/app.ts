#!/usr/bin/env -S deno run --allow-net --allow-env --allow-read

import { parseCliArgs } from "./cli/parser.ts";
import { loadEnv } from "./utils/env.ts";
import { GoogleCalendarClient } from "./clients/google-calendar.ts";
import { HubSpotClient } from "./clients/hubspot.ts";
import { AvailabilityAnalyzer } from "./processors/availability-analyzer.ts";
import { GoogleAuth } from "./utils/auth-google.ts";
import { HubSpotAuth } from "./utils/auth-hubspot.ts";
import { formatDateTime, findRawAvailableSlots } from "./utils/date-utils.ts";
import { MeetingCandidate, TimeSlot, OpenAIConfig, CliOptions } from "./types/index.ts";

async function main() {
  try {
    // .envファイルを読み込み
    await loadEnv();
    
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
        model: Deno.env.get("OPENAI_MODEL") || "gpt-4o-mini",
      } as OpenAIConfig : undefined
    );

    const availabilityResults = analyzer.combineAvailabilityResults(
      googleBusy,
      hubspotBusy,
      options.participants
    );

    // rawSlotsモードの場合は連続した空き時間を表示
    if (options.rawSlots) {
      const rawSlots = findRawAvailableSlots(
        availabilityResults,
        options.startDate,
        options.endDate,
        options.businessHoursOnly
      );
      
      if (options.verbose) {
        console.log(`\n連続した空き時間ブロック: ${rawSlots.length}件`);
      }
      
      outputRawSlots(rawSlots, options);
    } else {
      // 通常の会議候補時間表示
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
      outputResults(candidates, options);
    }

  } catch (error) {
    console.error("エラー:", error);
    Deno.exit(1);
  }
}

function outputResults(
  candidates: MeetingCandidate[],
  options: CliOptions
): void {
  const format = options.outputFormat;
  const showAll = options.showAll;
  const limit = options.limit || (showAll ? candidates.length : 5);
  
  if (candidates.length === 0) {
    console.log("\n指定された条件で利用可能な時間が見つかりませんでした。");
    return;
  }

  switch (format) {
    case "json":
      console.log(JSON.stringify({
        totalCandidates: candidates.length,
        displayedCandidates: Math.min(limit, candidates.length),
        candidates: candidates.slice(0, limit).map(c => ({
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
      
      const displayCount = Math.min(limit, candidates.length);
      if (showAll) {
        console.log(`## 全ての空き時間（${displayCount}件）\n`);
      } else {
        console.log(`## おすすめトップ${displayCount}\n`);
      }
      
      candidates.slice(0, limit).forEach((candidate, index) => {
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
      
      if (candidates.length > limit && !showAll) {
        console.log(`\n他 ${candidates.length - limit} 件の候補があります。全て表示するには --show-all オプションを使用してください。`);
      }
      break;

    case "text":
    default:
      console.log("\n=== 会議候補時間 ===");
      console.log(`合計 ${candidates.length} 件の候補が見つかりました。\n`);
      
      const displayCount2 = Math.min(limit, candidates.length);
      if (showAll) {
        console.log(`全ての空き時間（${displayCount2}件）:\n`);
      } else {
        console.log(`おすすめトップ${displayCount2}:\n`);
      }
      
      candidates.slice(0, limit).forEach((candidate, index) => {
        const duration = Math.round((candidate.slot.end.getTime() - candidate.slot.start.getTime()) / 60000);
        console.log(`${index + 1}. ${formatDateTime(candidate.slot.start)} - ${formatDateTime(candidate.slot.end)} （${duration}分）`);
        if (candidate.reasons.length > 0) {
          console.log(`   理由: ${candidate.reasons.join(", ")} | スコア: ${candidate.score}/100`);
        } else {
          console.log(`   スコア: ${candidate.score}/100`);
        }
      });
      
      if (candidates.length > limit && !showAll) {
        console.log(`\n他 ${candidates.length - limit} 件の候補があります。全て表示するには --show-all オプションを使用してください。`);
      }
      break;
  }
}

function outputRawSlots(
  slots: TimeSlot[],
  options: CliOptions
): void {
  const format = options.outputFormat;
  
  if (slots.length === 0) {
    console.log("\n指定された条件で利用可能な時間が見つかりませんでした。");
    return;
  }

  switch (format) {
    case "json":
      console.log(JSON.stringify({
        totalSlots: slots.length,
        availableSlots: slots.map(s => ({
          start: s.start.toISOString(),
          end: s.end.toISOString(),
          duration: Math.round((s.end.getTime() - s.start.getTime()) / 60000) + "分"
        }))
      }, null, 2));
      break;

    case "markdown":
      console.log("# 空き時間ブロック\n");
      console.log(`合計 ${slots.length} 件の連続した空き時間が見つかりました。\n`);
      console.log("## 利用可能な時間帯\n");
      
      slots.forEach((slot, index) => {
        const duration = Math.round((slot.end.getTime() - slot.start.getTime()) / 60000);
        console.log(`### ${index + 1}. ${formatDateTime(slot.start)} - ${formatDateTime(slot.end)}`);
        console.log(`- **時間**: ${duration}分`);
        console.log(`- **開始**: ${formatDateTime(slot.start)}`);
        console.log(`- **終了**: ${formatDateTime(slot.end)}`);
        console.log("");
      });
      break;

    case "text":
    default:
      console.log("\n=== 空き時間ブロック ===");
      console.log(`合計 ${slots.length} 件の連続した空き時間が見つかりました。\n`);
      console.log("利用可能な時間帯:\n");
      
      slots.forEach((slot, index) => {
        const duration = Math.round((slot.end.getTime() - slot.start.getTime()) / 60000);
        console.log(`${index + 1}. ${formatDateTime(slot.start)} - ${formatDateTime(slot.end)} （${duration}分）`);
      });
      break;
  }
}

// メイン関数を実行
if (import.meta.main) {
  main();
}
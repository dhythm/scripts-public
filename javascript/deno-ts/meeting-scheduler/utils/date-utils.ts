import { TimeSlot } from "../types/index.ts";

const HOLIDAY_API_URL = "https://holidays-jp.github.io/api/v1/date.json";
const JST_TIMEZONE = "Asia/Tokyo";

let japaneseHolidayCache: Map<string, string> | null = null;
let holidayFetchPromise: Promise<Map<string, string>> | null = null;

export function parseDateTime(dateStr: string): Date {
  const date = new Date(dateStr);
  if (isNaN(date.getTime())) {
    throw new Error(`無効な日時形式: ${dateStr}`);
  }
  return date;
}

export function formatDateTime(date: Date, timezone = "Asia/Tokyo"): string {
  return date.toLocaleString("ja-JP", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatDateTimeWithDay(date: Date, timezone = "Asia/Tokyo"): string {
  const dateStr = date.toLocaleString("ja-JP", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
  const dayStr = ["日", "月", "火", "水", "木", "金", "土"][date.getDay()];
  // 日付と時刻の間に曜日を挿入 (例: 2024/08/19 → 2024/08/19(月))
  const parts = dateStr.split(" ");
  return `${parts[0]}(${dayStr}) ${parts[1]}`;
}

export function formatTimeOnly(date: Date, timezone = "Asia/Tokyo"): string {
  return date.toLocaleString("ja-JP", {
    timeZone: timezone,
    hour: "2-digit",
    minute: "2-digit",
  });
}

function toJstDateKey(date: Date): string {
  const [year, month, day] = date
    .toLocaleDateString("ja-JP", {
      timeZone: JST_TIMEZONE,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    })
    .split("/");
  return `${year}-${month}-${day}`;
}

async function ensureHolidayCache(): Promise<Map<string, string>> {
  if (japaneseHolidayCache) {
    return japaneseHolidayCache;
  }

  if (!holidayFetchPromise) {
    holidayFetchPromise = (async () => {
      try {
        const response = await fetch(HOLIDAY_API_URL);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = (await response.json()) as Record<string, string>;
        japaneseHolidayCache = new Map(Object.entries(data));
      } catch (error) {
        console.warn("祝日データの取得に失敗しました。祝日除外をスキップします。", error);
        japaneseHolidayCache = new Map();
      }
      return japaneseHolidayCache!;
    })();
  }

  return holidayFetchPromise;
}

function createHolidaySlot(dateKey: string): TimeSlot {
  const start = new Date(`${dateKey}T00:00:00+09:00`);
  const end = new Date(start);
  end.setDate(end.getDate() + 1);
  return { start, end };
}

export async function loadJapaneseHolidays(
  start: Date,
  end: Date
): Promise<Set<string>> {
  const cache = await ensureHolidayCache();
  if (!cache || cache.size === 0) {
    return new Set();
  }

  const startKey = toJstDateKey(start);
  const endKey = toJstDateKey(end);
  const result = new Set<string>();

  for (const dateKey of cache.keys()) {
    if (dateKey >= startKey && dateKey <= endKey) {
      result.add(dateKey);
    }
  }

  return result;
}

export function isJapaneseHoliday(
  date: Date,
  holidays?: Set<string>
): boolean {
  if (!holidays || holidays.size === 0) {
    return false;
  }
  return holidays.has(toJstDateKey(date));
}

export function isSameDay(date1: Date, date2: Date): boolean {
  return date1.getFullYear() === date2.getFullYear() &&
         date1.getMonth() === date2.getMonth() &&
         date1.getDate() === date2.getDate();
}

export function addMinutes(date: Date, minutes: number): Date {
  return new Date(date.getTime() + minutes * 60 * 1000);
}

export function isBusinessHours(
  date: Date,
  holidays?: Set<string>,
  skipHolidays = false
): boolean {
  const hours = date.getHours();
  const day = date.getDay();

  if (day === 0 || day === 6) {
    return false;
  }

  if (skipHolidays && isJapaneseHoliday(date, holidays)) {
    return false;
  }

  // 平日の9時から18時
  return hours >= 9 && hours < 18;
}

export function generateTimeSlots(
  start: Date,
  end: Date,
  durationMinutes: number,
  businessHoursOnly = false,
  skipHolidays = false,
  holidays?: Set<string>
): TimeSlot[] {
  const slots: TimeSlot[] = [];
  let current = new Date(start);
  
  while (current < end) {
    const slotEnd = addMinutes(current, durationMinutes);
    
    if (slotEnd <= end) {
      const isHolidaySlot = skipHolidays && (
        isJapaneseHoliday(current, holidays) || isJapaneseHoliday(slotEnd, holidays)
      );

      if (!isHolidaySlot && (!businessHoursOnly || (
        isBusinessHours(current, holidays, skipHolidays) &&
        isBusinessHours(slotEnd, holidays, skipHolidays)
      ))) {
        slots.push({
          start: new Date(current),
          end: slotEnd,
        });
      }
    }
    
    current = addMinutes(current, 30); // 30分刻みでスロットを生成
  }
  
  return slots;
}

export function isSlotAvailable(
  slot: TimeSlot,
  busySlots: TimeSlot[]
): boolean {
  return !busySlots.some(busy => 
    (slot.start >= busy.start && slot.start < busy.end) ||
    (slot.end > busy.start && slot.end <= busy.end) ||
    (slot.start <= busy.start && slot.end >= busy.end)
  );
}

export function findCommonAvailableSlots(
  availabilityResults: { busySlots: TimeSlot[] }[],
  start: Date,
  end: Date,
  durationMinutes: number,
  businessHoursOnly = false,
  skipHolidays = false,
  holidays?: Set<string>
): TimeSlot[] {
  const allSlots = generateTimeSlots(
    start,
    end,
    durationMinutes,
    businessHoursOnly,
    skipHolidays,
    holidays
  );
  
  return allSlots.filter(slot => 
    availabilityResults.every(result => 
      isSlotAvailable(slot, result.busySlots)
    )
  );
}

export function findRawAvailableSlots(
  availabilityResults: { busySlots: TimeSlot[] }[],
  start: Date,
  end: Date,
  businessHoursOnly = false,
  minDurationMinutes?: number,
  skipHolidays = false,
  holidays?: Set<string>
): TimeSlot[] {
  // 全ての忙しい時間を統合
  const allBusySlots: TimeSlot[] = [];
  for (const result of availabilityResults) {
    allBusySlots.push(...result.busySlots);
  }

  if (skipHolidays && holidays && holidays.size > 0) {
    for (const dateKey of holidays) {
      allBusySlots.push(createHolidaySlot(dateKey));
    }
  }
  
  // 忙しい時間をマージして重複を除去
  const mergedBusy = mergeTimeSlots(allBusySlots);
  
  // 忙しい時間がない場合、全時間が空き
  if (mergedBusy.length === 0) {
    if (businessHoursOnly) {
      return splitByBusinessHours([{ start, end }], holidays, skipHolidays);
    }
    return [{ start, end }];
  }
  
  // 空き時間を計算
  const availableSlots: TimeSlot[] = [];
  let currentStart = new Date(start);
  
  for (const busy of mergedBusy) {
    if (currentStart < busy.start) {
      availableSlots.push({
        start: new Date(currentStart),
        end: new Date(busy.start),
      });
    }
    currentStart = new Date(Math.max(currentStart.getTime(), busy.end.getTime()));
  }
  
  // 最後の忙しい時間から終了時刻までの空き時間
  if (currentStart < end) {
    availableSlots.push({
      start: new Date(currentStart),
      end: new Date(end),
    });
  }
  
  // 営業時間でフィルタリング（必要な場合）
  let filteredSlots = availableSlots;
  if (businessHoursOnly) {
    filteredSlots = splitByBusinessHours(availableSlots, holidays, skipHolidays);
  } else if (skipHolidays && holidays && holidays.size > 0) {
    filteredSlots = filteredSlots.filter(slot =>
      !isJapaneseHoliday(slot.start, holidays) &&
      !isJapaneseHoliday(slot.end, holidays)
    );
  }
  
  // 最小時間でフィルタリング
  if (minDurationMinutes && minDurationMinutes > 0) {
    filteredSlots = filteredSlots.filter(slot => {
      const durationInMinutes = (slot.end.getTime() - slot.start.getTime()) / 60000;
      return durationInMinutes >= minDurationMinutes;
    });
  }
  
  return filteredSlots;
}

function splitByBusinessHours(
  slots: TimeSlot[],
  holidays?: Set<string>,
  skipHolidays = false
): TimeSlot[] {
  const result: TimeSlot[] = [];
  
  for (const slot of slots) {
    let current = new Date(slot.start);
    
    while (current < slot.end) {
      const dayStart = new Date(current);
      dayStart.setHours(9, 0, 0, 0);
      const dayEnd = new Date(current);
      dayEnd.setHours(18, 0, 0, 0);
      
      const isWeekend = current.getDay() === 0 || current.getDay() === 6;
      const isHolidayDay = skipHolidays && isJapaneseHoliday(current, holidays);

      if (isWeekend || isHolidayDay) {
        current = new Date(current);
        current.setDate(current.getDate() + 1);
        current.setHours(9, 0, 0, 0);
        continue;
      }
      
      const slotStart = new Date(Math.max(current.getTime(), dayStart.getTime(), slot.start.getTime()));
      const slotEnd = new Date(Math.min(dayEnd.getTime(), slot.end.getTime()));
      
      if (slotStart < slotEnd) {
        result.push({
          start: slotStart,
          end: slotEnd,
        });
      }
      
      // 次の日の9時に移動
      current = new Date(current);
      current.setDate(current.getDate() + 1);
      current.setHours(9, 0, 0, 0);
    }
  }
  
  return result;
}

export function mergeTimeSlots(slots: TimeSlot[]): TimeSlot[] {
  if (slots.length === 0) return [];
  
  const sorted = slots.sort((a, b) => a.start.getTime() - b.start.getTime());
  const merged: TimeSlot[] = [sorted[0]];
  
  for (let i = 1; i < sorted.length; i++) {
    const last = merged[merged.length - 1];
    const current = sorted[i];
    
    if (last.end >= current.start) {
      last.end = new Date(Math.max(last.end.getTime(), current.end.getTime()));
    } else {
      merged.push(current);
    }
  }
  
  return merged;
}

import { TimeSlot } from "../types/index.ts";

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

export function addMinutes(date: Date, minutes: number): Date {
  return new Date(date.getTime() + minutes * 60 * 1000);
}

export function isBusinessHours(date: Date): boolean {
  const hours = date.getHours();
  const day = date.getDay();
  
  // 平日の9時から18時
  return day >= 1 && day <= 5 && hours >= 9 && hours < 18;
}

export function generateTimeSlots(
  start: Date,
  end: Date,
  durationMinutes: number,
  businessHoursOnly = false
): TimeSlot[] {
  const slots: TimeSlot[] = [];
  let current = new Date(start);
  
  while (current < end) {
    const slotEnd = addMinutes(current, durationMinutes);
    
    if (slotEnd <= end) {
      if (!businessHoursOnly || (isBusinessHours(current) && isBusinessHours(slotEnd))) {
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
  businessHoursOnly = false
): TimeSlot[] {
  const allSlots = generateTimeSlots(start, end, durationMinutes, businessHoursOnly);
  
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
  businessHoursOnly = false
): TimeSlot[] {
  // 全ての忙しい時間を統合
  const allBusySlots: TimeSlot[] = [];
  for (const result of availabilityResults) {
    allBusySlots.push(...result.busySlots);
  }
  
  // 忙しい時間をマージして重複を除去
  const mergedBusy = mergeTimeSlots(allBusySlots);
  
  // 忙しい時間がない場合、全時間が空き
  if (mergedBusy.length === 0) {
    if (businessHoursOnly) {
      return splitByBusinessHours([{ start, end }]);
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
  if (businessHoursOnly) {
    return splitByBusinessHours(availableSlots);
  }
  
  return availableSlots;
}

function splitByBusinessHours(slots: TimeSlot[]): TimeSlot[] {
  const result: TimeSlot[] = [];
  
  for (const slot of slots) {
    let current = new Date(slot.start);
    
    while (current < slot.end) {
      const dayStart = new Date(current);
      dayStart.setHours(9, 0, 0, 0);
      const dayEnd = new Date(current);
      dayEnd.setHours(18, 0, 0, 0);
      
      // 週末をスキップ
      if (current.getDay() === 0 || current.getDay() === 6) {
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
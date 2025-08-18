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
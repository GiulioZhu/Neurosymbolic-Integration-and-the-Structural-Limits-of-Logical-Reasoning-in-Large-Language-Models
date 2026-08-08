import { it } from "date-fns/locale";
import {
  addDays,
  addMonths,
  differenceInCalendarDays,
  endOfMonth,
  endOfWeek,
  format,
  isSameDay,
  parseISO,
  startOfMonth,
  startOfWeek,
} from "date-fns";
import type { ISODate, ISOTime } from "./types";

export const LOCALE = it;
/** In Italia la settimana comincia di lunedì. */
export const WEEK_OPTS = { weekStartsOn: 1 as const, locale: it };

/** `yyyy-MM-dd` di una data locale (mai `toISOString`, che sposta di fuso). */
export function toISODate(d: Date): ISODate {
  return format(d, "yyyy-MM-dd");
}

/** Interpreta `yyyy-MM-dd` come mezzanotte **locale**. */
export function fromISODate(s: ISODate): Date {
  const [y, m, d] = s.split("-").map(Number);
  return new Date(y, (m ?? 1) - 1, d ?? 1);
}

export function todayISO(): ISODate {
  return toISODate(new Date());
}

/** Normalizza `HH:mm:ss` (Postgres) o `H:mm` in `HH:mm`. */
export function normalizeTime(t: string | null | undefined): ISOTime | null {
  if (!t) return null;
  const m = /^(\d{1,2}):(\d{2})/.exec(t.trim());
  if (!m) return null;
  const h = Math.min(23, Number(m[1]));
  const min = Math.min(59, Number(m[2]));
  return `${String(h).padStart(2, "0")}:${String(min).padStart(2, "0")}`;
}

/** Unisce data e (eventuale) ora in un `Date`. Senza ora → mezzanotte. */
export function combine(date: ISODate, time?: ISOTime | null): Date {
  const base = fromISODate(date);
  const t = normalizeTime(time ?? null);
  if (t) {
    const [h, m] = t.split(":").map(Number);
    base.setHours(h, m, 0, 0);
  }
  return base;
}

export function addMinutes(d: Date, minutes: number): Date {
  return new Date(d.getTime() + minutes * 60_000);
}

// --- formattazione ---------------------------------------------------------

export function formatLongDate(d: Date): string {
  return format(d, "EEEE d MMMM", { locale: it });
}

export function formatDayMonth(d: Date): string {
  return format(d, "d MMM", { locale: it });
}

export function formatWeekdayShort(d: Date): string {
  return format(d, "EEE", { locale: it });
}

export function formatMonthYear(d: Date): string {
  return format(d, "LLLL yyyy", { locale: it });
}

export function formatMonthName(d: Date): string {
  return format(d, "LLLL", { locale: it });
}

export function formatTime(t: ISOTime | null | undefined): string {
  return normalizeTime(t ?? null) ?? "";
}

/** "oggi", "domani", "ieri", "lun 10", "10 mar 2027". */
export function relativeDay(dateISO: ISODate, from = new Date()): string {
  const d = fromISODate(dateISO);
  const diff = differenceInCalendarDays(d, from);
  if (diff === 0) return "oggi";
  if (diff === 1) return "domani";
  if (diff === -1) return "ieri";
  if (diff > 1 && diff < 7) return format(d, "EEEE", { locale: it });
  if (diff < -1 && diff > -7) return `${Math.abs(diff)} giorni fa`;
  if (d.getFullYear() === from.getFullYear()) return format(d, "d MMM", { locale: it });
  return format(d, "d MMM yyyy", { locale: it });
}

export function isToday(dateISO: ISODate): boolean {
  return isSameDay(fromISODate(dateISO), new Date());
}

export function isPast(dateISO: ISODate): boolean {
  return differenceInCalendarDays(fromISODate(dateISO), new Date()) < 0;
}

// --- intervalli ------------------------------------------------------------

export interface Range {
  start: Date;
  end: Date;
}

export function dayRange(d = new Date()): Range {
  const start = new Date(d);
  start.setHours(0, 0, 0, 0);
  const end = new Date(start);
  end.setHours(23, 59, 59, 999);
  return { start, end };
}

export function weekRange(d = new Date()): Range {
  return { start: startOfWeek(d, WEEK_OPTS), end: endOfWeek(d, WEEK_OPTS) };
}

export function monthRange(d = new Date()): Range {
  return { start: startOfMonth(d), end: endOfMonth(d) };
}

export function yearRange(d = new Date()): Range {
  return {
    start: new Date(d.getFullYear(), 0, 1),
    end: new Date(d.getFullYear(), 11, 31, 23, 59, 59, 999),
  };
}

export function inRange(dateISO: ISODate, r: Range): boolean {
  const t = fromISODate(dateISO).getTime();
  return t >= r.start.getTime() && t <= r.end.getTime();
}

/** Le 6 righe (42 giorni) mostrate dalla griglia mensile. */
export function monthGrid(month: Date): Date[] {
  const start = startOfWeek(startOfMonth(month), WEEK_OPTS);
  return Array.from({ length: 42 }, (_, i) => addDays(start, i));
}

export function weekDays(d: Date): Date[] {
  const start = startOfWeek(d, WEEK_OPTS);
  return Array.from({ length: 7 }, (_, i) => addDays(start, i));
}

export { addDays, addMonths, differenceInCalendarDays, isSameDay, parseISO, startOfMonth, endOfMonth };

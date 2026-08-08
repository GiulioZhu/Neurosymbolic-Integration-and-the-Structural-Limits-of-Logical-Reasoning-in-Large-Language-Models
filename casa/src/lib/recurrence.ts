import { addDays, differenceInCalendarDays, fromISODate, toISODate, startOfMonth } from "./date";
import type { ISODate, Recurrence } from "./types";

const WEEKDAY_NAMES = [
  "domenica",
  "lunedì",
  "martedì",
  "mercoledì",
  "giovedì",
  "venerdì",
  "sabato",
];

function weeksBetween(a: Date, b: Date): number {
  // lunedì come inizio settimana
  const monday = (d: Date) => {
    const x = new Date(d);
    const shift = (x.getDay() + 6) % 7;
    x.setDate(x.getDate() - shift);
    x.setHours(0, 0, 0, 0);
    return x;
  };
  return Math.round((monday(b).getTime() - monday(a).getTime()) / (7 * 86_400_000));
}

function monthsBetween(a: Date, b: Date): number {
  return (b.getFullYear() - a.getFullYear()) * 12 + (b.getMonth() - a.getMonth());
}

/** La serie ancorata a `anchor` ha un'occorrenza il giorno `day`? */
export function occursOn(anchor: ISODate, rule: Recurrence, day: Date): boolean {
  const start = fromISODate(anchor);
  const target = new Date(day);
  target.setHours(0, 0, 0, 0);

  if (target < start) return false;
  if (rule.until && target > fromISODate(rule.until)) return false;

  const interval = Math.max(1, rule.interval || 1);

  switch (rule.freq) {
    case "daily":
      return differenceInCalendarDays(target, start) % interval === 0;

    case "weekly": {
      if (weeksBetween(start, target) % interval !== 0) return false;
      const days = rule.byweekday?.length ? rule.byweekday : [start.getDay()];
      return days.includes(target.getDay());
    }

    case "monthly": {
      if (monthsBetween(start, target) % interval !== 0) return false;
      const wanted = rule.bymonthday ?? start.getDate();
      if (target.getDate() === wanted) return true;
      // giorno 31 in un mese da 30: cade sull'ultimo giorno disponibile
      const lastOfMonth = new Date(target.getFullYear(), target.getMonth() + 1, 0).getDate();
      return wanted > lastOfMonth && target.getDate() === lastOfMonth;
    }

    default:
      return false;
  }
}

/** Prima occorrenza strettamente successiva ad `after`. */
export function nextOccurrence(
  anchor: ISODate,
  rule: Recurrence,
  after: ISODate,
): ISODate | null {
  let cursor = addDays(fromISODate(after), 1);
  // 800 giorni coprono anche "ogni 2 anni" espresso in mesi
  for (let i = 0; i < 800; i++) {
    if (rule.until && cursor > fromISODate(rule.until)) return null;
    if (occursOn(anchor, rule, cursor)) return toISODate(cursor);
    cursor = addDays(cursor, 1);
  }
  return null;
}

/** Tutte le occorrenze dentro un intervallo (estremi inclusi). */
export function expandDates(
  anchor: ISODate,
  rule: Recurrence | null,
  from: Date,
  to: Date,
): ISODate[] {
  if (!rule) {
    const d = fromISODate(anchor);
    return d >= startOfDay(from) && d <= to ? [anchor] : [];
  }
  const out: ISODate[] = [];
  let cursor = startOfDay(from);
  const limit = new Date(to);
  let guard = 0;
  while (cursor <= limit && guard++ < 400) {
    if (occursOn(anchor, rule, cursor)) out.push(toISODate(cursor));
    cursor = addDays(cursor, 1);
  }
  return out;
}

function startOfDay(d: Date): Date {
  const x = new Date(d);
  x.setHours(0, 0, 0, 0);
  return x;
}

/** "ogni lunedì e giovedì", "ogni 2 mesi il 15". */
export function describeRecurrence(rule: Recurrence | null): string {
  if (!rule) return "Mai";
  const n = Math.max(1, rule.interval || 1);

  if (rule.freq === "daily") return n === 1 ? "Ogni giorno" : `Ogni ${n} giorni`;

  if (rule.freq === "weekly") {
    const base = n === 1 ? "Ogni" : `Ogni ${n} settimane,`;
    if (rule.byweekday?.length) {
      const names = [...rule.byweekday].sort().map((d) => WEEKDAY_NAMES[d]);
      const joined =
        names.length === 1
          ? names[0]
          : `${names.slice(0, -1).join(", ")} e ${names[names.length - 1]}`;
      return `${base} ${joined}`;
    }
    return n === 1 ? "Ogni settimana" : `Ogni ${n} settimane`;
  }

  const day = rule.bymonthday ? ` il ${rule.bymonthday}` : "";
  return n === 1 ? `Ogni mese${day}` : `Ogni ${n} mesi${day}`;
}

/** Regola pronta all'uso a partire da una data (usata dai form). */
export function ruleFromPreset(
  preset: "none" | "daily" | "weekly" | "monthly",
  date: ISODate,
): Recurrence | null {
  if (preset === "none") return null;
  const d = fromISODate(date);
  if (preset === "daily") return { freq: "daily", interval: 1 };
  if (preset === "weekly") return { freq: "weekly", interval: 1, byweekday: [d.getDay()] };
  return { freq: "monthly", interval: 1, bymonthday: d.getDate() };
}

export function presetFromRule(rule: Recurrence | null): "none" | "daily" | "weekly" | "monthly" | "custom" {
  if (!rule) return "none";
  const simple = (rule.interval ?? 1) === 1;
  if (!simple) return "custom";
  if (rule.freq === "daily") return "daily";
  if (rule.freq === "weekly") return (rule.byweekday?.length ?? 0) <= 1 ? "weekly" : "custom";
  if (rule.freq === "monthly") return "monthly";
  return "custom";
}

export { WEEKDAY_NAMES, startOfMonth };

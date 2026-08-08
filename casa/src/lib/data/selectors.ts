/** Viste derivate sullo snapshot. Funzioni pure: facili da testare, veloci. */

import {
  combine,
  dayRange,
  differenceInCalendarDays,
  fromISODate,
  inRange,
  monthRange,
  todayISO,
  toISODate,
  weekRange,
  type Range,
} from "../date";
import { expandDates } from "../recurrence";
import type {
  Category,
  EventItem,
  Expense,
  ISODate,
  Note,
  Task,
  UUID,
} from "../types";
import type { Snapshot } from "./driver";

// ---------------------------------------------------------------------------
//  Task
// ---------------------------------------------------------------------------

export const isOpen = (t: Task) => t.status !== "done";

export function tasksDueOn(tasks: Task[], date: ISODate): Task[] {
  return tasks.filter((t) => t.due_date === date).sort(byTaskOrder);
}

export function tasksToday(tasks: Task[], date = todayISO()): Task[] {
  return tasks.filter((t) => t.due_date === date).sort(byTaskOrder);
}

export function tasksOverdue(tasks: Task[], date = todayISO()): Task[] {
  return tasks
    .filter((t) => isOpen(t) && t.due_date !== null && t.due_date < date)
    .sort((a, b) => (a.due_date ?? "").localeCompare(b.due_date ?? ""));
}

export function tasksUpcoming(tasks: Task[], days = 7, date = todayISO()): Task[] {
  const limit = toISODate(new Date(fromISODate(date).getTime() + days * 86_400_000));
  return tasks
    .filter((t) => isOpen(t) && t.due_date !== null && t.due_date > date && t.due_date <= limit)
    .sort(byTaskOrder);
}

export function tasksSomeday(tasks: Task[]): Task[] {
  return tasks.filter((t) => isOpen(t) && !t.due_date).sort(byTaskOrder);
}

const PRIORITY_RANK = { urgent: 0, high: 1, normal: 2, low: 3 } as const;

export function byTaskOrder(a: Task, b: Task): number {
  if (isOpen(a) !== isOpen(b)) return isOpen(a) ? -1 : 1;
  const da = a.due_date ?? "9999-12-31";
  const db = b.due_date ?? "9999-12-31";
  if (da !== db) return da.localeCompare(db);
  const ta = a.due_time ?? "99:99";
  const tb = b.due_time ?? "99:99";
  if (ta !== tb) return ta.localeCompare(tb);
  if (a.priority !== b.priority) return PRIORITY_RANK[a.priority] - PRIORITY_RANK[b.priority];
  return a.created_at.localeCompare(b.created_at);
}

/** La prossima cosa da fare: task aperta con l'orario più vicino da adesso. */
export function nextTask(tasks: Task[], now = new Date()): Task | null {
  const candidates = tasks
    .filter((t) => isOpen(t) && t.due_date)
    .map((t) => ({ t, at: combine(t.due_date!, t.due_time) }))
    .filter((x) => x.at.getTime() >= now.getTime() - 60_000)
    .sort((a, b) => a.at.getTime() - b.at.getTime());
  return candidates[0]?.t ?? null;
}

// ---------------------------------------------------------------------------
//  Eventi (con espansione delle ricorrenze)
// ---------------------------------------------------------------------------

export interface EventOccurrence {
  event: EventItem;
  date: ISODate;
  /** vero se generata da una ricorrenza e non dalla riga originale */
  repeated: boolean;
}

export function eventsBetween(events: EventItem[], from: Date, to: Date): EventOccurrence[] {
  const out: EventOccurrence[] = [];
  for (const e of events) {
    for (const date of expandDates(e.start_date, e.recurrence, from, to)) {
      out.push({ event: e, date, repeated: date !== e.start_date });
    }
  }
  return out.sort(
    (a, b) =>
      a.date.localeCompare(b.date) ||
      (a.event.start_time ?? "00:00").localeCompare(b.event.start_time ?? "00:00"),
  );
}

export function eventsOn(events: EventItem[], date: ISODate): EventOccurrence[] {
  const d = fromISODate(date);
  return eventsBetween(events, d, d);
}

export function nextEvent(events: EventItem[], now = new Date()): EventOccurrence | null {
  const horizon = new Date(now.getTime() + 60 * 86_400_000);
  const upcoming = eventsBetween(events, now, horizon).filter(
    (o) => combine(o.date, o.event.start_time).getTime() >= now.getTime() - 60_000,
  );
  return upcoming[0] ?? null;
}

// ---------------------------------------------------------------------------
//  Spese
// ---------------------------------------------------------------------------

export function expensesIn(expenses: Expense[], range: Range): Expense[] {
  return expenses
    .filter((e) => inRange(e.spent_on, range))
    .sort((a, b) => b.spent_on.localeCompare(a.spent_on) || b.created_at.localeCompare(a.created_at));
}

export const sum = (rows: Expense[]) => rows.reduce((acc, e) => acc + e.amount_cents, 0);

export interface CategoryTotal {
  category: Category | null;
  slug: string;
  name: string;
  color: string;
  total: number;
  share: number;
}

export function totalsByCategory(rows: Expense[], categories: Category[]): CategoryTotal[] {
  const total = sum(rows);
  const map = new Map<string, number>();
  for (const e of rows) {
    const key = e.category_id ?? "none";
    map.set(key, (map.get(key) ?? 0) + e.amount_cents);
  }
  return [...map.entries()]
    .map(([key, value]) => {
      const cat = categories.find((c) => c.id === key) ?? null;
      return {
        category: cat,
        slug: cat?.slug ?? "altro",
        name: cat?.name ?? "Senza categoria",
        color: cat?.color ?? "#8A8175",
        total: value,
        share: total ? value / total : 0,
      };
    })
    .sort((a, b) => b.total - a.total);
}

export interface PersonTotal {
  userId: UUID;
  total: number;
  share: number;
}

export function totalsByPerson(rows: Expense[]): PersonTotal[] {
  const total = sum(rows);
  const map = new Map<UUID, number>();
  for (const e of rows) map.set(e.paid_by, (map.get(e.paid_by) ?? 0) + e.amount_cents);
  return [...map.entries()]
    .map(([userId, value]) => ({ userId, total: value, share: total ? value / total : 0 }))
    .sort((a, b) => b.total - a.total);
}

/** Totale per ciascun giorno del mese, per la sparkline. */
export function dailyTotals(rows: Expense[], range: Range): { date: ISODate; total: number }[] {
  const out: { date: ISODate; total: number }[] = [];
  const cursor = new Date(range.start);
  while (cursor <= range.end) {
    const iso = toISODate(cursor);
    out.push({ date: iso, total: sum(rows.filter((e) => e.spent_on === iso)) });
    cursor.setDate(cursor.getDate() + 1);
  }
  return out;
}

export function previousMonthRange(d = new Date()): Range {
  return monthRange(new Date(d.getFullYear(), d.getMonth() - 1, 1));
}

/** Solo la parte di mese precedente confrontabile con oggi (1→giorno corrente). */
export function comparableRange(d = new Date()): Range {
  const prev = new Date(d.getFullYear(), d.getMonth() - 1, 1);
  const lastDay = new Date(prev.getFullYear(), prev.getMonth() + 1, 0).getDate();
  const end = new Date(prev.getFullYear(), prev.getMonth(), Math.min(d.getDate(), lastDay), 23, 59, 59);
  return { start: prev, end };
}

// ---------------------------------------------------------------------------
//  Riepiloghi
// ---------------------------------------------------------------------------

export interface TodaySummary {
  date: ISODate;
  tasks: Task[];
  openTasks: Task[];
  doneTasks: Task[];
  events: EventOccurrence[];
  overdue: Task[];
  urgent: Task[];
  upcoming: Task[];
  next: Task | null;
  nextEvent: EventOccurrence | null;
  todaySpend: number;
  monthSpend: number;
  prevMonthSpend: number;
}

export function todaySummary(data: Snapshot, now = new Date()): TodaySummary {
  const date = toISODate(now);
  const tasks = tasksToday(data.tasks, date);
  const overdue = tasksOverdue(data.tasks, date);
  return {
    date,
    tasks,
    openTasks: tasks.filter(isOpen),
    doneTasks: tasks.filter((t) => !isOpen(t)),
    events: eventsOn(data.events, date),
    overdue,
    urgent: data.tasks.filter((t) => isOpen(t) && t.priority === "urgent"),
    upcoming: tasksUpcoming(data.tasks, 7, date),
    next: nextTask(data.tasks, now),
    nextEvent: nextEvent(data.events, now),
    todaySpend: sum(expensesIn(data.expenses, dayRange(now))),
    monthSpend: sum(expensesIn(data.expenses, monthRange(now))),
    prevMonthSpend: sum(expensesIn(data.expenses, comparableRange(now))),
  };
}

export interface WeekSummary {
  range: Range;
  completed: Task[];
  open: Task[];
  events: EventOccurrence[];
  spend: number;
  topCategories: CategoryTotal[];
}

export function weekSummary(data: Snapshot, now = new Date()): WeekSummary {
  const range = weekRange(now);
  const inWeek = (iso: ISODate | null) => Boolean(iso && inRange(iso, range));
  const rows = expensesIn(data.expenses, range);
  return {
    range,
    completed: data.tasks.filter(
      (t) => t.status === "done" && t.completed_at && inRange(t.completed_at.slice(0, 10), range),
    ),
    open: data.tasks.filter((t) => isOpen(t) && inWeek(t.due_date)),
    events: eventsBetween(data.events, range.start, range.end),
    spend: sum(rows),
    topCategories: totalsByCategory(rows, data.categories).slice(0, 4),
  };
}

/** Frase del riepilogo giornaliero. */
export function todayHeadline(s: TodaySummary): string {
  const t = s.openTasks.length;
  const e = s.events.length;
  if (t === 0 && e === 0) return "Nessun impegno per oggi.";
  const parts: string[] = [];
  if (t > 0) parts.push(`${t} ${t === 1 ? "attività" : "attività"}`);
  if (e > 0) parts.push(`${e} ${e === 1 ? "evento" : "eventi"}`);
  return `Oggi hai ${parts.join(" e ")}.`;
}

// ---------------------------------------------------------------------------
//  Ricerca globale
// ---------------------------------------------------------------------------

export type SearchHit =
  | { kind: "task"; item: Task; score: number }
  | { kind: "event"; item: EventItem; score: number }
  | { kind: "note"; item: Note; score: number }
  | { kind: "expense"; item: Expense; score: number };

function fold(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function score(haystack: string[], needle: string): number {
  const n = fold(needle);
  let best = 0;
  for (const h of haystack) {
    const f = fold(h ?? "");
    if (!f) continue;
    if (f === n) best = Math.max(best, 3);
    else if (f.startsWith(n)) best = Math.max(best, 2.5);
    else if (new RegExp(`\\b${escapeRe(n)}`).test(f)) best = Math.max(best, 2);
    else if (f.includes(n)) best = Math.max(best, 1);
  }
  return best;
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function search(data: Snapshot, query: string, limit = 40): SearchHit[] {
  const q = query.trim();
  if (q.length < 2) return [];
  const hits: SearchHit[] = [];

  for (const item of data.tasks) {
    const s = score([item.title, item.description ?? "", item.notes ?? ""], q);
    if (s) hits.push({ kind: "task", item, score: s });
  }
  for (const item of data.events) {
    const s = score([item.title, item.description ?? "", item.location ?? ""], q);
    if (s) hits.push({ kind: "event", item, score: s });
  }
  for (const item of data.notes) {
    const s = score([item.body], q);
    if (s) hits.push({ kind: "note", item, score: s });
  }
  for (const item of data.expenses) {
    const cat = data.categories.find((c) => c.id === item.category_id)?.name ?? "";
    const s = score([item.description, item.note ?? "", cat], q);
    if (s) hits.push({ kind: "expense", item, score: s });
  }

  return hits.sort((a, b) => b.score - a.score || sortDate(b) - sortDate(a)).slice(0, limit);
}

function sortDate(h: SearchHit): number {
  switch (h.kind) {
    case "task":
      return h.item.due_date ? fromISODate(h.item.due_date).getTime() : 0;
    case "event":
      return fromISODate(h.item.start_date).getTime();
    case "expense":
      return fromISODate(h.item.spent_on).getTime();
    default:
      return new Date(h.item.created_at).getTime();
  }
}

// ---------------------------------------------------------------------------
//  Calendario: elementi di un giorno, di ogni tipo
// ---------------------------------------------------------------------------

export type DayEntry =
  | { kind: "event"; id: string; time: string | null; endTime: string | null; title: string; source: EventItem }
  | { kind: "task"; id: string; time: string | null; endTime: null; title: string; source: Task }
  | { kind: "expense"; id: string; time: null; endTime: null; title: string; source: Expense };

export function dayEntries(data: Snapshot, date: ISODate): DayEntry[] {
  const entries: DayEntry[] = [];

  for (const occ of eventsOn(data.events, date)) {
    entries.push({
      kind: "event",
      id: `${occ.event.id}:${occ.date}`,
      time: occ.event.all_day ? null : occ.event.start_time,
      endTime: occ.event.end_time,
      title: occ.event.title,
      source: occ.event,
    });
  }
  for (const t of tasksDueOn(data.tasks, date)) {
    entries.push({ kind: "task", id: t.id, time: t.due_time, endTime: null, title: t.title, source: t });
  }
  for (const e of data.expenses.filter((x) => x.spent_on === date)) {
    entries.push({
      kind: "expense",
      id: e.id,
      time: null,
      endTime: null,
      title: e.description || "Spesa",
      source: e,
    });
  }

  return entries.sort((a, b) => (a.time ?? "99:99").localeCompare(b.time ?? "99:99"));
}

/** Quali tipi cadono in un giorno — per i puntini della griglia mensile. */
export function dayMarkers(data: Snapshot, date: ISODate): ("event" | "task" | "expense")[] {
  const set = new Set<"event" | "task" | "expense">();
  for (const e of dayEntries(data, date)) set.add(e.kind);
  return [...set];
}

export { differenceInCalendarDays };

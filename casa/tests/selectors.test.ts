import assert from "node:assert/strict";
import { test } from "node:test";
import {
  dayEntries,
  search,
  sum,
  todayHeadline,
  todaySummary,
  totalsByCategory,
  totalsByPerson,
} from "../src/lib/data/selectors.ts";
import { parseAmountToCents, percentDelta } from "../src/lib/format.ts";
import type { Snapshot } from "../src/lib/data/driver.ts";
import type { Category, EventItem, Expense, Note, Task } from "../src/lib/types.ts";

const NOW = new Date(2026, 2, 14, 10, 0, 0);
const TODAY = "2026-03-14";
const HH = "h1";
const LUCA = "u1";
const MARTA = "u2";

const stamp = { created_at: "2026-03-01T08:00:00.000Z", updated_at: "2026-03-01T08:00:00.000Z" };

const cat = (id: string, slug: string, color: string): Category => ({
  id,
  household_id: HH,
  kind: "expense",
  slug,
  name: slug,
  icon: "tag",
  color,
  position: 1,
  created_by: LUCA,
  ...stamp,
});

const task = (p: Partial<Task> & { id: string; title: string }): Task => ({
  household_id: HH,
  description: null,
  notes: null,
  due_date: null,
  due_time: null,
  status: "todo",
  priority: "normal",
  category_id: null,
  assigned_to: null,
  reminders: [],
  recurrence: null,
  recurring_item_id: null,
  completed_at: null,
  completed_by: null,
  visibility: "shared",
  created_by: LUCA,
  ...stamp,
  ...p,
});

const event = (p: Partial<EventItem> & { id: string; title: string; start_date: string }): EventItem => ({
  household_id: HH,
  description: null,
  location: null,
  start_time: null,
  end_date: null,
  end_time: null,
  all_day: false,
  attendees: [],
  reminders: [],
  recurrence: null,
  recurring_item_id: null,
  color: null,
  visibility: "shared",
  created_by: LUCA,
  ...stamp,
  ...p,
});

const expense = (p: Partial<Expense> & { id: string; amount_cents: number; spent_on: string }): Expense => ({
  household_id: HH,
  currency: "EUR",
  description: "",
  category_id: null,
  payment_method: "card",
  paid_by: LUCA,
  note: null,
  visibility: "shared",
  created_by: LUCA,
  ...stamp,
  ...p,
});

const note = (p: Partial<Note> & { id: string; body: string }): Note => ({
  household_id: HH,
  pinned: false,
  archived_at: null,
  converted_kind: null,
  converted_id: null,
  visibility: "shared",
  created_by: LUCA,
  ...stamp,
  ...p,
});

const CASA = cat("c1", "casa", "#C2703C");
const CIBO = cat("c2", "alimentari", "#5E8C61");

const data: Snapshot = {
  categories: [CASA, CIBO],
  tasks: [
    task({ id: "t1", title: "Comprare latte", due_date: TODAY }),
    task({ id: "t2", title: "Chiamare assicurazione", due_date: TODAY, priority: "urgent" }),
    task({ id: "t3", title: "Fatto", due_date: TODAY, status: "done", completed_at: "2026-03-14T09:00:00.000Z" }),
    task({ id: "t4", title: "Rinnovare assicurazione auto", due_date: "2026-03-10" }),
    task({ id: "t5", title: "Senza data" }),
    task({ id: "t6", title: "Prossima settimana", due_date: "2026-03-18" }),
  ],
  events: [
    event({ id: "e1", title: "Cena con Marco", start_date: TODAY, start_time: "20:00" }),
    event({ id: "e2", title: "Dentista", start_date: "2026-03-20", start_time: "09:30" }),
  ],
  notes: [note({ id: "n1", body: "Controllare assicurazione casa" })],
  expenses: [
    expense({ id: "x1", amount_cents: 6420, spent_on: TODAY, category_id: CIBO.id, description: "Esselunga" }),
    expense({ id: "x2", amount_cents: 350, spent_on: TODAY, category_id: CIBO.id, paid_by: MARTA }),
    expense({ id: "x3", amount_cents: 85000, spent_on: "2026-03-01", category_id: CASA.id, description: "Affitto" }),
    expense({ id: "x4", amount_cents: 5000, spent_on: "2026-02-05", category_id: CASA.id }),
  ],
  notifications: [],
};

test("il riepilogo di oggi separa aperte, fatte, ritardi e urgenze", () => {
  const s = todaySummary(data, NOW);
  assert.equal(s.tasks.length, 3);
  assert.equal(s.openTasks.length, 2);
  assert.equal(s.doneTasks.length, 1);
  assert.equal(s.events.length, 1);
  assert.deepEqual(
    s.overdue.map((t) => t.id),
    ["t4"],
  );
  assert.deepEqual(
    s.urgent.map((t) => t.id),
    ["t2"],
  );
  assert.deepEqual(
    s.upcoming.map((t) => t.id),
    ["t6"],
  );
});

test("i totali di spesa rispettano i periodi", () => {
  const s = todaySummary(data, NOW);
  assert.equal(s.todaySpend, 6770);
  assert.equal(s.monthSpend, 91770);
  // il confronto usa solo 1→14 febbraio, quindi include x4
  assert.equal(s.prevMonthSpend, 5000);
});

test("la frase del riepilogo si accorda al numero", () => {
  assert.equal(todayHeadline(todaySummary(data, NOW)), "Oggi hai 2 attività e 1 evento.");
  const empty = todaySummary(
    { ...data, tasks: [], events: [] },
    NOW,
  );
  assert.equal(todayHeadline(empty), "Nessun impegno per oggi.");
});

test("totali per categoria e per persona", () => {
  const rows = data.expenses.filter((e) => e.spent_on.startsWith("2026-03"));
  const byCat = totalsByCategory(rows, data.categories);
  assert.equal(byCat[0].slug, "casa");
  assert.equal(byCat[0].total, 85000);
  assert.equal(Math.round(byCat[0].share * 100), 93);

  const byPerson = totalsByPerson(rows);
  assert.equal(byPerson[0].userId, LUCA);
  assert.equal(byPerson[0].total, 91420);
  assert.equal(byPerson[1].total, 350);
  assert.equal(sum(rows), 91770);
});

test("la ricerca globale attraversa tutti i tipi", () => {
  const hits = search(data, "assicurazione");
  const kinds = new Set(hits.map((h) => h.kind));
  assert.ok(kinds.has("task"));
  assert.ok(kinds.has("note"));
  assert.equal(hits.length, 3); // due task + una nota
});

test("la ricerca ignora accenti e maiuscole e ignora le query cortissime", () => {
  assert.ok(search(data, "ESSELUNGA").length > 0);
  assert.equal(search(data, "a").length, 0);
});

test("un giorno del calendario mette insieme eventi, task e spese", () => {
  const entries = dayEntries(data, TODAY);
  assert.deepEqual(new Set(entries.map((e) => e.kind)), new Set(["event", "task", "expense"]));
  // gli elementi con orario vengono prima
  assert.equal(entries[0].kind, "event");
});

test("utilità sugli importi", () => {
  assert.equal(parseAmountToCents("42,50"), 4250);
  assert.equal(parseAmountToCents("42.50"), 4250);
  assert.equal(parseAmountToCents("1.250,00"), 125000);
  assert.equal(parseAmountToCents("abc"), null);
  assert.equal(percentDelta(90, 100), "−10%");
  assert.equal(percentDelta(110, 100), "+10%");
  assert.equal(percentDelta(100, 100), "in linea");
  assert.equal(percentDelta(50, 0), null);
});

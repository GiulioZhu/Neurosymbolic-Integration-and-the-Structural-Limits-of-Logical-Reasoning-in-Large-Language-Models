import assert from "node:assert/strict";
import { test } from "node:test";
import { describeRecurrence, expandDates, nextOccurrence, occursOn } from "../src/lib/recurrence.ts";
import { fromISODate } from "../src/lib/date.ts";

test("giornaliera con intervallo", () => {
  const rule = { freq: "daily" as const, interval: 2 };
  assert.equal(occursOn("2026-03-01", rule, fromISODate("2026-03-03")), true);
  assert.equal(occursOn("2026-03-01", rule, fromISODate("2026-03-04")), false);
  assert.equal(occursOn("2026-03-01", rule, fromISODate("2026-02-28")), false);
});

test("settimanale su più giorni", () => {
  const rule = { freq: "weekly" as const, interval: 1, byweekday: [1, 4] }; // lun, gio
  assert.equal(occursOn("2026-03-02", rule, fromISODate("2026-03-05")), true); // giovedì
  assert.equal(occursOn("2026-03-02", rule, fromISODate("2026-03-06")), false); // venerdì
});

test("mensile il giorno 31 cade sull'ultimo giorno dei mesi corti", () => {
  const rule = { freq: "monthly" as const, interval: 1, bymonthday: 31 };
  assert.equal(occursOn("2026-01-31", rule, fromISODate("2026-04-30")), true);
  assert.equal(occursOn("2026-01-31", rule, fromISODate("2026-04-29")), false);
});

test("l'affitto del giorno 1 salta al mese successivo", () => {
  const rule = { freq: "monthly" as const, interval: 1, bymonthday: 1 };
  assert.equal(nextOccurrence("2026-03-01", rule, "2026-03-01"), "2026-04-01");
});

test("until interrompe la serie", () => {
  const rule = { freq: "daily" as const, interval: 1, until: "2026-03-05" };
  assert.equal(nextOccurrence("2026-03-01", rule, "2026-03-04"), "2026-03-05");
  assert.equal(nextOccurrence("2026-03-01", rule, "2026-03-05"), null);
});

test("espansione in un intervallo", () => {
  const dates = expandDates(
    "2026-03-02",
    { freq: "weekly", interval: 1, byweekday: [1] },
    fromISODate("2026-03-01"),
    fromISODate("2026-03-31"),
  );
  assert.deepEqual(dates, ["2026-03-02", "2026-03-09", "2026-03-16", "2026-03-23", "2026-03-30"]);
});

test("senza regola l'elemento compare una volta sola e solo se nell'intervallo", () => {
  assert.deepEqual(
    expandDates("2026-03-10", null, fromISODate("2026-03-01"), fromISODate("2026-03-31")),
    ["2026-03-10"],
  );
  assert.deepEqual(
    expandDates("2026-04-10", null, fromISODate("2026-03-01"), fromISODate("2026-03-31")),
    [],
  );
});

test("descrizioni leggibili", () => {
  assert.equal(describeRecurrence(null), "Mai");
  assert.equal(describeRecurrence({ freq: "daily", interval: 1 }), "Ogni giorno");
  assert.equal(describeRecurrence({ freq: "daily", interval: 3 }), "Ogni 3 giorni");
  assert.equal(
    describeRecurrence({ freq: "weekly", interval: 1, byweekday: [1, 4] }),
    "Ogni lunedì e giovedì",
  );
  assert.equal(describeRecurrence({ freq: "monthly", interval: 1, bymonthday: 1 }), "Ogni mese il 1");
});

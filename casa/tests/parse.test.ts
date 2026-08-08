import assert from "node:assert/strict";
import { test } from "node:test";
import { parseCapture } from "../src/lib/parse.ts";

/** Riferimento fisso: sabato 14 marzo 2026, ore 10:00. */
const NOW = new Date(2026, 2, 14, 10, 0, 0);

test("Comprare latte domani → task con scadenza domani", () => {
  const r = parseCapture("Comprare latte domani", NOW);
  assert.equal(r.kind, "task");
  assert.equal(r.title, "Comprare latte");
  assert.equal(r.date, "2026-03-15");
  assert.equal(r.time, null);
});

test("Chiamare commercialista venerdì → task al prossimo venerdì", () => {
  const r = parseCapture("Chiamare commercialista venerdì", NOW);
  assert.equal(r.kind, "task");
  assert.equal(r.title, "Chiamare commercialista");
  assert.equal(r.date, "2026-03-20");
});

test("Cena con Marco sabato alle 20 → evento con orario e fine a +1h", () => {
  const r = parseCapture("Cena con Marco sabato alle 20", NOW);
  assert.equal(r.kind, "event");
  assert.equal(r.title, "Cena con Marco");
  assert.equal(r.date, "2026-03-14");
  assert.equal(r.time, "20:00");
  assert.equal(r.endTime, "21:00");
});

test("Pagato 35€ benzina → spesa nella categoria trasporti", () => {
  const r = parseCapture("Pagato 35€ benzina", NOW);
  assert.equal(r.kind, "expense");
  assert.equal(r.amountCents, 3500);
  assert.equal(r.categorySlug, "trasporti");
  assert.equal(r.date, "2026-03-14");
  assert.equal(r.title, "Benzina");
});

test("45 euro cena → spesa nella categoria ristoranti", () => {
  const r = parseCapture("45 euro cena", NOW);
  assert.equal(r.kind, "expense");
  assert.equal(r.amountCents, 4500);
  assert.equal(r.categorySlug, "ristoranti");
  assert.equal(r.title, "Cena");
});

test("importi con decimali e separatore italiano", () => {
  assert.equal(parseCapture("speso 12,50 al bar", NOW).amountCents, 1250);
  assert.equal(parseCapture("€ 1.250,00 affitto", NOW).amountCents, 125000);
});

test("Domani alle 17 ricordami di comprare un regalo per Luca", () => {
  const r = parseCapture("Domani alle 17 ricordami di comprare un regalo per Luca", NOW);
  assert.equal(r.kind, "task");
  assert.equal(r.date, "2026-03-15");
  assert.equal(r.time, "17:00");
  assert.deepEqual(r.reminders, [0]);
  assert.match(r.title, /regalo per Luca/i);
});

test("Pagare affitto ogni mese il 1 → task mensile", () => {
  const r = parseCapture("Pagare affitto ogni mese il 1", NOW);
  assert.equal(r.kind, "task");
  assert.equal(r.recurrence?.freq, "monthly");
  assert.equal(r.recurrence?.bymonthday, 1);
  assert.equal(r.date, "2026-04-01");
  assert.equal(r.title, "Pagare affitto");
});

test("ogni lunedì → ricorrenza settimanale sul giorno giusto", () => {
  const r = parseCapture("Portare fuori la spazzatura ogni lunedì", NOW);
  assert.equal(r.recurrence?.freq, "weekly");
  assert.deepEqual(r.recurrence?.byweekday, [1]);
});

test("le ore 1–7 senza indizi si leggono come pomeriggio", () => {
  assert.equal(parseCapture("Appuntamento alle 5", NOW).time, "17:00");
  assert.equal(parseCapture("Sveglia alle 7 di mattina", NOW).time, "07:00");
  assert.equal(parseCapture("Riunione alle 11", NOW).time, "11:00");
});

test("intervallo orario", () => {
  const r = parseCapture("Corso di yoga dalle 18 alle 19:30 martedì", NOW);
  assert.equal(r.time, "18:00");
  assert.equal(r.endTime, "19:30");
});

test("priorità e urgenza", () => {
  assert.equal(parseCapture("Rinnovare assicurazione urgente", NOW).priority, "urgent");
  assert.equal(parseCapture("Chiamare idraulico importante", NOW).priority, "high");
});

test("data numerica e mese scritto", () => {
  assert.equal(parseCapture("Visita il 12/05", NOW).date, "2026-05-12");
  assert.equal(parseCapture("Scadenza 3 aprile", NOW).date, "2026-04-03");
  // una data già passata quest'anno si intende l'anno prossimo
  assert.equal(parseCapture("Compleanno 2 gennaio", NOW).date, "2027-01-02");
});

test("tra N giorni", () => {
  assert.equal(parseCapture("Richiamare tra 3 giorni", NOW).date, "2026-03-17");
  assert.equal(parseCapture("Controllare fra una settimana", NOW).date, "2026-03-21");
});

test("testo senza appigli resta una nota", () => {
  const r = parseCapture("Nuova idea per il negozio", NOW);
  assert.equal(r.kind, "note");
  assert.equal(r.date, null);
});

test("il titolo non perde parole quando 'da' non è un luogo", () => {
  const r = parseCapture("Comprare da Amazon il caricabatterie", NOW);
  assert.match(r.title, /Amazon/);
});

test("un luogo dopo 'da' in contesto di cena diventa location", () => {
  const r = parseCapture("Cena da Vito sabato alle 21", NOW);
  assert.equal(r.location, "Vito");
  assert.equal(r.title, "Cena");
});

test("nessun crash su input degeneri", () => {
  for (const s of ["", "   ", "?", "€", "12", "ogni", "alle"]) {
    assert.doesNotThrow(() => parseCapture(s, NOW));
  }
});

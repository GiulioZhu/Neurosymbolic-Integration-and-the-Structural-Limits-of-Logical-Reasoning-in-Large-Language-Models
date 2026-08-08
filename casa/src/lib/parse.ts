/**
 * Parser italiano per il Quick Capture.
 *
 * Trasforma "Cena con Marco sabato alle 20" in un oggetto strutturato.
 * Funziona interamente offline e in modo deterministico: è il comportamento
 * predefinito dell'app. Quando è configurata una chiave Anthropic,
 * `/api/parse` prova prima con il modello e ricade qui in caso di problemi,
 * così il campo non smette mai di funzionare.
 *
 * Tecnica: il testo viene "ripiegato" (minuscole + accenti rimossi 1:1, quindi
 * gli indici restano allineati all'originale). Ogni pattern riconosciuto marca
 * il proprio intervallo in una maschera; il titolo è ciò che resta.
 */

import { addDays, fromISODate, toISODate } from "./date";
import type { ISODate, ISOTime, Priority, Recurrence, SourceKind } from "./types";

export interface ParseResult {
  kind: SourceKind;
  title: string;
  date: ISODate | null;
  time: ISOTime | null;
  endTime: ISOTime | null;
  amountCents: number | null;
  categorySlug: string | null;
  priority: Priority;
  reminders: number[];
  recurrence: Recurrence | null;
  location: string | null;
  /** 0–1: quanto il parser è sicuro del *tipo*. Sotto 0.5 la UI lo dice. */
  confidence: number;
  raw: string;
}

// ---------------------------------------------------------------------------
//  Normalizzazione che preserva gli indici
// ---------------------------------------------------------------------------

const ACCENTS: Record<string, string> = {
  à: "a", á: "a", â: "a", ä: "a",
  è: "e", é: "e", ê: "e", ë: "e",
  ì: "i", í: "i", î: "i", ï: "i",
  ò: "o", ó: "o", ô: "o", ö: "o",
  ù: "u", ú: "u", û: "u", ü: "u",
  ç: "c", ñ: "n", "’": "'", "‘": "'",
};

function fold(input: string): string {
  return input
    .toLowerCase()
    .split("")
    .map((c) => ACCENTS[c] ?? c)
    .join("");
}

class Mask {
  private readonly taken: boolean[];
  private readonly length: number;

  constructor(length: number) {
    this.length = length;
    this.taken = new Array(length).fill(false);
  }

  take(start: number, end: number) {
    for (let i = Math.max(0, start); i < Math.min(this.length, end); i++) this.taken[i] = true;
  }
  isFree(start: number, end: number): boolean {
    for (let i = start; i < end; i++) if (this.taken[i]) return false;
    return true;
  }
  remainder(original: string): string {
    let out = "";
    for (let i = 0; i < original.length; i++) out += this.taken[i] ? " " : original[i];
    return out;
  }
}

/** Applica il primo match libero di `re` e ne marca l'intervallo. */
function consume(
  folded: string,
  mask: Mask,
  re: RegExp,
): RegExpExecArray | null {
  const rx = new RegExp(re.source, re.flags.includes("g") ? re.flags : re.flags + "g");
  let m: RegExpExecArray | null;
  while ((m = rx.exec(folded)) !== null) {
    if (mask.isFree(m.index, m.index + m[0].length)) {
      mask.take(m.index, m.index + m[0].length);
      return m;
    }
    if (m.index === rx.lastIndex) rx.lastIndex++;
  }
  return null;
}

// ---------------------------------------------------------------------------
//  Vocabolari
// ---------------------------------------------------------------------------

const WEEKDAYS: Record<string, number> = {
  domenica: 0, lunedi: 1, martedi: 2, mercoledi: 3,
  giovedi: 4, venerdi: 5, sabato: 6,
};

const MONTHS: Record<string, number> = {
  gennaio: 0, febbraio: 1, marzo: 2, aprile: 3, maggio: 4, giugno: 5,
  luglio: 6, agosto: 7, settembre: 8, ottobre: 9, novembre: 10, dicembre: 11,
  gen: 0, feb: 1, mar: 2, apr: 3, mag: 4, giu: 5,
  lug: 6, ago: 7, set: 8, ott: 9, nov: 10, dic: 11,
};

/** Parole che spostano una spesa in una categoria. Ordine = priorità. */
const CATEGORY_HINTS: [slug: string, words: string[]][] = [
  ["trasporti", ["benzina", "gasolio", "diesel", "carburante", "rifornimento", "treno", "taxi", "autobus", "metro", "pedaggio", "autostrada", "parcheggio", "uber", "meccanico", "gomme", "revisione", "bollo", "monopattino"]],
  ["bollette", ["bolletta", "bollette", "luce", "gas", "internet", "adsl", "fibra", "telefono", "enel", "tim", "vodafone", "fastweb", "iliad", "tari", "rifiuti", "canone"]],
  ["alimentari", ["spesa", "supermercato", "esselunga", "coop", "conad", "lidl", "carrefour", "eurospin", "penny", "alimentari", "latte", "pane", "frutta", "verdura", "macellaio", "panetteria", "fruttivendolo"]],
  ["ristoranti", ["ristorante", "pizzeria", "pizza", "cena", "pranzo", "colazione", "aperitivo", "bar", "caffe", "sushi", "trattoria", "osteria", "hamburger", "gelato", "brunch"]],
  ["salute", ["farmacia", "medicine", "medicinali", "dottore", "medico", "dentista", "analisi", "occhiali", "fisioterapia", "ticket", "visita medica"]],
  ["viaggi", ["hotel", "albergo", "volo", "aereo", "vacanza", "vacanze", "viaggio", "airbnb", "b&b", "soggiorno", "traghetto", "crociera"]],
  ["casa", ["affitto", "mutuo", "condominio", "idraulico", "elettricista", "mobili", "ikea", "detersivi", "pulizie", "arredamento", "giardino", "traslo"]],
  ["shopping", ["vestiti", "scarpe", "abbigliamento", "zara", "amazon", "shopping", "giacca", "maglione", "pantaloni", "borsa", "profumo"]],
  ["tempolibero", ["cinema", "teatro", "concerto", "palestra", "netflix", "spotify", "libro", "libri", "museo", "abbonamento", "mostra", "partita"]],
  ["regali", ["regalo", "regali", "regalino", "bomboniera"]],
];

const EXPENSE_TRIGGERS = [
  "pagato", "pagata", "speso", "spesa di", "spesi", "costa", "costato",
  "comprato", "presa", "preso", "euro", "eur", "scontrino", "fattura", "bonifico",
];

const TASK_VERBS = [
  "comprare", "compra", "chiamare", "chiama", "pagare", "paga", "prenotare", "prenota",
  "mandare", "manda", "inviare", "invia", "scrivere", "scrivi", "portare", "porta",
  "ritirare", "ritira", "controllare", "controlla", "fissare", "fissa", "rinnovare",
  "rinnova", "spedire", "spedisci", "ordinare", "ordina", "finire", "finisci",
  "preparare", "prepara", "cercare", "cerca", "lavare", "pulire", "stirare",
  "annaffiare", "buttare", "verificare", "confermare", "disdire", "restituire",
];

const EVENT_WORDS = [
  "cena", "pranzo", "aperitivo", "colazione", "brunch", "riunione", "meeting",
  "appuntamento", "visita", "compleanno", "anniversario", "matrimonio", "battesimo",
  "concerto", "partita", "cinema", "teatro", "spettacolo", "volo", "treno per",
  "call", "videochiamata", "corso", "lezione", "allenamento", "festa", "vacanza",
];

const EVENING_WORDS = ["cena", "sera", "stasera", "aperitivo", "concerto", "spettacolo", "teatro", "notte"];
const MORNING_WORDS = ["mattina", "mattino", "colazione", "stamattina", "stamane"];

// ---------------------------------------------------------------------------
//  Parser
// ---------------------------------------------------------------------------

export function parseCapture(input: string, now = new Date()): ParseResult {
  const raw = input.trim();
  const folded = fold(raw);
  const mask = new Mask(raw.length);

  const result: ParseResult = {
    kind: "note",
    title: "",
    date: null,
    time: null,
    endTime: null,
    amountCents: null,
    categorySlug: null,
    priority: "normal",
    reminders: [],
    recurrence: null,
    location: null,
    confidence: 0.4,
    raw,
  };

  if (!raw) return result;

  const has = (w: string) => folded.includes(w);
  const hasAny = (ws: string[]) => ws.some((w) => folded.includes(w));

  // --- 1. ricorrenza ------------------------------------------------------
  result.recurrence = extractRecurrence(folded, mask, now);

  // --- 2. data ------------------------------------------------------------
  result.date = extractDate(folded, mask, now);

  // --- 3. orario (prima l'intervallo, poi il singolo) ---------------------
  const eveningHint = hasAny(EVENING_WORDS);
  const morningHint = hasAny(MORNING_WORDS);
  const span = consume(folded, mask, /\b(?:dalle|da)\s+(\d{1,2})(?:[:.](\d{2}))?\s+(?:alle|a)\s+(\d{1,2})(?:[:.](\d{2}))?\b/);
  if (span) {
    result.time = toTime(Number(span[1]), Number(span[2] ?? 0), eveningHint, morningHint);
    result.endTime = toTime(Number(span[3]), Number(span[4] ?? 0), eveningHint, morningHint);
  } else {
    result.time = extractTime(folded, mask, eveningHint, morningHint);
  }

  // --- 4. importo ---------------------------------------------------------
  const expenseTrigger = hasAny(EXPENSE_TRIGGERS);
  result.amountCents = extractAmount(folded, mask, expenseTrigger);

  // --- 5. priorità --------------------------------------------------------
  if (has("urgente") || has("urgentissim") || folded.includes("!!!")) {
    result.priority = "urgent";
    consume(folded, mask, /\burgentissimo?\b|\burgente\b|!!!/);
  } else if (has("importante") || folded.includes("!!")) {
    result.priority = "high";
    consume(folded, mask, /\bimportante\b|!!/);
  }

  // --- 6. promemoria ------------------------------------------------------
  const explicitReminder = consume(folded, mask, /\b(?:avvisami|ricordamelo)\s+(\d{1,3})\s*(minuti?|min|ore?|giorni?)\s+prima\b/);
  if (explicitReminder) {
    const n = Number(explicitReminder[1]);
    const unit = explicitReminder[2];
    const minutes = unit.startsWith("or") ? n * 60 : unit.startsWith("giorn") ? n * 1440 : n;
    result.reminders = [minutes];
  }
  const wantsReminder = /\b(?:ricordami|ricordati|promemoria|avvisami)\b/.test(folded);
  consume(folded, mask, /\b(?:ricordami|ricordati)\s+(?:di\s+|che\s+devo\s+)?/);
  consume(folded, mask, /\b(?:promemoria|avvisami)\b:?\s*/);
  if (wantsReminder && result.reminders.length === 0) result.reminders = [0];

  // --- 7. luogo -----------------------------------------------------------
  // Solo in contesti dove "da X" significa davvero un posto: altrimenti
  // "Comprare da Amazon" perderebbe metà del titolo.
  if (/\b(?:cena|pranzo|aperitivo|colazione|appuntamento|visita|festa)\b/.test(folded)) {
    const place = consume(folded, mask, /\b(?:da|al|alla|presso|@)\s+([a-z][\w'’]{2,})\b(?!\s*\d)/);
    if (place) {
      result.location = capitalize(
        raw.slice(place.index, place.index + place[0].length).replace(/^\s*(?:da|al|alla|presso|@)\s+/i, ""),
      );
    }
  }

  // --- 8. categoria -------------------------------------------------------
  result.categorySlug = detectCategory(folded);

  // --- 9. tipo ------------------------------------------------------------
  const { kind, confidence } = detectKind(folded, result, expenseTrigger);
  result.kind = kind;
  result.confidence = confidence;

  // --- 10. titolo ---------------------------------------------------------
  result.title = cleanTitle(mask.remainder(raw)) || fallbackTitle(result, raw);

  // --- 11. coerenze finali ------------------------------------------------
  const today = toISODate(now);

  if (result.kind === "expense") {
    result.date ??= today;
    result.categorySlug ??= "altro";
    result.recurrence = null;
    result.reminders = [];
    // la descrizione di una spesa è la cosa comprata, non il verbo:
    // "Pagato benzina" → "Benzina"
    result.title = capitalize(
      result.title
        .replace(
          /^(?:pagat[oaie]|spes[oaie]|comprat[oaie]|pres[oaie]|costat[oaie]|costa)\b\s*(?:per\s+|di\s+|il\s+|lo\s+|la\s+|un[oa]?\s+)?/i,
          "",
        )
        .trim() || result.title,
    );
  }
  if (result.kind === "event") {
    result.date ??= today;
    if (result.time && !result.endTime) result.endTime = shiftTime(result.time, 60);
  }
  if (result.kind === "task" && result.recurrence && !result.date) {
    result.date = nextDateForRule(result.recurrence, now);
  }
  if (result.kind === "task" && result.time && !result.date) {
    // "sveglia alle 7": se le 7 sono già passate si intende domani
    const [h, m] = result.time.split(":").map(Number);
    const at = new Date(now);
    at.setHours(h, m, 0, 0);
    result.date = at.getTime() >= now.getTime() ? today : toISODate(addDays(fromISODate(today), 1));
  }
  if (result.kind === "note") {
    result.date = null;
    result.time = null;
    result.recurrence = null;
  }
  if (result.reminders.length && !result.date) result.reminders = [];

  return result;
}

// ---------------------------------------------------------------------------
//  Estrattori
// ---------------------------------------------------------------------------

function extractRecurrence(folded: string, mask: Mask, now: Date): Recurrence | null {
  // "ogni mese il 1", "il 1 di ogni mese", "ogni giorno 1 del mese"
  const monthly = consume(folded, mask, /\b(?:ogni\s+(?:mese|giorno)\s+(?:il\s+)?(\d{1,2})(?:\s+del\s+mese)?|il\s+(\d{1,2})\s+di\s+ogni\s+mese|ogni\s+(\d{1,2})\s+del\s+mese)\b/);
  if (monthly) {
    const day = Number(monthly[1] ?? monthly[2] ?? monthly[3]);
    if (day >= 1 && day <= 31) return { freq: "monthly", interval: 1, bymonthday: day };
  }

  const everyN = consume(folded, mask, /\bogni\s+(\d{1,2})\s+(giorni|settimane|mesi)\b/);
  if (everyN) {
    const n = Number(everyN[1]);
    const unit = everyN[2];
    if (unit === "giorni") return { freq: "daily", interval: n };
    if (unit === "settimane") return { freq: "weekly", interval: n, byweekday: [now.getDay()] };
    return { freq: "monthly", interval: n, bymonthday: now.getDate() };
  }

  const weekdayNames = Object.keys(WEEKDAYS).join("|");
  const everyWeekday = consume(folded, mask, new RegExp(`\\b(?:ogni|tutti i|tutte le)\\s+(${weekdayNames})\\b`));
  if (everyWeekday) {
    return { freq: "weekly", interval: 1, byweekday: [WEEKDAYS[everyWeekday[1]]] };
  }

  const simple = consume(folded, mask, /\b(?:ogni\s+giorno|tutti\s+i\s+giorni|quotidianamente|ogni\s+settimana|settimanalmente|ogni\s+mese|mensilmente|ogni\s+anno|annualmente)\b/);
  if (simple) {
    const s = simple[0];
    if (s.includes("giorn") || s.includes("quotidian")) return { freq: "daily", interval: 1 };
    if (s.includes("settiman")) return { freq: "weekly", interval: 1, byweekday: [now.getDay()] };
    if (s.includes("anno") || s.includes("annual")) return { freq: "monthly", interval: 12, bymonthday: now.getDate() };
    return { freq: "monthly", interval: 1, bymonthday: now.getDate() };
  }

  return null;
}

function extractDate(folded: string, mask: Mask, now: Date): ISODate | null {
  const today = new Date(now);
  today.setHours(0, 0, 0, 0);

  // oggi / domani / dopodomani / ieri
  const rel = consume(folded, mask, /\b(oggi|stasera|stamattina|stamane|stanotte|domani|dopodomani|ieri)\b/);
  if (rel) {
    const w = rel[1];
    if (w === "domani") return toISODate(addDays(today, 1));
    if (w === "dopodomani") return toISODate(addDays(today, 2));
    if (w === "ieri") return toISODate(addDays(today, -1));
    return toISODate(today);
  }

  // tra/fra N giorni|settimane|mesi
  const inN = consume(folded, mask, /\b(?:tra|fra)\s+(un[oa]?|\d{1,3})\s+(giorni?|settiman[ae]|mes[ei])\b/);
  if (inN) {
    const n = /^un/.test(inN[1]) ? 1 : Number(inN[1]);
    const unit = inN[2];
    if (unit.startsWith("giorn")) return toISODate(addDays(today, n));
    if (unit.startsWith("settiman")) return toISODate(addDays(today, n * 7));
    const d = new Date(today);
    d.setMonth(d.getMonth() + n);
    return toISODate(d);
  }

  // data numerica: 12/03, 12-03-2027
  const numeric = consume(folded, mask, /\b(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{2,4}))?\b/);
  if (numeric) {
    const day = Number(numeric[1]);
    const month = Number(numeric[2]) - 1;
    let year = numeric[3] ? Number(numeric[3]) : today.getFullYear();
    if (year < 100) year += 2000;
    if (day >= 1 && day <= 31 && month >= 0 && month <= 11) {
      const d = new Date(year, month, day);
      if (!numeric[3] && d < today) d.setFullYear(year + 1);
      return toISODate(d);
    }
  }

  // "12 marzo", "1 gen 2027", "il 3 aprile"
  const monthNames = Object.keys(MONTHS).join("|");
  const named = consume(folded, mask, new RegExp(`\\b(?:il\\s+|l'\\s*)?(\\d{1,2})\\s+(${monthNames})\\b(?:\\s+(\\d{4}))?`));
  if (named) {
    const day = Number(named[1]);
    const month = MONTHS[named[2]];
    const year = named[3] ? Number(named[3]) : today.getFullYear();
    const d = new Date(year, month, day);
    if (!named[3] && d < today) d.setFullYear(year + 1);
    return toISODate(d);
  }

  // giorno della settimana, con eventuale "prossimo"
  const weekdayNames = Object.keys(WEEKDAYS).join("|");
  const wd = consume(folded, mask, new RegExp(`\\b(?:di\\s+|il\\s+)?(${weekdayNames})(\\s+prossim[oa])?\\b`));
  if (wd) {
    const target = WEEKDAYS[wd[1]];
    const next = wd[2] !== undefined;
    let delta = (target - today.getDay() + 7) % 7;
    if (next && delta < 7) delta += 7;
    return toISODate(addDays(today, delta));
  }

  // "il 15" (giorno del mese corrente o del prossimo)
  const dayOnly = consume(folded, mask, /\b(?:il|per\s+il|entro\s+il)\s+(\d{1,2})\b(?!\s*[:.]\d)(?!\s*(?:euro|eur|€))/);
  if (dayOnly) {
    const day = Number(dayOnly[1]);
    if (day >= 1 && day <= 31) {
      const d = new Date(today.getFullYear(), today.getMonth(), day);
      if (d < today) d.setMonth(d.getMonth() + 1);
      return toISODate(d);
    }
  }

  // "settimana prossima", "il mese prossimo"
  const nextPeriod = consume(folded, mask, /\b(?:la\s+)?(?:settimana|mese)\s+prossim[oa]\b/);
  if (nextPeriod) {
    if (nextPeriod[0].includes("settimana")) return toISODate(addDays(today, 7));
    const d = new Date(today);
    d.setMonth(d.getMonth() + 1);
    return toISODate(d);
  }

  return null;
}

function extractTime(
  folded: string,
  mask: Mask,
  evening: boolean,
  morning: boolean,
): ISOTime | null {
  const word = consume(folded, mask, /\b(mezzogiorno|mezzanotte)\b/);
  if (word) return word[1] === "mezzogiorno" ? "12:00" : "00:00";

  const explicit = consume(folded, mask, /\b(?:alle|all'|ore|h\.?)\s*(\d{1,2})(?:[:.](\d{2}))?\b/);
  if (explicit) return toTime(Number(explicit[1]), Number(explicit[2] ?? 0), evening, morning);

  const bare = consume(folded, mask, /\b(\d{1,2}):(\d{2})\b/);
  if (bare) return toTime(Number(bare[1]), Number(bare[2]), false, false);

  return null;
}

function toTime(hour: number, minute: number, evening: boolean, morning: boolean): ISOTime {
  let h = hour;
  if (h > 23) h = 23;
  // 1–7 senza indizio di mattina si leggono come pomeriggio/sera:
  // "alle 5" in italiano quotidiano è quasi sempre le 17.
  if (h >= 1 && h <= 7 && !morning) h += 12;
  // 8 e 9 diventano serali solo se il testo parla di sera/cena
  else if ((h === 8 || h === 9) && evening) h += 12;
  const m = Math.min(59, Math.max(0, minute));
  return `${String(h % 24).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function shiftTime(t: ISOTime, minutes: number): ISOTime {
  const [h, m] = t.split(":").map(Number);
  const total = (h * 60 + m + minutes) % 1440;
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function extractAmount(folded: string, mask: Mask, expenseTrigger: boolean): number | null {
  // con simbolo o parola valuta: "35€", "€ 35", "45 euro", "12,50 eur"
  const withCurrency = consume(
    folded,
    mask,
    /(?:€\s*(\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{1,2})?)|(\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{1,2})?)\s*(?:€|euro|eur\b))/,
  );
  if (withCurrency) {
    return toCents(withCurrency[1] ?? withCurrency[2]);
  }

  // numero nudo, ammesso solo se il testo dichiara una spesa
  if (expenseTrigger) {
    const bare = consume(folded, mask, /\b(\d{1,4}(?:[.,]\d{1,2})?)\b/);
    if (bare) return toCents(bare[1]);
  }

  return null;
}

function toCents(s: string): number | null {
  const normalized = s.replace(/[\s.](?=\d{3}\b)/g, "").replace(",", ".");
  const n = Number(normalized);
  if (!Number.isFinite(n)) return null;
  return Math.round(n * 100);
}

function detectCategory(folded: string): string | null {
  for (const [slug, words] of CATEGORY_HINTS) {
    if (words.some((w) => folded.includes(w))) return slug;
  }
  return null;
}

function detectKind(
  folded: string,
  partial: ParseResult,
  expenseTrigger: boolean,
): { kind: SourceKind; confidence: number } {
  const hasAmount = partial.amountCents !== null;
  const hasTime = partial.time !== null;
  const hasDate = partial.date !== null;

  if (hasAmount) {
    // un importo insieme a "pagare"/"prenotare" al futuro resta una task
    const futureIntent = /\b(?:pagare|prenotare|comprare|ordinare)\b/.test(folded);
    if (!futureIntent || expenseTrigger) return { kind: "expense", confidence: 0.95 };
    return { kind: "task", confidence: 0.6 };
  }

  const isTaskVerb = TASK_VERBS.some((v) => new RegExp(`\\b${v}\\b`).test(folded));
  const isEventWord = EVENT_WORDS.some((w) => folded.includes(w));

  if (isEventWord && (hasTime || /\bcon\s+\w/.test(folded)) && !isTaskVerb) {
    return { kind: "event", confidence: hasTime ? 0.9 : 0.7 };
  }
  if (isTaskVerb) return { kind: "task", confidence: hasDate ? 0.9 : 0.75 };
  if (isEventWord && hasDate) return { kind: "event", confidence: 0.65 };
  if (hasTime && hasDate) return { kind: "event", confidence: 0.55 };
  if (hasDate) return { kind: "task", confidence: 0.6 };
  // un orario da solo ("sveglia alle 7") è comunque qualcosa da fare
  if (hasTime) return { kind: "task", confidence: 0.55 };
  if (partial.recurrence) return { kind: "task", confidence: 0.6 };

  return { kind: "note", confidence: 0.5 };
}

// ---------------------------------------------------------------------------
//  Titolo
// ---------------------------------------------------------------------------

const EDGE_FILLERS = new Set([
  "di", "a", "ad", "al", "allo", "alla", "alle", "agli", "il", "lo", "la", "i",
  "gli", "le", "per", "con", "da", "in", "e", "un", "una", "uno", "del", "dello",
  "della", "dei", "degli", "delle", "che", "devo", "dobbiamo", "bisogna", "poi",
]);

function cleanTitle(text: string): string {
  let words = text
    .replace(/[·•]/g, " ")
    .split(/\s+/)
    .map((w) => w.trim())
    .filter(Boolean);

  const strip = () => {
    while (words.length && EDGE_FILLERS.has(fold(words[0]).replace(/[^\p{L}']/gu, ""))) words.shift();
    while (words.length && EDGE_FILLERS.has(fold(words[words.length - 1]).replace(/[^\p{L}']/gu, ""))) words.pop();
  };
  strip();

  let out = words.join(" ").replace(/\s+([,.;:!?])/g, "$1").replace(/[,;:]+$/, "").trim();
  // "comprare" → "Comprare"
  out = capitalize(out);
  return out;
}

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

function fallbackTitle(r: ParseResult, raw: string): string {
  if (r.kind === "expense") return r.categorySlug ? "Spesa" : "Spesa";
  return capitalize(raw.slice(0, 60));
}

function nextDateForRule(rule: Recurrence, now: Date): ISODate {
  const today = new Date(now);
  today.setHours(0, 0, 0, 0);
  if (rule.freq === "monthly" && rule.bymonthday) {
    const d = new Date(today.getFullYear(), today.getMonth(), rule.bymonthday);
    if (d < today) d.setMonth(d.getMonth() + 1);
    return toISODate(d);
  }
  if (rule.freq === "weekly" && rule.byweekday?.length) {
    const delta = (rule.byweekday[0] - today.getDay() + 7) % 7;
    return toISODate(addDays(today, delta));
  }
  return toISODate(today);
}

/** Ricostruisce un `Date` completo dalla coppia data+ora del risultato. */
export function parseResultStart(r: ParseResult): Date | null {
  if (!r.date) return null;
  const d = fromISODate(r.date);
  if (r.time) {
    const [h, m] = r.time.split(":").map(Number);
    d.setHours(h, m, 0, 0);
  }
  return d;
}

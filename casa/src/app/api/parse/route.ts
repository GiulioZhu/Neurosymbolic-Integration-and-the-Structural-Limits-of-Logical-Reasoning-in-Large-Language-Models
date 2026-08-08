import { NextResponse } from "next/server";
import { parseCapture, type ParseResult } from "@/lib/parse";
import { todayISO } from "@/lib/date";

export const runtime = "nodejs";

/**
 * Interpretazione del Quick Capture.
 *
 * Il parser a regole è sempre la base: è immediato e non costa nulla. Se è
 * configurata `ANTHROPIC_API_KEY`, si chiede anche al modello — utile per le
 * frasi contorte ("dopo la riunione di martedì ricordami di...") — ma il
 * risultato viene validato campo per campo e, al minimo dubbio, si tiene
 * quello locale. Il campo non deve mai smettere di funzionare.
 */
export async function POST(request: Request) {
  let text = "";
  try {
    const body = (await request.json()) as { text?: unknown };
    text = typeof body.text === "string" ? body.text.slice(0, 500) : "";
  } catch {
    return NextResponse.json({ error: "Corpo non valido" }, { status: 400 });
  }

  if (!text.trim()) return NextResponse.json({ error: "Testo mancante" }, { status: 400 });

  const local = parseCapture(text);
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return NextResponse.json({ result: local, source: "rules" });

  try {
    const ai = await askModel(text, apiKey);
    if (ai) return NextResponse.json({ result: merge(local, ai), source: "ai" });
  } catch {
    /* silenzioso: il risultato locale è già buono */
  }

  return NextResponse.json({ result: local, source: "rules" });
}

// ---------------------------------------------------------------------------

interface ModelOutput {
  kind?: string;
  title?: string;
  date?: string | null;
  time?: string | null;
  endTime?: string | null;
  amountCents?: number | null;
  categorySlug?: string | null;
  priority?: string;
  reminders?: number[];
  location?: string | null;
}

const CATEGORIES = [
  "casa",
  "alimentari",
  "ristoranti",
  "trasporti",
  "shopping",
  "viaggi",
  "salute",
  "bollette",
  "regali",
  "tempolibero",
  "altro",
];

async function askModel(text: string, apiKey: string): Promise<ModelOutput | null> {
  const today = todayISO();
  const weekday = new Intl.DateTimeFormat("it-IT", { weekday: "long" }).format(new Date());

  const system = [
    "Sei l'analizzatore di una app italiana per la gestione familiare.",
    `Oggi è ${weekday} ${today} (fuso Europe/Rome).`,
    "Ricevi una frase scritta di getta e restituisci SOLO un oggetto JSON, senza testo attorno.",
    "Campi:",
    '  kind: "task" | "event" | "expense" | "note"',
    "  title: stringa breve, senza la parte di data/ora/importo",
    '  date: "YYYY-MM-DD" o null',
    '  time: "HH:MM" o null',
    '  endTime: "HH:MM" o null',
    "  amountCents: intero in centesimi o null",
    `  categorySlug: uno fra ${CATEGORIES.join(", ")} oppure null`,
    '  priority: "low" | "normal" | "high" | "urgent"',
    "  reminders: array di minuti PRIMA della scadenza (0 = all'orario)",
    "  location: stringa o null",
    "Regole: una spesa già sostenuta è expense; qualcosa da fare è task;",
    "un impegno con orario e persone è event; se non c'è nulla di azionabile è note.",
  ].join("\n");

  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 400,
      system,
      messages: [{ role: "user", content: text }],
    }),
    signal: AbortSignal.timeout(6000),
  });

  if (!res.ok) return null;
  const json = (await res.json()) as { content?: { type: string; text?: string }[] };
  const raw = json.content?.find((c) => c.type === "text")?.text ?? "";
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) return null;

  try {
    return JSON.parse(match[0]) as ModelOutput;
  } catch {
    return null;
  }
}

/** Accetta dal modello solo i valori ben formati; per il resto tiene il locale. */
function merge(local: ParseResult, ai: ModelOutput): ParseResult {
  const out: ParseResult = { ...local };

  if (ai.kind && ["task", "event", "expense", "note"].includes(ai.kind)) {
    out.kind = ai.kind as ParseResult["kind"];
    out.confidence = 0.95;
  }
  if (typeof ai.title === "string" && ai.title.trim()) out.title = ai.title.trim().slice(0, 120);
  if (ai.date === null || isISODate(ai.date)) out.date = ai.date ?? null;
  if (ai.time === null || isTime(ai.time)) out.time = ai.time ?? null;
  if (ai.endTime === null || isTime(ai.endTime)) out.endTime = ai.endTime ?? null;
  if (ai.amountCents === null || (typeof ai.amountCents === "number" && ai.amountCents >= 0)) {
    out.amountCents = ai.amountCents ?? null;
  }
  if (ai.categorySlug === null || (ai.categorySlug && CATEGORIES.includes(ai.categorySlug))) {
    out.categorySlug = ai.categorySlug ?? null;
  }
  if (ai.priority && ["low", "normal", "high", "urgent"].includes(ai.priority)) {
    out.priority = ai.priority as ParseResult["priority"];
  }
  if (Array.isArray(ai.reminders)) {
    out.reminders = ai.reminders.filter((n) => Number.isInteger(n) && n >= 0 && n <= 43200);
  }
  if (typeof ai.location === "string" || ai.location === null) out.location = ai.location;

  // la ricorrenza resta al parser locale: il modello sbaglia troppo spesso
  out.recurrence = local.recurrence;

  if (out.kind === "expense" && out.amountCents === null) out.amountCents = local.amountCents;
  if (out.kind === "expense" && !out.date) out.date = todayISO();
  if (!out.date) out.reminders = [];

  return out;
}

function isISODate(v: unknown): v is string {
  return typeof v === "string" && /^\d{4}-\d{2}-\d{2}$/.test(v);
}

function isTime(v: unknown): v is string {
  return typeof v === "string" && /^([01]\d|2[0-3]):[0-5]\d$/.test(v);
}

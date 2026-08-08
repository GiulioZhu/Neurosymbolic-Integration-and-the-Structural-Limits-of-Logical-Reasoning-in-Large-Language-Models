"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CalendarPlus, CheckCircle2, CornerDownLeft, Sparkles, StickyNote, Wallet } from "lucide-react";
import { cn } from "@/lib/cn";
import { Button } from "@/components/ui/Button";
import { Chip } from "@/components/ui/Segmented";
import { useComposer } from "@/components/forms/Composer";
import { useStore } from "@/lib/data/store";
import { formatDayMonth, fromISODate, relativeDay } from "@/lib/date";
import { money } from "@/lib/format";
import { describeRecurrence } from "@/lib/recurrence";
import { parseCapture, type ParseResult } from "@/lib/parse";
import { KIND_COLOR, KIND_LABEL, PRIORITY_LABEL, type SourceKind } from "@/lib/types";

const ICONS: Record<SourceKind, React.ReactNode> = {
  task: <CheckCircle2 size={16} />,
  event: <CalendarPlus size={16} />,
  expense: <Wallet size={16} />,
  note: <StickyNote size={16} />,
};

/**
 * Il campo più importante dell'app.
 *
 * Interpreta il testo mentre si scrive (parser locale, istantaneo) e mostra
 * un'anteprima correggibile. Con `ANTHROPIC_API_KEY` configurata, alla pausa
 * di digitazione chiede anche a `/api/parse` un'interpretazione migliore.
 */
export function QuickCapture({ autoFocus }: { autoFocus?: boolean }) {
  const { createTask, createEvent, createExpense, createNote, categoryBySlug, pushToast } = useStore();
  const { open } = useComposer();

  const [text, setText] = useState("");
  // interpretazioni "appiccicate" al testo che le ha prodotte: quando il testo
  // cambia decadono da sole, senza effetti di reset
  const [ai, setAi] = useState<{ text: string; result: ParseResult } | null>(null);
  const [override, setOverride] = useState<{ text: string; kind: SourceKind } | null>(null);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const trimmed = text.trim();

  // parsing locale a ogni tasto: è sincrono e istantaneo, quindi è stato derivato
  const local = useMemo(
    () => (trimmed.length >= 3 ? parseCapture(trimmed) : null),
    [trimmed],
  );

  const aiUsed = ai !== null && ai.text === trimmed;
  const result = aiUsed ? ai.result : local;
  const kind = (override?.text === trimmed ? override.kind : undefined) ?? result?.kind ?? "note";

  // eventuale rifinitura AI, solo dopo una pausa e senza bloccare nulla
  useEffect(() => {
    if (trimmed.length < 8) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const res = await fetch("/api/parse", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ text: trimmed }),
          signal: controller.signal,
        });
        if (!res.ok) return;
        const json = (await res.json()) as { result: ParseResult; source: string };
        if (json.source === "ai" && json.result) setAi({ text: trimmed, result: json.result });
      } catch {
        /* il parser locale ha già dato una risposta valida */
      }
    }, 700);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [trimmed]);

  const reset = useCallback(() => {
    setText("");
    setAi(null);
    setOverride(null);
    inputRef.current?.focus();
  }, []);

  const save = async () => {
    if (!result) return;
    setSaving(true);
    try {
      const categoryId = categoryBySlug(result.categorySlug)?.id ?? null;
      switch (kind) {
        case "task":
          await createTask({
            title: result.title,
            due_date: result.date,
            due_time: result.time,
            priority: result.priority,
            reminders: result.reminders,
            recurrence: result.recurrence,
            category_id: categoryId,
          });
          pushToast("Attività creata");
          break;
        case "event":
          await createEvent({
            title: result.title,
            start_date: result.date ?? new Date().toISOString().slice(0, 10),
            start_time: result.time,
            end_time: result.endTime,
            all_day: !result.time,
            location: result.location,
            reminders: result.reminders,
            recurrence: result.recurrence,
          });
          pushToast("Evento creato");
          break;
        case "expense":
          await createExpense({
            amount_cents: result.amountCents ?? 0,
            description: result.title,
            category_id: categoryId ?? categoryBySlug("altro")?.id ?? null,
            spent_on: result.date ?? new Date().toISOString().slice(0, 10),
          });
          pushToast("Spesa registrata");
          break;
        default:
          await createNote({ body: result.raw });
          pushToast("Nota salvata in Inbox");
      }
      reset();
    } finally {
      setSaving(false);
    }
  };

  const editInForm = () => {
    if (!result) return;
    const categoryId = categoryBySlug(result.categorySlug)?.id ?? null;
    if (kind === "task") {
      open({
        kind: "task",
        initial: {
          title: result.title,
          due_date: result.date,
          due_time: result.time,
          priority: result.priority,
          reminders: result.reminders,
          recurrence: result.recurrence,
          category_id: categoryId,
        },
      });
    } else if (kind === "event") {
      open({
        kind: "event",
        initial: {
          title: result.title,
          start_date: result.date ?? new Date().toISOString().slice(0, 10),
          start_time: result.time,
          end_time: result.endTime,
          all_day: !result.time,
          location: result.location,
          reminders: result.reminders,
          recurrence: result.recurrence,
        },
      });
    } else if (kind === "expense") {
      open({
        kind: "expense",
        initial: {
          amount_cents: result.amountCents ?? 0,
          description: result.title,
          category_id: categoryId,
          spent_on: result.date ?? undefined,
        },
      });
    } else {
      open({ kind: "note", initial: { body: result.raw } });
    }
    reset();
  };

  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-2 rounded-[var(--radius-card)] border px-3.5 py-2.5 transition-colors",
          "border-[var(--border)] bg-[var(--surface)]",
          result && "rounded-b-none border-b-transparent",
        )}
      >
        <Sparkles size={17} className="flex-shrink-0 text-[var(--faint)]" />
        <input
          ref={inputRef}
          autoFocus={autoFocus}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && result) {
              e.preventDefault();
              void save();
            }
            if (e.key === "Escape") reset();
          }}
          placeholder="Scrivi qualsiasi cosa…"
          aria-label="Scrivi qualsiasi cosa"
          className="min-w-0 flex-1 border-none bg-transparent text-[16px] text-[var(--text)] placeholder:text-[var(--faint)] outline-none"
        />
        {text && (
          <button
            onClick={reset}
            aria-label="Cancella"
            className="press text-[13px] text-[var(--faint)]"
          >
            Annulla
          </button>
        )}
      </div>

      {result && (
        <div className="animate-rise rounded-b-[var(--radius-card)] border border-t-0 border-[var(--border)] bg-[var(--surface-2)] px-3.5 pt-3 pb-3">
          <div className="flex items-start gap-2.5">
            <span
              className="mt-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full"
              style={{ background: `${KIND_COLOR[kind]}1F`, color: KIND_COLOR[kind] }}
            >
              {ICONS[kind]}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <span
                  className="text-[11px] font-semibold tracking-wide uppercase"
                  style={{ color: KIND_COLOR[kind] }}
                >
                  {KIND_LABEL[kind]}
                </span>
                {aiUsed && (
                  <span className="rounded-full bg-[var(--accent-soft)] px-1.5 py-0.5 text-[10px] font-medium text-[var(--accent)]">
                    AI
                  </span>
                )}
                {!aiUsed && result.confidence < 0.6 && (
                  <span className="text-[11px] text-[var(--faint)]">interpretazione incerta</span>
                )}
              </div>
              <p className="truncate text-[16px] font-semibold">{result.title || "Senza titolo"}</p>
              <p className="mt-0.5 text-[13px] text-[var(--muted)]">{summarize(result, kind)}</p>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            <span className="mr-0.5 text-[12px] text-[var(--faint)]">È invece:</span>
            {(["task", "event", "expense", "note"] as SourceKind[])
              .filter((k) => k !== kind)
              .map((k) => (
                <Chip key={k} onClick={() => setOverride({ text: trimmed, kind: k })}>
                  {KIND_LABEL[k]}
                </Chip>
              ))}
          </div>

          <div className="mt-3 flex gap-2">
            <Button size="sm" onClick={editInForm} className="flex-1">
              Modifica
            </Button>
            <Button size="sm" variant="primary" onClick={save} loading={saving} className="flex-1">
              Salva
              <CornerDownLeft size={14} className="opacity-70" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function summarize(r: ParseResult, kind: SourceKind): string {
  const bits: string[] = [];

  if (kind === "expense") {
    if (r.amountCents !== null) bits.push(money(r.amountCents));
    if (r.categorySlug) bits.push(labelForSlug(r.categorySlug));
    bits.push(r.date ? relativeDay(r.date) : "oggi");
    return bits.join(" · ");
  }

  if (r.date) {
    const day = relativeDay(r.date);
    const exact = formatDayMonth(fromISODate(r.date));
    bits.push(day === exact ? day : `${day} ${exact}`);
  } else {
    bits.push("senza data");
  }
  if (r.time) bits.push(r.endTime ? `${r.time}–${r.endTime}` : r.time);
  if (r.location) bits.push(r.location);
  if (r.recurrence) bits.push(describeRecurrence(r.recurrence).toLowerCase());
  if (r.reminders.length) bits.push("con promemoria");
  if (r.priority !== "normal") bits.push(`priorità ${PRIORITY_LABEL[r.priority].toLowerCase()}`);

  return bits.join(" · ");
}

const SLUG_LABELS: Record<string, string> = {
  casa: "Casa",
  alimentari: "Spesa alimentare",
  ristoranti: "Ristoranti",
  trasporti: "Trasporti",
  shopping: "Shopping",
  viaggi: "Viaggi",
  salute: "Salute",
  bollette: "Bollette",
  regali: "Regali",
  tempolibero: "Tempo libero",
  altro: "Altro",
};

function labelForSlug(slug: string): string {
  return SLUG_LABELS[slug] ?? slug;
}

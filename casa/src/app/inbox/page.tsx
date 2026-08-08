"use client";

import { useState } from "react";
import { Archive, CalendarPlus, CheckCircle2, Inbox, Trash2, Wallet } from "lucide-react";
import { AppShell, PageHeader } from "@/components/shell/AppShell";
import { NoteRow, RowList } from "@/components/items/Rows";
import { Card, EmptyState } from "@/components/ui/Bits";
import { Button } from "@/components/ui/Button";
import { Sheet } from "@/components/ui/Sheet";
import { useComposer } from "@/components/forms/Composer";
import { useStore } from "@/lib/data/store";
import { parseCapture } from "@/lib/parse";
import { KIND_COLOR, type Note } from "@/lib/types";

export default function InboxPage() {
  return (
    <AppShell>
      <InboxScreen />
    </AppShell>
  );
}

function InboxScreen() {
  const { data, updateNote, deleteNote, categoryBySlug, pushToast } = useStore();
  const { open } = useComposer();
  const [active, setActive] = useState<Note | null>(null);

  const notes = data.notes
    .filter((n) => !n.archived_at && !n.converted_kind)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));

  /**
   * Convertire una nota significa aprire il form giusto già compilato: il
   * parser fa una prima lettura del testo, l'utente conferma.
   */
  const convert = (note: Note, kind: "task" | "event" | "expense") => {
    const parsed = parseCapture(note.body);
    const categoryId = categoryBySlug(parsed.categorySlug)?.id ?? null;
    const markConverted = () => void updateNote(note.id, { converted_kind: kind, archived_at: new Date().toISOString() });

    if (kind === "task") {
      open({
        kind: "task",
        initial: {
          title: parsed.title || note.body,
          due_date: parsed.date,
          due_time: parsed.time,
          priority: parsed.priority,
          reminders: parsed.reminders,
          category_id: categoryId,
        },
      });
    } else if (kind === "event") {
      open({
        kind: "event",
        initial: {
          title: parsed.title || note.body,
          start_date: parsed.date ?? new Date().toISOString().slice(0, 10),
          start_time: parsed.time,
          end_time: parsed.endTime,
          all_day: !parsed.time,
          location: parsed.location,
        },
      });
    } else {
      open({
        kind: "expense",
        initial: {
          amount_cents: parsed.amountCents ?? 0,
          description: parsed.title || note.body,
          category_id: categoryId,
        },
      });
    }

    markConverted();
    setActive(null);
    pushToast("Nota trasformata");
  };

  return (
    <>
      <PageHeader
        title="Inbox"
        subtitle="Scrivi ora, organizza dopo."
      />

      {notes.length === 0 ? (
        <Card>
          <EmptyState
            icon={<Inbox size={30} strokeWidth={1.5} />}
            title="Inbox vuota"
            description="Qui finiscono i pensieri buttati giù al volo. Tocca il + e scegli Nota."
            action={
              <Button variant="primary" size="sm" onClick={() => open({ kind: "note" })}>
                Scrivi una nota
              </Button>
            }
          />
        </Card>
      ) : (
        <RowList>
          {notes.map((n) => (
            <NoteRow key={n.id} note={n} onAct={() => setActive(n)} />
          ))}
        </RowList>
      )}

      <Sheet open={active !== null} onClose={() => setActive(null)} title="Cosa ne facciamo?">
        {active && (
          <div className="pb-4">
            <p className="mb-4 rounded-[var(--radius-field)] bg-[var(--surface-2)] px-3.5 py-3 text-[15px]">
              {active.body}
            </p>

            <div className="grid grid-cols-3 gap-2">
              <ConvertButton
                icon={<CheckCircle2 size={20} />}
                color={KIND_COLOR.task}
                label="Attività"
                onClick={() => convert(active, "task")}
              />
              <ConvertButton
                icon={<CalendarPlus size={20} />}
                color={KIND_COLOR.event}
                label="Evento"
                onClick={() => convert(active, "event")}
              />
              <ConvertButton
                icon={<Wallet size={20} />}
                color={KIND_COLOR.expense}
                label="Spesa"
                onClick={() => convert(active, "expense")}
              />
            </div>

            <div className="mt-4 flex gap-2">
              <Button
                full
                onClick={() => {
                  void updateNote(active.id, { archived_at: new Date().toISOString() });
                  setActive(null);
                  pushToast("Nota archiviata");
                }}
              >
                <Archive size={16} /> Archivia
              </Button>
              <Button
                variant="danger"
                onClick={() => {
                  void deleteNote(active.id);
                  setActive(null);
                  pushToast("Nota eliminata");
                }}
                aria-label="Elimina"
              >
                <Trash2 size={16} />
              </Button>
            </div>

            <button
              onClick={() => {
                open({ kind: "note", note: active });
                setActive(null);
              }}
              className="press mt-3 w-full text-center text-[14px] font-medium text-[var(--accent)]"
            >
              Modifica il testo
            </button>
          </div>
        )}
      </Sheet>
    </>
  );
}

function ConvertButton({
  icon,
  label,
  color,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  color: string;
  onClick(): void;
}) {
  return (
    <button
      onClick={onClick}
      className="press card flex flex-col items-center gap-1.5 py-3.5 text-[13px] font-medium"
    >
      <span style={{ color }}>{icon}</span>
      {label}
    </button>
  );
}

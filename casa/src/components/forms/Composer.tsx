"use client";

import { createContext, useCallback, useContext, useMemo, useState } from "react";
import { CalendarPlus, CheckCircle2, StickyNote, Wallet } from "lucide-react";
import { Sheet } from "@/components/ui/Sheet";
import { KIND_COLOR } from "@/lib/types";
import type { EventItem, Expense, Note, Task } from "@/lib/types";
import { TaskForm } from "./TaskForm";
import { EventForm } from "./EventForm";
import { ExpenseForm } from "./ExpenseForm";
import { NoteForm } from "./NoteForm";

export type ComposerTarget =
  | { kind: "picker" }
  | { kind: "task"; task?: Task; initial?: Partial<Task> }
  | { kind: "event"; event?: EventItem; initial?: Partial<EventItem> }
  | { kind: "note"; note?: Note; initial?: Partial<Note> }
  | { kind: "expense"; expense?: Expense; initial?: Partial<Expense> };

interface ComposerValue {
  open(target: ComposerTarget): void;
  close(): void;
}

const ComposerContext = createContext<ComposerValue | null>(null);

export function useComposer(): ComposerValue {
  const ctx = useContext(ComposerContext);
  if (!ctx) throw new Error("useComposer va usato dentro <ComposerProvider>");
  return ctx;
}

const TITLES: Record<Exclude<ComposerTarget["kind"], "picker">, [string, string]> = {
  task: ["Nuova attività", "Attività"],
  event: ["Nuovo evento", "Evento"],
  note: ["Nuova nota", "Nota"],
  expense: ["Nuova spesa", "Spesa"],
};

export function ComposerProvider({ children }: { children: React.ReactNode }) {
  const [target, setTarget] = useState<ComposerTarget | null>(null);

  const open = useCallback((t: ComposerTarget) => setTarget(t), []);
  const close = useCallback(() => setTarget(null), []);
  const value = useMemo(() => ({ open, close }), [open, close]);

  const isEditing =
    target !== null &&
    target.kind !== "picker" &&
    Boolean(
      (target.kind === "task" && target.task) ||
        (target.kind === "event" && target.event) ||
        (target.kind === "note" && target.note) ||
        (target.kind === "expense" && target.expense),
    );

  return (
    <ComposerContext.Provider value={value}>
      {children}

      <Sheet open={target?.kind === "picker"} onClose={close} title="Aggiungi">
        <div className="grid grid-cols-2 gap-2.5 pt-1 pb-4">
          <PickerButton
            icon={<CheckCircle2 size={22} />}
            color={KIND_COLOR.task}
            label="Attività"
            hint="Qualcosa da fare"
            onClick={() => open({ kind: "task" })}
          />
          <PickerButton
            icon={<CalendarPlus size={22} />}
            color={KIND_COLOR.event}
            label="Evento"
            hint="Con data e ora"
            onClick={() => open({ kind: "event" })}
          />
          <PickerButton
            icon={<Wallet size={22} />}
            color={KIND_COLOR.expense}
            label="Spesa"
            hint="Registra un importo"
            onClick={() => open({ kind: "expense" })}
          />
          <PickerButton
            icon={<StickyNote size={22} />}
            color={KIND_COLOR.note}
            label="Nota"
            hint="Salva nell'Inbox"
            onClick={() => open({ kind: "note" })}
          />
        </div>
      </Sheet>

      <Sheet
        open={target !== null && target.kind !== "picker"}
        onClose={close}
        height="tall"
        title={
          target && target.kind !== "picker"
            ? TITLES[target.kind][isEditing ? 1 : 0]
            : undefined
        }
      >
        {target?.kind === "task" && (
          <TaskForm task={target.task} initial={target.initial} onDone={close} />
        )}
        {target?.kind === "event" && (
          <EventForm event={target.event} initial={target.initial} onDone={close} />
        )}
        {target?.kind === "note" && (
          <NoteForm note={target.note} initial={target.initial} onDone={close} />
        )}
        {target?.kind === "expense" && (
          <ExpenseForm expense={target.expense} initial={target.initial} onDone={close} />
        )}
      </Sheet>
    </ComposerContext.Provider>
  );
}

function PickerButton({
  icon,
  label,
  hint,
  color,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  hint: string;
  color: string;
  onClick(): void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="press card flex flex-col items-start gap-1.5 p-4 text-left"
    >
      <span style={{ color }}>{icon}</span>
      <span className="text-[15px] font-semibold">{label}</span>
      <span className="text-[12px] text-[var(--faint)]">{hint}</span>
    </button>
  );
}

"use client";

import { Check, Clock, Lock, MapPin, Repeat, Bell } from "lucide-react";
import { cn } from "@/lib/cn";
import { Avatar, KindDot } from "@/components/ui/Bits";
import { useComposer } from "@/components/forms/Composer";
import { useStore } from "@/lib/data/store";
import { formatTime, relativeDay } from "@/lib/date";
import { money } from "@/lib/format";
import { KIND_COLOR, PAYMENT_LABEL, PRIORITY_COLOR, type EventItem, type Expense, type Note, type Task } from "@/lib/types";

// ---------------------------------------------------------------------------
//  Task
// ---------------------------------------------------------------------------

export function TaskRow({
  task,
  showDate,
  className,
}: {
  task: Task;
  showDate?: boolean;
  className?: string;
}) {
  const { toggleTask, memberName, member, category, user } = useStore();
  const { open } = useComposer();
  const done = task.status === "done";
  const overdue = !done && task.due_date !== null && task.due_date < new Date().toISOString().slice(0, 10);
  const author = member(task.created_by);
  const assignee = member(task.assigned_to);
  const cat = category(task.category_id);

  const meta: React.ReactNode[] = [];
  if (showDate && task.due_date) {
    meta.push(
      <span key="date" className={cn(overdue && "text-[var(--danger)]")}>
        {relativeDay(task.due_date)}
      </span>,
    );
  }
  if (task.due_time) {
    meta.push(
      <span key="time" className="inline-flex items-center gap-1">
        <Clock size={11} />
        {formatTime(task.due_time)}
      </span>,
    );
  }
  if (cat) {
    meta.push(
      <span key="cat" className="inline-flex items-center gap-1">
        <KindDot color={cat.color} size={5} />
        {cat.name}
      </span>,
    );
  }
  if (task.assigned_to) {
    meta.push(
      <span key="assignee">{task.assigned_to === user?.id ? "assegnata a te" : memberName(task.assigned_to)}</span>,
    );
  }
  if (task.recurrence) {
    meta.push(
      <span key="rec" className="inline-flex items-center gap-1">
        <Repeat size={11} />
      </span>,
    );
  }
  if (task.reminders.length > 0) {
    meta.push(
      <span key="rem" className="inline-flex items-center gap-1">
        <Bell size={11} />
      </span>,
    );
  }
  if (task.visibility === "private") {
    meta.push(
      <span key="priv" className="inline-flex items-center gap-1">
        <Lock size={11} /> solo io
      </span>,
    );
  }

  return (
    <div className={cn("flex items-start gap-3 px-4 py-3", className)}>
      <button
        onClick={() => void toggleTask(task)}
        aria-label={done ? `Riapri ${task.title}` : `Completa ${task.title}`}
        aria-pressed={done}
        className="press -m-2 flex h-11 w-11 flex-shrink-0 items-center justify-center p-2"
      >
        <span
          className={cn(
            "flex h-[22px] w-[22px] items-center justify-center rounded-full border-2 transition-colors",
            done
              ? "border-[var(--kind-task)] bg-[var(--kind-task)] text-white"
              : "border-[var(--surface-3)]",
          )}
          style={
            !done && task.priority !== "normal"
              ? { borderColor: PRIORITY_COLOR[task.priority] }
              : undefined
          }
        >
          {done && <Check size={13} strokeWidth={3} className="animate-pop" />}
        </span>
      </button>

      <button
        onClick={() => open({ kind: "task", task })}
        className="min-w-0 flex-1 text-left"
      >
        <div className="flex items-start gap-2">
          <span
            className={cn(
              "min-w-0 flex-1 text-[15px] leading-snug",
              done && "text-[var(--faint)] line-through",
            )}
          >
            {task.title}
          </span>
          {!done && (task.priority === "urgent" || task.priority === "high") && (
            <span
              className="mt-[3px] flex-shrink-0 text-[11px] font-bold"
              style={{ color: PRIORITY_COLOR[task.priority] }}
              title={task.priority === "urgent" ? "Urgente" : "Alta"}
            >
              {task.priority === "urgent" ? "!!" : "!"}
            </span>
          )}
        </div>
        {meta.length > 0 && (
          <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[12px] text-[var(--faint)]">
            {meta.map((m, i) => (
              <span key={i} className="inline-flex items-center gap-1">
                {i > 0 && <span aria-hidden className="text-[var(--surface-3)]">·</span>}
                {m}
              </span>
            ))}
          </div>
        )}
      </button>

      {author && (
        <Avatar
          name={author.display_name}
          accent={author.accent}
          size={22}
          title={`Creata da ${author.display_name}`}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
//  Evento
// ---------------------------------------------------------------------------

export function EventRow({
  event,
  date,
  showDate,
}: {
  event: EventItem;
  date?: string;
  showDate?: boolean;
}) {
  const { member } = useStore();
  const { open } = useComposer();
  const author = member(event.created_by);

  const when = event.all_day
    ? "Tutto il giorno"
    : [formatTime(event.start_time), event.end_time ? formatTime(event.end_time) : null]
        .filter(Boolean)
        .join("–");

  return (
    <button
      onClick={() => open({ kind: "event", event })}
      className="flex w-full items-start gap-3 px-4 py-3 text-left"
    >
      <span
        aria-hidden
        className="mt-1 h-[38px] w-[3px] flex-shrink-0 rounded-full"
        style={{ background: event.color ?? KIND_COLOR.event }}
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-[15px] leading-snug">{event.title}</p>
        <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[12px] text-[var(--faint)]">
          {showDate && date && <span>{relativeDay(date)}</span>}
          <span>{when}</span>
          {event.location && (
            <span className="inline-flex items-center gap-1">
              <MapPin size={11} />
              {event.location}
            </span>
          )}
          {event.recurrence && <Repeat size={11} />}
          {event.visibility === "private" && <Lock size={11} />}
        </div>
      </div>
      {author && <Avatar name={author.display_name} accent={author.accent} size={22} />}
    </button>
  );
}

// ---------------------------------------------------------------------------
//  Spesa
// ---------------------------------------------------------------------------

export function ExpenseRow({ expense, showDate = true }: { expense: Expense; showDate?: boolean }) {
  const { category, member } = useStore();
  const { open } = useComposer();
  const cat = category(expense.category_id);
  const payer = member(expense.paid_by);

  return (
    <button
      onClick={() => open({ kind: "expense", expense })}
      className="flex w-full items-center gap-3 px-4 py-3 text-left"
    >
      <span
        aria-hidden
        className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full text-[13px] font-semibold"
        style={{ background: `${cat?.color ?? KIND_COLOR.expense}1F`, color: cat?.color ?? KIND_COLOR.expense }}
      >
        {(cat?.name ?? "A").slice(0, 1)}
      </span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[15px] leading-snug">{expense.description || cat?.name || "Spesa"}</p>
        <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[12px] text-[var(--faint)]">
          {showDate && <span>{relativeDay(expense.spent_on)}</span>}
          <span>{PAYMENT_LABEL[expense.payment_method]}</span>
          {payer && <span>{payer.display_name}</span>}
          {expense.visibility === "private" && <Lock size={11} />}
        </div>
      </div>
      <span className="tnum flex-shrink-0 text-[15px] font-semibold">−{money(expense.amount_cents)}</span>
    </button>
  );
}

// ---------------------------------------------------------------------------
//  Nota
// ---------------------------------------------------------------------------

export function NoteRow({ note, onAct }: { note: Note; onAct?(): void }) {
  const { member } = useStore();
  const author = member(note.created_by);

  return (
    <button onClick={onAct} className="flex w-full items-start gap-3 px-4 py-3 text-left">
      <span
        aria-hidden
        className="mt-[7px] h-2 w-2 flex-shrink-0 rounded-full"
        style={{ background: KIND_COLOR.note }}
      />
      <div className="min-w-0 flex-1">
        <p className="text-[15px] leading-snug">{note.body}</p>
        <div className="mt-0.5 flex items-center gap-2 text-[12px] text-[var(--faint)]">
          <span>{relativeDay(note.created_at.slice(0, 10))}</span>
          {author && <span>{author.display_name}</span>}
          {note.visibility === "private" && <Lock size={11} />}
        </div>
      </div>
    </button>
  );
}

/** Elenco con separatori sottili fra le righe. */
export function RowList({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("card divide-y divide-[var(--border)]", className)}>{children}</div>
  );
}

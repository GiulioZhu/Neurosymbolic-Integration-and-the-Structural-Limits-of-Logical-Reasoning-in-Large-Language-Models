"use client";

import { useMemo, useState } from "react";
import { CalendarDays, ChevronLeft, ChevronRight } from "lucide-react";
import { AppShell } from "@/components/shell/AppShell";
import { Card, EmptyState, KindDot } from "@/components/ui/Bits";
import { IconButton } from "@/components/ui/Button";
import { Segmented } from "@/components/ui/Segmented";
import { EventRow, ExpenseRow, RowList, TaskRow } from "@/components/items/Rows";
import { cn } from "@/lib/cn";
import { useStore } from "@/lib/data/store";
import {
  addDays,
  addMonths,
  formatLongDate,
  formatMonthYear,
  fromISODate,
  monthGrid,
  toISODate,
  todayISO,
  weekDays,
} from "@/lib/date";
import { dayEntries, dayMarkers } from "@/lib/data/selectors";
import { KIND_COLOR, type EventItem, type Expense, type Task } from "@/lib/types";

type View = "day" | "week" | "month";

export default function CalendarPage() {
  return (
    <AppShell>
      <CalendarScreen />
    </AppShell>
  );
}

function CalendarScreen() {
  const [view, setView] = useState<View>("month");
  const [cursor, setCursor] = useState(() => new Date());
  const [selected, setSelected] = useState<string>(todayISO());

  const step = (dir: 1 | -1) => {
    if (view === "month") {
      const next = addMonths(cursor, dir);
      setCursor(next);
      setSelected(toISODate(next));
    } else {
      const next = addDays(cursor, dir * (view === "week" ? 7 : 1));
      setCursor(next);
      setSelected(toISODate(next));
    }
  };

  const label =
    view === "month"
      ? formatMonthYear(cursor)
      : view === "week"
        ? `${toShort(weekDays(cursor)[0])} – ${toShort(weekDays(cursor)[6])}`
        : formatLongDate(cursor);

  return (
    <>
      <header className="flex items-center gap-1 pt-5 pb-3">
        <IconButton label="Periodo precedente" onClick={() => step(-1)}>
          <ChevronLeft size={20} />
        </IconButton>
        <h1 className="flex-1 text-center text-[18px] font-semibold first-letter:uppercase">
          {label}
        </h1>
        <IconButton label="Periodo successivo" onClick={() => step(1)}>
          <ChevronRight size={20} />
        </IconButton>
      </header>

      <div className="mb-4 flex items-center gap-2">
        <Segmented
          className="flex-1"
          size="sm"
          value={view}
          onChange={setView}
          options={[
            { value: "day", label: "Giorno" },
            { value: "week", label: "Settimana" },
            { value: "month", label: "Mese" },
          ]}
        />
        <button
          onClick={() => {
            const now = new Date();
            setCursor(now);
            setSelected(toISODate(now));
          }}
          className="press rounded-full border border-[var(--border)] px-3 py-2 text-[13px] font-medium text-[var(--muted)]"
        >
          Oggi
        </button>
      </div>

      {view === "month" && (
        <MonthView cursor={cursor} selected={selected} onSelect={setSelected} />
      )}
      {view === "week" && <WeekView cursor={cursor} selected={selected} onSelect={setSelected} />}
      {view === "day" && <DayTimeline date={selected} />}

      {view !== "day" && <DayDetail date={selected} />}

      <Legend />
    </>
  );
}

// ---------------------------------------------------------------------------

function MonthView({
  cursor,
  selected,
  onSelect,
}: {
  cursor: Date;
  selected: string;
  onSelect(d: string): void;
}) {
  const { data } = useStore();
  const days = useMemo(() => monthGrid(cursor), [cursor]);
  const today = todayISO();

  return (
    <Card className="mb-5 p-2">
      <div className="grid grid-cols-7 pb-1">
        {["L", "M", "M", "G", "V", "S", "D"].map((d, i) => (
          <span key={i} className="text-center text-[11px] font-medium text-[var(--faint)]">
            {d}
          </span>
        ))}
      </div>
      <div className="grid grid-cols-7">
        {days.map((d) => {
          const iso = toISODate(d);
          const inMonth = d.getMonth() === cursor.getMonth();
          const isToday = iso === today;
          const isSelected = iso === selected;
          const markers = dayMarkers(data, iso);
          return (
            <button
              key={iso}
              onClick={() => onSelect(iso)}
              className={cn(
                "press flex h-[46px] flex-col items-center justify-center gap-1 rounded-[10px]",
                !inMonth && "opacity-35",
                isSelected && !isToday && "bg-[var(--surface-2)]",
              )}
            >
              <span
                className={cn(
                  "tnum flex h-6 w-6 items-center justify-center rounded-full text-[13px]",
                  isToday && "bg-[var(--accent)] font-semibold text-[var(--accent-contrast)]",
                  !isToday && isSelected && "font-semibold",
                )}
              >
                {d.getDate()}
              </span>
              <span className="flex h-1.5 items-center gap-[3px]">
                {markers.map((m) => (
                  <KindDot key={m} color={KIND_COLOR[m]} size={4} />
                ))}
              </span>
            </button>
          );
        })}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------

function WeekView({
  cursor,
  selected,
  onSelect,
}: {
  cursor: Date;
  selected: string;
  onSelect(d: string): void;
}) {
  const { data } = useStore();
  const days = useMemo(() => weekDays(cursor), [cursor]);
  const today = todayISO();

  return (
    <Card className="mb-5 overflow-hidden">
      <div className="grid grid-cols-7 divide-x divide-[var(--border)]">
        {days.map((d) => {
          const iso = toISODate(d);
          const entries = dayEntries(data, iso);
          const isToday = iso === today;
          return (
            <button
              key={iso}
              onClick={() => onSelect(iso)}
              className={cn(
                "press min-h-[112px] p-1.5 text-left align-top",
                iso === selected && "bg-[var(--surface-2)]",
              )}
            >
              <div className="mb-1 flex flex-col items-center">
                <span className="text-[10px] text-[var(--faint)]">
                  {["dom", "lun", "mar", "mer", "gio", "ven", "sab"][d.getDay()]}
                </span>
                <span
                  className={cn(
                    "tnum flex h-5 w-5 items-center justify-center rounded-full text-[12px]",
                    isToday && "bg-[var(--accent)] font-semibold text-[var(--accent-contrast)]",
                  )}
                >
                  {d.getDate()}
                </span>
              </div>
              <div className="space-y-[3px]">
                {entries.slice(0, 3).map((e) => (
                  <span
                    key={e.id}
                    className="block truncate rounded-[4px] px-1 py-[1px] text-[9px] leading-tight"
                    style={{ background: `${KIND_COLOR[e.kind]}22`, color: KIND_COLOR[e.kind] }}
                  >
                    {e.time ? `${e.time} ` : ""}
                    {e.title}
                  </span>
                ))}
                {entries.length > 3 && (
                  <span className="block px-1 text-[9px] text-[var(--faint)]">
                    +{entries.length - 3}
                  </span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------

const DAY_START = 6;
const DAY_END = 24;
const HOUR_PX = 46;

function DayTimeline({ date }: { date: string }) {
  const { data } = useStore();
  const entries = dayEntries(data, date);
  const timed = entries.filter((e) => e.time);
  const untimed = entries.filter((e) => !e.time);

  return (
    <>
      {untimed.length > 0 && (
        <Card className="mb-3 p-2">
          <p className="mb-1.5 px-1 text-[11px] font-medium tracking-wide text-[var(--faint)] uppercase">
            Tutto il giorno
          </p>
          <div className="flex flex-wrap gap-1.5 px-1 pb-1">
            {untimed.map((e) => (
              <span
                key={e.id}
                className="rounded-full px-2.5 py-1 text-[12px]"
                style={{ background: `${KIND_COLOR[e.kind]}1F`, color: KIND_COLOR[e.kind] }}
              >
                {e.title}
              </span>
            ))}
          </div>
        </Card>
      )}

      <Card className="mb-5 overflow-hidden pt-3">
        <div className="relative" style={{ height: (DAY_END - DAY_START) * HOUR_PX }}>
          {Array.from({ length: DAY_END - DAY_START }, (_, i) => DAY_START + i).map((h, i) => (
            <div
              key={h}
              className="absolute right-0 left-0 border-t border-[var(--border)]"
              style={{ top: i * HOUR_PX }}
            >
              <span className="tnum absolute -top-2 left-2 bg-[var(--surface)] pr-1 text-[10px] text-[var(--faint)]">
                {String(h).padStart(2, "0")}:00
              </span>
            </div>
          ))}

          {timed.map((e) => {
            const [h, m] = (e.time ?? "00:00").split(":").map(Number);
            const top = (h - DAY_START + m / 60) * HOUR_PX;
            if (top < 0) return null;
            const end = e.endTime ? e.endTime.split(":").map(Number) : null;
            const durationH = end ? Math.max(0.5, end[0] + end[1] / 60 - (h + m / 60)) : 0.75;
            return (
              <div
                key={e.id}
                className="absolute right-2 left-[52px] overflow-hidden rounded-[8px] px-2 py-1"
                style={{
                  top: top + 1,
                  height: Math.max(24, durationH * HOUR_PX - 3),
                  background: `${KIND_COLOR[e.kind]}1A`,
                  borderLeft: `3px solid ${KIND_COLOR[e.kind]}`,
                }}
              >
                <p className="truncate text-[13px] leading-tight font-medium">{e.title}</p>
                <p className="text-[11px] text-[var(--muted)]">
                  {e.time}
                  {e.endTime ? `–${e.endTime}` : ""}
                </p>
              </div>
            );
          })}
        </div>
      </Card>
    </>
  );
}

// ---------------------------------------------------------------------------

function DayDetail({ date }: { date: string }) {
  const { data } = useStore();
  const entries = dayEntries(data, date);

  return (
    <section className="mb-5">
      <h2 className="section-title mb-2 px-1 first-letter:uppercase">
        {formatLongDate(fromISODate(date))}
      </h2>
      {entries.length === 0 ? (
        <Card>
          <EmptyState
            icon={<CalendarDays size={28} strokeWidth={1.5} />}
            title="Giornata libera"
            description="Nessun evento, nessuna scadenza."
          />
        </Card>
      ) : (
        <RowList>
          {entries.map((e) =>
            e.kind === "event" ? (
              <EventRow key={e.id} event={e.source as EventItem} date={date} />
            ) : e.kind === "task" ? (
              <TaskRow key={e.id} task={e.source as Task} />
            ) : (
              <ExpenseRow key={e.id} expense={e.source as Expense} showDate={false} />
            ),
          )}
        </RowList>
      )}
    </section>
  );
}

function Legend() {
  return (
    <div className="flex items-center justify-center gap-4 pb-2 text-[12px] text-[var(--faint)]">
      <span className="inline-flex items-center gap-1.5">
        <KindDot color={KIND_COLOR.event} /> Eventi
      </span>
      <span className="inline-flex items-center gap-1.5">
        <KindDot color={KIND_COLOR.task} /> Attività
      </span>
      <span className="inline-flex items-center gap-1.5">
        <KindDot color={KIND_COLOR.expense} /> Spese
      </span>
    </div>
  );
}

function toShort(d: Date): string {
  return `${d.getDate()}/${d.getMonth() + 1}`;
}

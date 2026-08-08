"use client";

import { Lock, Users } from "lucide-react";
import { Chip } from "@/components/ui/Segmented";
import { Avatar } from "@/components/ui/Bits";
import { Input, Label } from "@/components/ui/Field";
import { addDays, fromISODate, toISODate, todayISO } from "@/lib/date";
import { useStore } from "@/lib/data/store";
import { describeRecurrence, WEEKDAY_NAMES } from "@/lib/recurrence";
import {
  PRIORITY_COLOR,
  PRIORITY_LABEL,
  REMINDER_PRESETS,
  reminderLabel,
  type ISODate,
  type Priority,
  type Recurrence,
  type UUID,
  type Visibility,
} from "@/lib/types";

// --- data ------------------------------------------------------------------

export function DateQuickChips({
  value,
  onChange,
}: {
  value: ISODate | null;
  onChange(v: ISODate | null): void;
}) {
  const today = todayISO();
  const tomorrow = toISODate(addDays(new Date(), 1));
  const nextWeek = toISODate(addDays(new Date(), 7));
  const options: [string, ISODate | null][] = [
    ["Oggi", today],
    ["Domani", tomorrow],
    ["Fra 7 giorni", nextWeek],
    ["Nessuna", null],
  ];
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map(([label, date]) => (
        <Chip key={label} active={value === date} onClick={() => onChange(date)}>
          {label}
        </Chip>
      ))}
    </div>
  );
}

// --- priorità ---------------------------------------------------------------

export function PriorityPicker({
  value,
  onChange,
}: {
  value: Priority;
  onChange(v: Priority): void;
}) {
  const order: Priority[] = ["low", "normal", "high", "urgent"];
  return (
    <div className="flex flex-wrap gap-1.5">
      {order.map((p) => (
        <Chip key={p} active={value === p} color={PRIORITY_COLOR[p]} onClick={() => onChange(p)}>
          {PRIORITY_LABEL[p]}
        </Chip>
      ))}
    </div>
  );
}

// --- categoria --------------------------------------------------------------

export function CategoryPicker({
  value,
  onChange,
  allowNone = true,
}: {
  value: UUID | null;
  onChange(v: UUID | null): void;
  allowNone?: boolean;
}) {
  const { data } = useStore();
  return (
    <div className="flex flex-wrap gap-1.5">
      {allowNone && (
        <Chip active={value === null} onClick={() => onChange(null)}>
          Nessuna
        </Chip>
      )}
      {data.categories.map((c) => (
        <Chip key={c.id} active={value === c.id} color={c.color} onClick={() => onChange(c.id)}>
          {c.name}
        </Chip>
      ))}
    </div>
  );
}

// --- persone ----------------------------------------------------------------

export function MemberPicker({
  value,
  onChange,
  allowNone = true,
  noneLabel = "Nessuno",
}: {
  value: UUID | null;
  onChange(v: UUID | null): void;
  allowNone?: boolean;
  noneLabel?: string;
}) {
  const { members, user } = useStore();
  return (
    <div className="flex flex-wrap gap-1.5">
      {allowNone && (
        <Chip active={value === null} onClick={() => onChange(null)}>
          {noneLabel}
        </Chip>
      )}
      {members.map((m) => (
        <Chip key={m.id} active={value === m.id} onClick={() => onChange(m.id)}>
          <Avatar name={m.display_name} accent={m.accent} size={16} />
          {m.id === user?.id ? "Io" : m.display_name}
        </Chip>
      ))}
    </div>
  );
}

// --- privacy ----------------------------------------------------------------

export function VisibilityPicker({
  value,
  onChange,
}: {
  value: Visibility;
  onChange(v: Visibility): void;
}) {
  return (
    <div className="flex gap-1.5">
      <Chip active={value === "shared"} onClick={() => onChange("shared")}>
        <Users size={13} /> Condiviso
      </Chip>
      <Chip active={value === "private"} onClick={() => onChange("private")}>
        <Lock size={13} /> Solo io
      </Chip>
    </div>
  );
}

// --- promemoria -------------------------------------------------------------

export function ReminderPicker({
  value,
  onChange,
  disabled,
}: {
  value: number[];
  onChange(v: number[]): void;
  disabled?: boolean;
}) {
  const toggle = (minutes: number) => {
    onChange(
      value.includes(minutes) ? value.filter((v) => v !== minutes) : [...value, minutes].sort((a, b) => a - b),
    );
  };
  const custom = value.filter((v) => !REMINDER_PRESETS.some((p) => p.value === v));

  if (disabled) {
    return <p className="text-[13px] text-[var(--faint)]">Imposta prima una data.</p>;
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        {REMINDER_PRESETS.map((p) => (
          <Chip key={p.value} active={value.includes(p.value)} onClick={() => toggle(p.value)}>
            {p.label}
          </Chip>
        ))}
        {custom.map((m) => (
          <Chip key={m} active onClick={() => toggle(m)}>
            {reminderLabel(m)}
          </Chip>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <Input
          type="number"
          min={1}
          placeholder="Minuti prima"
          className="h-9 max-w-[150px] py-1 text-[14px]"
          onKeyDown={(e) => {
            if (e.key !== "Enter") return;
            e.preventDefault();
            const n = Number((e.target as HTMLInputElement).value);
            if (n > 0 && !value.includes(n)) onChange([...value, n].sort((a, b) => a - b));
            (e.target as HTMLInputElement).value = "";
          }}
        />
        <span className="text-[12px] text-[var(--faint)]">Invio per aggiungere</span>
      </div>
    </div>
  );
}

// --- ricorrenza -------------------------------------------------------------

export function RecurrencePicker({
  value,
  onChange,
  anchor,
}: {
  value: Recurrence | null;
  onChange(v: Recurrence | null): void;
  anchor: ISODate | null;
}) {
  const base = anchor ? fromISODate(anchor) : new Date();

  const presets: [string, Recurrence | null][] = [
    ["Mai", null],
    ["Ogni giorno", { freq: "daily", interval: 1 }],
    ["Ogni settimana", { freq: "weekly", interval: 1, byweekday: [base.getDay()] }],
    ["Ogni mese", { freq: "monthly", interval: 1, bymonthday: base.getDate() }],
  ];

  const isActive = (r: Recurrence | null) => {
    if (!value && !r) return true;
    if (!value || !r) return false;
    return value.freq === r.freq && (value.interval ?? 1) === 1;
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        {presets.map(([label, r]) => (
          <Chip key={label} active={isActive(r)} onClick={() => onChange(r ? { ...r } : null)}>
            {label}
          </Chip>
        ))}
      </div>

      {value?.freq === "weekly" && (
        <div>
          <Label>Giorni</Label>
          <div className="flex flex-wrap gap-1.5">
            {[1, 2, 3, 4, 5, 6, 0].map((d) => {
              const selected = value.byweekday?.includes(d) ?? false;
              return (
                <Chip
                  key={d}
                  active={selected}
                  onClick={() => {
                    const current = value.byweekday ?? [];
                    const next = selected ? current.filter((x) => x !== d) : [...current, d];
                    onChange({ ...value, byweekday: next.length ? next.sort() : [base.getDay()] });
                  }}
                >
                  {WEEKDAY_NAMES[d].slice(0, 3)}
                </Chip>
              );
            })}
          </div>
        </div>
      )}

      {value?.freq === "monthly" && (
        <div className="flex items-center gap-2">
          <Label>Giorno del mese</Label>
          <Input
            type="number"
            min={1}
            max={31}
            value={value.bymonthday ?? base.getDate()}
            onChange={(e) =>
              onChange({ ...value, bymonthday: Math.min(31, Math.max(1, Number(e.target.value) || 1)) })
            }
            className="h-9 max-w-[80px] py-1 text-center text-[14px]"
          />
        </div>
      )}

      {value && (
        <p className="text-[12px] text-[var(--faint)]">{describeRecurrence(value)}</p>
      )}
    </div>
  );
}

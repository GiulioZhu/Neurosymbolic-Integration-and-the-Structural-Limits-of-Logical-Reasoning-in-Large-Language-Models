"use client";

import { useState } from "react";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Chip } from "@/components/ui/Segmented";
import { Field, Input, Textarea, Toggle } from "@/components/ui/Field";
import { Avatar } from "@/components/ui/Bits";
import { useStore } from "@/lib/data/store";
import { todayISO } from "@/lib/date";
import type { EventItem, ISODate, Recurrence, UUID, Visibility } from "@/lib/types";
import { DateQuickChips, RecurrencePicker, ReminderPicker, VisibilityPicker } from "./pickers";

export interface EventFormProps {
  event?: EventItem;
  initial?: Partial<EventItem>;
  onDone(): void;
}

export function EventForm({ event, initial, onDone }: EventFormProps) {
  const { createEvent, updateEvent, deleteEvent, members, pushToast } = useStore();
  const src = event ?? initial ?? {};

  const [title, setTitle] = useState(src.title ?? "");
  const [startDate, setStartDate] = useState<ISODate>(src.start_date ?? todayISO());
  const [startTime, setStartTime] = useState(src.start_time ?? "");
  const [endTime, setEndTime] = useState(src.end_time ?? "");
  const [allDay, setAllDay] = useState(src.all_day ?? !src.start_time);
  const [location, setLocation] = useState(src.location ?? "");
  const [description, setDescription] = useState(src.description ?? "");
  const [attendees, setAttendees] = useState<UUID[]>(src.attendees ?? []);
  const [reminders, setReminders] = useState<number[]>(src.reminders ?? []);
  const [recurrence, setRecurrence] = useState<Recurrence | null>(src.recurrence ?? null);
  const [visibility, setVisibility] = useState<Visibility>(src.visibility ?? "shared");
  const [saving, setSaving] = useState(false);
  const [showDetails, setShowDetails] = useState(Boolean(event));

  const toggleAttendee = (id: UUID) =>
    setAttendees((a) => (a.includes(id) ? a.filter((x) => x !== id) : [...a, id]));

  const save = async () => {
    const clean = title.trim();
    if (!clean) return;
    setSaving(true);
    const payload = {
      title: clean,
      start_date: startDate,
      start_time: allDay ? null : startTime || null,
      end_date: null,
      end_time: allDay ? null : endTime || null,
      all_day: allDay,
      location: location.trim() || null,
      description: description.trim() || null,
      attendees,
      reminders,
      recurrence,
      visibility,
    };
    try {
      if (event) {
        await updateEvent(event.id, payload);
        pushToast("Evento aggiornato");
      } else {
        await createEvent(payload);
        pushToast("Evento creato");
      }
      onDone();
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!event) return;
    await deleteEvent(event.id);
    pushToast("Evento eliminato");
    onDone();
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void save();
      }}
      className="pb-4"
    >
      <Input
        data-autofocus
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Titolo dell'evento"
        className="mb-3 border-transparent bg-transparent px-0 text-[19px] font-semibold focus:bg-transparent"
        aria-label="Titolo"
      />

      <Field label="Quando">
        <DateQuickChips value={startDate} onChange={(d) => d && setStartDate(d)} />
        <div className="mt-2 flex gap-2">
          <Input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value || todayISO())}
            className="flex-1"
            aria-label="Data"
          />
        </div>
        <div className="mt-2 flex items-center justify-between rounded-[var(--radius-field)] bg-[var(--surface-2)] px-3.5 py-2.5">
          <span className="text-[14px] text-[var(--muted)]">Tutto il giorno</span>
          <Toggle checked={allDay} onChange={setAllDay} label="Tutto il giorno" />
        </div>
        {!allDay && (
          <div className="mt-2 flex items-center gap-2">
            <Input
              type="time"
              value={startTime}
              onChange={(e) => setStartTime(e.target.value)}
              className="flex-1"
              aria-label="Ora inizio"
            />
            <span className="text-[var(--faint)]">–</span>
            <Input
              type="time"
              value={endTime}
              onChange={(e) => setEndTime(e.target.value)}
              className="flex-1"
              aria-label="Ora fine"
            />
          </div>
        )}
      </Field>

      <Field label="Luogo">
        <Input
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          placeholder="Dove"
        />
      </Field>

      <Field label="Partecipanti">
        <div className="flex flex-wrap gap-1.5">
          {members.map((m) => (
            <Chip key={m.id} active={attendees.includes(m.id)} onClick={() => toggleAttendee(m.id)}>
              <Avatar name={m.display_name} accent={m.accent} size={16} />
              {m.display_name}
            </Chip>
          ))}
        </div>
      </Field>

      <Field label="Privacy">
        <VisibilityPicker value={visibility} onChange={setVisibility} />
      </Field>

      {!showDetails ? (
        <button
          type="button"
          onClick={() => setShowDetails(true)}
          className="press mb-4 text-[14px] font-medium text-[var(--accent)]"
        >
          Altre opzioni
        </button>
      ) : (
        <>
          <Field label="Promemoria">
            <ReminderPicker value={reminders} onChange={setReminders} />
          </Field>

          <Field label="Ricorrenza">
            <RecurrencePicker value={recurrence} onChange={setRecurrence} anchor={startDate} />
          </Field>

          <Field label="Descrizione">
            <Textarea value={description} onChange={(e) => setDescription(e.target.value)} />
          </Field>
        </>
      )}

      <div className="flex items-center gap-2 pt-1">
        <Button type="submit" variant="primary" full loading={saving} disabled={!title.trim()}>
          {event ? "Salva" : "Aggiungi"}
        </Button>
        {event && (
          <Button type="button" variant="danger" onClick={remove} aria-label="Elimina">
            <Trash2 size={17} />
          </Button>
        )}
      </div>
    </form>
  );
}

"use client";

import { useState } from "react";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Field, Input, Textarea } from "@/components/ui/Field";
import { Segmented } from "@/components/ui/Segmented";
import { useStore } from "@/lib/data/store";
import type { ISODate, Priority, Recurrence, Task, TaskStatus, UUID, Visibility } from "@/lib/types";
import { STATUS_LABEL } from "@/lib/types";
import {
  CategoryPicker,
  DateQuickChips,
  MemberPicker,
  PriorityPicker,
  RecurrencePicker,
  ReminderPicker,
  VisibilityPicker,
} from "./pickers";

export interface TaskFormProps {
  task?: Task;
  initial?: Partial<Task>;
  onDone(): void;
}

export function TaskForm({ task, initial, onDone }: TaskFormProps) {
  const { createTask, updateTask, deleteTask, pushToast } = useStore();
  const src = task ?? initial ?? {};

  const [title, setTitle] = useState(src.title ?? "");
  const [description, setDescription] = useState(src.description ?? "");
  const [notes, setNotes] = useState(src.notes ?? "");
  const [dueDate, setDueDate] = useState<ISODate | null>(src.due_date ?? null);
  const [dueTime, setDueTime] = useState<string>(src.due_time ?? "");
  const [status, setStatus] = useState<TaskStatus>(src.status ?? "todo");
  const [priority, setPriority] = useState<Priority>(src.priority ?? "normal");
  const [categoryId, setCategoryId] = useState<UUID | null>(src.category_id ?? null);
  const [assignedTo, setAssignedTo] = useState<UUID | null>(src.assigned_to ?? null);
  const [reminders, setReminders] = useState<number[]>(src.reminders ?? []);
  const [recurrence, setRecurrence] = useState<Recurrence | null>(src.recurrence ?? null);
  const [visibility, setVisibility] = useState<Visibility>(src.visibility ?? "shared");
  const [saving, setSaving] = useState(false);
  const [showDetails, setShowDetails] = useState(Boolean(task));

  const save = async () => {
    const clean = title.trim();
    if (!clean) return;
    setSaving(true);
    const payload = {
      title: clean,
      description: description.trim() || null,
      notes: notes.trim() || null,
      due_date: dueDate,
      due_time: dueTime || null,
      status,
      priority,
      category_id: categoryId,
      assigned_to: assignedTo,
      reminders,
      recurrence,
      visibility,
    };
    try {
      if (task) {
        await updateTask(task.id, payload);
        pushToast("Attività aggiornata");
      } else {
        await createTask(payload);
        pushToast("Attività creata");
      }
      onDone();
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!task) return;
    await deleteTask(task.id);
    pushToast("Attività eliminata");
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
        placeholder="Cosa c'è da fare?"
        className="mb-3 border-transparent bg-transparent px-0 text-[19px] font-semibold focus:bg-transparent"
        aria-label="Titolo"
      />

      <Field label="Quando">
        <DateQuickChips value={dueDate} onChange={setDueDate} />
        <div className="mt-2 flex gap-2">
          <Input
            type="date"
            value={dueDate ?? ""}
            onChange={(e) => setDueDate(e.target.value || null)}
            className="flex-1"
            aria-label="Data"
          />
          <Input
            type="time"
            value={dueTime}
            onChange={(e) => setDueTime(e.target.value)}
            className="w-[122px]"
            aria-label="Ora"
          />
        </div>
      </Field>

      <Field label="Priorità">
        <PriorityPicker value={priority} onChange={setPriority} />
      </Field>

      <Field label="Assegnata a">
        <MemberPicker value={assignedTo} onChange={setAssignedTo} noneLabel="Nessuno" />
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
          <Field label="Stato">
            <Segmented
              size="sm"
              value={status}
              onChange={setStatus}
              options={(["todo", "doing", "done"] as TaskStatus[]).map((s) => ({
                value: s,
                label: STATUS_LABEL[s],
              }))}
            />
          </Field>

          <Field label="Categoria">
            <CategoryPicker value={categoryId} onChange={setCategoryId} />
          </Field>

          <Field label="Promemoria">
            <ReminderPicker value={reminders} onChange={setReminders} disabled={!dueDate} />
          </Field>

          <Field label="Ricorrenza">
            <RecurrencePicker value={recurrence} onChange={setRecurrence} anchor={dueDate} />
          </Field>

          <Field label="Descrizione">
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Dettagli"
            />
          </Field>

          <Field label="Note">
            <Textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} />
          </Field>
        </>
      )}

      <div className="flex items-center gap-2 pt-1">
        <Button type="submit" variant="primary" full loading={saving} disabled={!title.trim()}>
          {task ? "Salva" : "Aggiungi"}
        </Button>
        {task && (
          <Button type="button" variant="danger" onClick={remove} aria-label="Elimina">
            <Trash2 size={17} />
          </Button>
        )}
      </div>
    </form>
  );
}

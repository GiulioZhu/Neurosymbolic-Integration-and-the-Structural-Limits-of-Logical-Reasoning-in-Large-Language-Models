"use client";

import { useState } from "react";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Field, Textarea } from "@/components/ui/Field";
import { useStore } from "@/lib/data/store";
import type { Note, Visibility } from "@/lib/types";
import { VisibilityPicker } from "./pickers";

export interface NoteFormProps {
  note?: Note;
  initial?: Partial<Note>;
  onDone(): void;
}

export function NoteForm({ note, initial, onDone }: NoteFormProps) {
  const { createNote, updateNote, deleteNote, pushToast } = useStore();
  const src = note ?? initial ?? {};

  const [body, setBody] = useState(src.body ?? "");
  const [visibility, setVisibility] = useState<Visibility>(src.visibility ?? "shared");
  const [saving, setSaving] = useState(false);

  const save = async () => {
    const clean = body.trim();
    if (!clean) return;
    setSaving(true);
    try {
      if (note) {
        await updateNote(note.id, { body: clean, visibility });
        pushToast("Nota aggiornata");
      } else {
        await createNote({ body: clean, visibility });
        pushToast("Nota salvata in Inbox");
      }
      onDone();
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!note) return;
    await deleteNote(note.id);
    pushToast("Nota eliminata");
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
      <Field>
        <Textarea
          data-autofocus
          value={body}
          onChange={(e) => setBody(e.target.value)}
          rows={5}
          placeholder="Scrivi un pensiero, un'idea, qualcosa da non dimenticare…"
          aria-label="Nota"
        />
      </Field>

      <Field label="Privacy">
        <VisibilityPicker value={visibility} onChange={setVisibility} />
      </Field>

      <div className="flex items-center gap-2 pt-1">
        <Button type="submit" variant="primary" full loading={saving} disabled={!body.trim()}>
          {note ? "Salva" : "Aggiungi all'Inbox"}
        </Button>
        {note && (
          <Button type="button" variant="danger" onClick={remove} aria-label="Elimina">
            <Trash2 size={17} />
          </Button>
        )}
      </div>
    </form>
  );
}

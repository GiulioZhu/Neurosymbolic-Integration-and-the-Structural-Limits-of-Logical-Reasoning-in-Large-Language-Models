/** Modello di dominio condiviso da UI, driver dati e parser. */

export type UUID = string;

export type Visibility = "shared" | "private";
export type TaskStatus = "todo" | "doing" | "done";
export type Priority = "low" | "normal" | "high" | "urgent";
export type SourceKind = "task" | "event" | "expense" | "note";
export type PaymentMethod = "card" | "cash" | "transfer" | "other";
export type CategoryKind = "expense" | "task";

/** Data in formato `yyyy-MM-dd`. Locale, senza fuso: "domani" non ha un'ora. */
export type ISODate = string;
/** Ora in formato `HH:mm`. */
export type ISOTime = string;

export interface Recurrence {
  freq: "daily" | "weekly" | "monthly";
  /** ogni N giorni / settimane / mesi */
  interval: number;
  /** solo weekly — 0 = domenica … 6 = sabato */
  byweekday?: number[];
  /** solo monthly — giorno del mese */
  bymonthday?: number;
  until?: ISODate | null;
}

export interface Profile {
  id: UUID;
  display_name: string;
  email: string | null;
  accent: string;
  created_at: string;
  updated_at: string;
}

export interface Household {
  id: UUID;
  name: string;
  invite_code: string;
  currency: string;
  created_by: UUID;
  created_at: string;
  updated_at: string;
}

export interface Category {
  id: UUID;
  household_id: UUID;
  kind: CategoryKind;
  slug: string;
  name: string;
  icon: string;
  color: string;
  position: number;
  created_by: UUID | null;
  created_at: string;
  updated_at: string;
}

export interface Task {
  id: UUID;
  household_id: UUID;
  title: string;
  description: string | null;
  notes: string | null;
  due_date: ISODate | null;
  due_time: ISOTime | null;
  status: TaskStatus;
  priority: Priority;
  category_id: UUID | null;
  assigned_to: UUID | null;
  reminders: number[];
  recurrence: Recurrence | null;
  recurring_item_id: UUID | null;
  completed_at: string | null;
  completed_by: UUID | null;
  visibility: Visibility;
  created_by: UUID;
  created_at: string;
  updated_at: string;
}

export interface EventItem {
  id: UUID;
  household_id: UUID;
  title: string;
  description: string | null;
  location: string | null;
  start_date: ISODate;
  start_time: ISOTime | null;
  end_date: ISODate | null;
  end_time: ISOTime | null;
  all_day: boolean;
  attendees: UUID[];
  reminders: number[];
  recurrence: Recurrence | null;
  recurring_item_id: UUID | null;
  color: string | null;
  visibility: Visibility;
  created_by: UUID;
  created_at: string;
  updated_at: string;
}

export interface Note {
  id: UUID;
  household_id: UUID;
  body: string;
  pinned: boolean;
  archived_at: string | null;
  converted_kind: SourceKind | null;
  converted_id: UUID | null;
  visibility: Visibility;
  created_by: UUID;
  created_at: string;
  updated_at: string;
}

export interface Expense {
  id: UUID;
  household_id: UUID;
  amount_cents: number;
  currency: string;
  description: string;
  category_id: UUID | null;
  spent_on: ISODate;
  payment_method: PaymentMethod;
  paid_by: UUID;
  note: string | null;
  visibility: Visibility;
  created_by: UUID;
  created_at: string;
  updated_at: string;
}

export interface NotificationRow {
  id: UUID;
  household_id: UUID;
  user_id: UUID | null;
  source_kind: SourceKind;
  source_id: UUID;
  title: string;
  body: string | null;
  scheduled_for: string;
  sent_at: string | null;
  read_at: string | null;
  visibility: Visibility;
  created_by: UUID;
  created_at: string;
  updated_at: string;
}

export interface RecurringItem {
  id: UUID;
  household_id: UUID;
  source_kind: SourceKind;
  rule: Recurrence;
  template: Record<string, unknown>;
  active: boolean;
  created_by: UUID;
  created_at: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
//  Etichette (unico posto dove vivono le stringhe italiane degli enum)
// ---------------------------------------------------------------------------

export const PRIORITY_LABEL: Record<Priority, string> = {
  low: "Bassa",
  normal: "Normale",
  high: "Alta",
  urgent: "Urgente",
};

export const PRIORITY_COLOR: Record<Priority, string> = {
  low: "var(--faint)",
  normal: "#4E7C9B",
  high: "#D08A24",
  urgent: "#C0453C",
};

export const STATUS_LABEL: Record<TaskStatus, string> = {
  todo: "Da fare",
  doing: "In corso",
  done: "Completato",
};

export const PAYMENT_LABEL: Record<PaymentMethod, string> = {
  card: "Carta",
  cash: "Contanti",
  transfer: "Bonifico",
  other: "Altro",
};

export const KIND_COLOR: Record<SourceKind, string> = {
  task: "#3F8F7E",
  event: "#7A6BB5",
  expense: "#C4873C",
  note: "#8A8175",
};

export const KIND_LABEL: Record<SourceKind, string> = {
  task: "Attività",
  event: "Evento",
  expense: "Spesa",
  note: "Nota",
};

/** Opzioni di promemoria, in minuti prima della scadenza. */
export const REMINDER_PRESETS: { value: number; label: string }[] = [
  { value: 0, label: "All'orario" },
  { value: 10, label: "10 minuti prima" },
  { value: 30, label: "30 minuti prima" },
  { value: 60, label: "1 ora prima" },
  { value: 1440, label: "1 giorno prima" },
];

export function reminderLabel(minutes: number): string {
  const preset = REMINDER_PRESETS.find((p) => p.value === minutes);
  if (preset) return preset.label;
  if (minutes % 1440 === 0) return `${minutes / 1440} giorni prima`;
  if (minutes % 60 === 0) return `${minutes / 60} ore prima`;
  return `${minutes} minuti prima`;
}

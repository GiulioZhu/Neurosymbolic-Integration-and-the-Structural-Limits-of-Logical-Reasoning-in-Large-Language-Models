"use client";

/**
 * Stato globale dell'app.
 *
 * Carica l'intero spazio condiviso in memoria (sono poche migliaia di righe
 * all'anno), applica le mutazioni in modo ottimistico e riconcilia con il
 * driver. Tutte le viste sono funzioni pure su questo snapshot: è il motivo
 * per cui la ricerca e la dashboard sono istantanee.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { combine, todayISO } from "../date";
import { nextOccurrence } from "../recurrence";
import type {
  Category,
  EventItem,
  Expense,
  Household,
  Note,
  NotificationRow,
  Profile,
  SourceKind,
  Task,
  UUID,
  Visibility,
} from "../types";
import {
  EMPTY_SNAPSHOT,
  friendlyError,
  type AuthUser,
  type ChangeEvent,
  type Driver,
  type Snapshot,
  type TableName,
} from "./driver";
import { LocalDriver } from "./local";
import { SupabaseDriver, supabaseConfigured } from "./supabase";

export type StoreStatus = "loading" | "anonymous" | "onboarding" | "ready";

export interface Toast {
  id: string;
  message: string;
  tone: "default" | "error";
  action?: { label: string; run: () => void };
}

export interface StoreValue {
  status: StoreStatus;
  driverKind: Driver["kind"];
  user: AuthUser | null;
  profile: Profile | null;
  household: Household | null;
  members: Profile[];
  data: Snapshot;
  toasts: Toast[];

  // auth
  signIn(email: string, password: string): Promise<void>;
  signUp(email: string, password: string, name: string): Promise<{ needsConfirmation: boolean }>;
  signOut(): Promise<void>;
  requestPasswordReset(email: string): Promise<void>;
  updatePassword(password: string): Promise<void>;

  // spazio
  createHousehold(name: string): Promise<void>;
  joinHousehold(code: string): Promise<void>;
  rotateInviteCode(): Promise<string>;
  saveProfile(patch: Partial<Profile>): Promise<void>;

  // contenuti
  createTask(input: TaskInput): Promise<Task>;
  updateTask(id: UUID, patch: Partial<Task>): Promise<void>;
  toggleTask(task: Task): Promise<void>;
  deleteTask(id: UUID): Promise<void>;

  createEvent(input: EventInput): Promise<EventItem>;
  updateEvent(id: UUID, patch: Partial<EventItem>): Promise<void>;
  deleteEvent(id: UUID): Promise<void>;

  createNote(input: NoteInput): Promise<Note>;
  updateNote(id: UUID, patch: Partial<Note>): Promise<void>;
  deleteNote(id: UUID): Promise<void>;

  createExpense(input: ExpenseInput): Promise<Expense>;
  updateExpense(id: UUID, patch: Partial<Expense>): Promise<void>;
  deleteExpense(id: UUID): Promise<void>;

  markNotificationSent(id: UUID): Promise<void>;

  // utilità
  refresh(): Promise<void>;
  pushToast(message: string, opts?: Partial<Omit<Toast, "id" | "message">>): void;
  dismissToast(id: string): void;
  memberName(id: UUID | null | undefined): string;
  member(id: UUID | null | undefined): Profile | null;
  category(id: UUID | null | undefined): Category | null;
  categoryBySlug(slug: string | null | undefined): Category | null;
  driver: Driver;
}

export type TaskInput = Partial<Task> & { title: string };
export type EventInput = Partial<EventItem> & { title: string; start_date: string };
export type NoteInput = Partial<Note> & { body: string };
export type ExpenseInput = Partial<Expense> & { amount_cents: number };

const StoreContext = createContext<StoreValue | null>(null);

export function useStore(): StoreValue {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStore va usato dentro <StoreProvider>");
  return ctx;
}

const newId = (): UUID =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `id-${Math.random().toString(36).slice(2)}-${Date.now()}`;

const nowISO = () => new Date().toISOString();

function upsert<T extends { id: UUID }>(rows: T[], row: T): T[] {
  const idx = rows.findIndex((r) => r.id === row.id);
  if (idx === -1) return [...rows, row];
  const copy = rows.slice();
  copy[idx] = { ...copy[idx], ...row };
  return copy;
}

export function StoreProvider({ children }: { children: React.ReactNode }) {
  const driver = useMemo<Driver>(
    () => (supabaseConfigured ? new SupabaseDriver() : new LocalDriver()),
    [],
  );

  const [status, setStatus] = useState<StoreStatus>("loading");
  const [user, setUser] = useState<AuthUser | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [household, setHousehold] = useState<Household | null>(null);
  const [members, setMembers] = useState<Profile[]>([]);
  const [data, setData] = useState<Snapshot>(EMPTY_SNAPSHOT);
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Copie in ref di household/user: servono ai callback (base(), syncReminders,
  // il merge realtime) per leggere il valore corrente senza finire fra le
  // dipendenze e ricreare mezza API a ogni cambio. Si aggiornano in un effetto,
  // mai durante il render.
  const householdRef = useRef<Household | null>(null);
  const userRef = useRef<AuthUser | null>(null);

  useEffect(() => {
    householdRef.current = household;
  }, [household]);

  useEffect(() => {
    userRef.current = user;
  }, [user]);

  // --- toast --------------------------------------------------------------

  const dismissToast = useCallback((id: string) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const pushToast = useCallback<StoreValue["pushToast"]>(
    (message, opts) => {
      const id = newId();
      setToasts((t) => [...t.slice(-2), { id, message, tone: "default", ...opts }]);
      setTimeout(() => dismissToast(id), opts?.action ? 6000 : 3200);
    },
    [dismissToast],
  );

  // --- caricamento --------------------------------------------------------

  const loadFor = useCallback(
    async (authUser: AuthUser) => {
      const boot = await driver.loadBootstrap(authUser);
      setProfile(boot.profile);
      setHousehold(boot.household);
      setMembers(boot.members);
      if (!boot.household) {
        setData(EMPTY_SNAPSHOT);
        setStatus("onboarding");
        return;
      }
      const snapshot = await driver.fetchSnapshot(boot.household.id);
      setData(snapshot);
      setStatus("ready");
    },
    [driver],
  );

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const current = await driver.getUser();
        if (cancelled) return;
        setUser(current);
        if (!current) {
          setStatus("anonymous");
          return;
        }
        await loadFor(current);
      } catch (err) {
        if (!cancelled) {
          setStatus("anonymous");
          pushToast(friendlyError(err), { tone: "error" });
        }
      }
    })();

    const off = driver.onAuthChange((u) => {
      setUser(u);
      if (!u) {
        setProfile(null);
        setHousehold(null);
        setMembers([]);
        setData(EMPTY_SNAPSHOT);
        setStatus("anonymous");
      } else {
        setStatus("loading");
        loadFor(u).catch((err) => {
          setStatus("anonymous");
          pushToast(friendlyError(err), { tone: "error" });
        });
      }
    });

    return () => {
      cancelled = true;
      off();
    };
  }, [driver, loadFor, pushToast]);

  // --- realtime -----------------------------------------------------------

  useEffect(() => {
    if (!household) return;
    const off = driver.subscribe(household.id, (ev: ChangeEvent) => {
      setData((prev) => {
        const table = ev.table;
        const rows = prev[table] as { id: UUID }[];
        if (ev.type === "DELETE") {
          return { ...prev, [table]: rows.filter((r) => r.id !== ev.row.id) };
        }
        // un elemento diventato privato di qualcun altro esce dalla cache
        const visibility = (ev.row as { visibility?: Visibility }).visibility;
        const createdBy = (ev.row as { created_by?: UUID }).created_by;
        if (visibility === "private" && createdBy && createdBy !== userRef.current?.id) {
          return { ...prev, [table]: rows.filter((r) => r.id !== ev.row.id) };
        }
        return { ...prev, [table]: upsert(rows, ev.row as { id: UUID }) };
      });
    });
    return off;
  }, [driver, household]);

  // --- helper di mutazione ------------------------------------------------

  const applyLocal = useCallback(<T extends TableName>(table: T, row: { id: UUID }) => {
    setData((prev) => ({ ...prev, [table]: upsert(prev[table] as { id: UUID }[], row) }));
  }, []);

  const removeLocal = useCallback((table: TableName, id: UUID) => {
    setData((prev) => ({
      ...prev,
      [table]: (prev[table] as { id: UUID }[]).filter((r) => r.id !== id),
    }));
  }, []);

  /** Inserisce in modo ottimistico e fa rollback se il driver rifiuta. */
  const insertOptimistic = useCallback(
    async <T extends TableName>(table: T, row: Snapshot[T][number]): Promise<Snapshot[T][number]> => {
      const typed = row as { id: UUID };
      applyLocal(table, typed);
      try {
        const saved = await driver.insert(table, row);
        applyLocal(table, saved as { id: UUID });
        return saved;
      } catch (err) {
        removeLocal(table, typed.id);
        pushToast(friendlyError(err), { tone: "error" });
        throw err;
      }
    },
    [applyLocal, driver, pushToast, removeLocal],
  );

  const updateOptimistic = useCallback(
    async <T extends TableName>(table: T, id: UUID, patch: Partial<Snapshot[T][number]>) => {
      const before = (data[table] as { id: UUID }[]).find((r) => r.id === id);
      applyLocal(table, { ...(before ?? { id }), ...(patch as object), id } as { id: UUID });
      try {
        const saved = await driver.update(table, id, patch);
        applyLocal(table, saved as { id: UUID });
      } catch (err) {
        if (before) applyLocal(table, before);
        pushToast(friendlyError(err), { tone: "error" });
        throw err;
      }
    },
    [applyLocal, data, driver, pushToast],
  );

  const deleteOptimistic = useCallback(
    async (table: TableName, id: UUID) => {
      const before = (data[table] as { id: UUID }[]).find((r) => r.id === id);
      removeLocal(table, id);
      try {
        await driver.remove(table, id);
      } catch (err) {
        if (before) applyLocal(table, before);
        pushToast(friendlyError(err), { tone: "error" });
        throw err;
      }
    },
    [applyLocal, data, driver, pushToast, removeLocal],
  );

  // --- promemoria ---------------------------------------------------------

  /** Riscrive le notifiche pendenti di un elemento a partire dai suoi reminder. */
  const syncReminders = useCallback(
    async (
      kind: SourceKind,
      id: UUID,
      opts: {
        title: string;
        body: string | null;
        date: string | null;
        time: string | null;
        reminders: number[];
        visibility: Visibility;
      },
    ) => {
      const hid = householdRef.current?.id;
      const uidNow = userRef.current?.id;
      if (!hid || !uidNow) return;

      try {
        await driver.clearNotificationsFor(kind, id);
      } catch {
        /* non bloccare il salvataggio per i promemoria */
      }
      setData((prev) => ({
        ...prev,
        notifications: prev.notifications.filter(
          (n) => !(n.source_kind === kind && n.source_id === id && !n.sent_at),
        ),
      }));

      if (!opts.date || opts.reminders.length === 0) return;
      const base = combine(opts.date, opts.time ?? "09:00");

      for (const minutes of opts.reminders) {
        const when = new Date(base.getTime() - minutes * 60_000);
        if (when.getTime() <= Date.now()) continue;
        const row: NotificationRow = {
          id: newId(),
          household_id: hid,
          user_id: null,
          source_kind: kind,
          source_id: id,
          title: opts.title,
          body: opts.body,
          scheduled_for: when.toISOString(),
          sent_at: null,
          read_at: null,
          visibility: opts.visibility,
          created_by: uidNow,
          created_at: nowISO(),
          updated_at: nowISO(),
        };
        try {
          await insertOptimistic("notifications", row);
        } catch {
          /* idem */
        }
      }
    },
    [driver, insertOptimistic],
  );

  // --- API contenuti ------------------------------------------------------

  const base = useCallback(() => {
    const hid = householdRef.current?.id;
    const uidNow = userRef.current?.id;
    if (!hid || !uidNow) throw new Error("Spazio non pronto");
    return {
      household_id: hid,
      created_by: uidNow,
      created_at: nowISO(),
      updated_at: nowISO(),
    };
  }, []);

  const createTask = useCallback<StoreValue["createTask"]>(
    async (input) => {
      const row: Task = {
        id: input.id ?? newId(),
        title: input.title.trim(),
        description: input.description ?? null,
        notes: input.notes ?? null,
        due_date: input.due_date ?? null,
        due_time: input.due_time ?? null,
        status: input.status ?? "todo",
        priority: input.priority ?? "normal",
        category_id: input.category_id ?? null,
        assigned_to: input.assigned_to ?? null,
        reminders: input.reminders ?? [],
        recurrence: input.recurrence ?? null,
        recurring_item_id: input.recurring_item_id ?? null,
        completed_at: null,
        completed_by: null,
        visibility: input.visibility ?? "shared",
        ...base(),
      };
      const saved = (await insertOptimistic("tasks", row)) as Task;
      void syncReminders("task", saved.id, {
        title: saved.title,
        body: "Attività in scadenza",
        date: saved.due_date,
        time: saved.due_time,
        reminders: saved.reminders,
        visibility: saved.visibility,
      });
      return saved;
    },
    [base, insertOptimistic, syncReminders],
  );

  const updateTask = useCallback<StoreValue["updateTask"]>(
    async (id, patch) => {
      await updateOptimistic("tasks", id, patch);
      const current = { ...(data.tasks.find((t) => t.id === id) as Task), ...patch };
      if ("reminders" in patch || "due_date" in patch || "due_time" in patch || "title" in patch) {
        void syncReminders("task", id, {
          title: current.title,
          body: "Attività in scadenza",
          date: current.due_date,
          time: current.due_time,
          reminders: current.reminders ?? [],
          visibility: current.visibility,
        });
      }
    },
    [data.tasks, syncReminders, updateOptimistic],
  );

  const toggleTask = useCallback<StoreValue["toggleTask"]>(
    async (task) => {
      const done = task.status === "done";
      if (done) {
        await updateOptimistic("tasks", task.id, {
          status: "todo",
          completed_at: null,
          completed_by: null,
        });
        return;
      }

      await updateOptimistic("tasks", task.id, {
        status: "done",
        completed_at: nowISO(),
        completed_by: userRef.current?.id ?? null,
      });

      // una task ricorrente chiude l'occorrenza e ne apre subito la successiva
      if (task.recurrence && task.due_date) {
        const next = nextOccurrence(task.due_date, task.recurrence, task.due_date);
        if (next) {
          await createTask({
            title: task.title,
            description: task.description,
            notes: task.notes,
            due_date: next,
            due_time: task.due_time,
            priority: task.priority,
            category_id: task.category_id,
            assigned_to: task.assigned_to,
            reminders: task.reminders,
            recurrence: task.recurrence,
            recurring_item_id: task.recurring_item_id,
            visibility: task.visibility,
          });
        }
      }
    },
    [createTask, updateOptimistic],
  );

  const createEvent = useCallback<StoreValue["createEvent"]>(
    async (input) => {
      const row: EventItem = {
        id: input.id ?? newId(),
        title: input.title.trim(),
        description: input.description ?? null,
        location: input.location ?? null,
        start_date: input.start_date,
        start_time: input.start_time ?? null,
        end_date: input.end_date ?? null,
        end_time: input.end_time ?? null,
        all_day: input.all_day ?? !input.start_time,
        attendees: input.attendees ?? [],
        reminders: input.reminders ?? [],
        recurrence: input.recurrence ?? null,
        recurring_item_id: input.recurring_item_id ?? null,
        color: input.color ?? null,
        visibility: input.visibility ?? "shared",
        ...base(),
      };
      const saved = (await insertOptimistic("events", row)) as EventItem;
      void syncReminders("event", saved.id, {
        title: saved.title,
        body: saved.location ? `Evento · ${saved.location}` : "Evento in arrivo",
        date: saved.start_date,
        time: saved.start_time,
        reminders: saved.reminders,
        visibility: saved.visibility,
      });
      return saved;
    },
    [base, insertOptimistic, syncReminders],
  );

  const updateEvent = useCallback<StoreValue["updateEvent"]>(
    async (id, patch) => {
      await updateOptimistic("events", id, patch);
      const current = { ...(data.events.find((e) => e.id === id) as EventItem), ...patch };
      if ("reminders" in patch || "start_date" in patch || "start_time" in patch || "title" in patch) {
        void syncReminders("event", id, {
          title: current.title,
          body: current.location ? `Evento · ${current.location}` : "Evento in arrivo",
          date: current.start_date,
          time: current.start_time,
          reminders: current.reminders ?? [],
          visibility: current.visibility,
        });
      }
    },
    [data.events, syncReminders, updateOptimistic],
  );

  const createNote = useCallback<StoreValue["createNote"]>(
    async (input) => {
      const row: Note = {
        id: input.id ?? newId(),
        body: input.body.trim(),
        pinned: input.pinned ?? false,
        archived_at: null,
        converted_kind: null,
        converted_id: null,
        visibility: input.visibility ?? "shared",
        ...base(),
      };
      return (await insertOptimistic("notes", row)) as Note;
    },
    [base, insertOptimistic],
  );

  const createExpense = useCallback<StoreValue["createExpense"]>(
    async (input) => {
      const row: Expense = {
        id: input.id ?? newId(),
        amount_cents: Math.round(input.amount_cents),
        currency: input.currency ?? household?.currency ?? "EUR",
        description: (input.description ?? "").trim(),
        category_id: input.category_id ?? null,
        spent_on: input.spent_on ?? todayISO(),
        payment_method: input.payment_method ?? "card",
        paid_by: input.paid_by ?? userRef.current?.id ?? "",
        note: input.note ?? null,
        visibility: input.visibility ?? "shared",
        ...base(),
      };
      return (await insertOptimistic("expenses", row)) as Expense;
    },
    [base, household?.currency, insertOptimistic],
  );

  // --- auth / spazio ------------------------------------------------------

  const signIn = useCallback<StoreValue["signIn"]>(
    async (email, password) => {
      await driver.signIn(email, password);
      const u = await driver.getUser();
      if (u) {
        setUser(u);
        setStatus("loading");
        await loadFor(u);
      }
    },
    [driver, loadFor],
  );

  const signUp = useCallback<StoreValue["signUp"]>(
    async (email, password, name) => {
      const res = await driver.signUp(email, password, name);
      if (!res.needsConfirmation) {
        const u = await driver.getUser();
        if (u) {
          setUser(u);
          setStatus("loading");
          await loadFor(u);
        }
      }
      return res;
    },
    [driver, loadFor],
  );

  const signOut = useCallback(async () => {
    await driver.signOut();
    setUser(null);
    setProfile(null);
    setHousehold(null);
    setMembers([]);
    setData(EMPTY_SNAPSHOT);
    setStatus("anonymous");
  }, [driver]);

  const createHousehold = useCallback<StoreValue["createHousehold"]>(
    async (name) => {
      const h = await driver.createHousehold(name);
      setHousehold(h);
      const snapshot = await driver.fetchSnapshot(h.id);
      setData(snapshot);
      if (user) {
        const boot = await driver.loadBootstrap(user);
        setMembers(boot.members);
      }
      setStatus("ready");
    },
    [driver, user],
  );

  const joinHousehold = useCallback<StoreValue["joinHousehold"]>(
    async (code) => {
      const h = await driver.joinHousehold(code);
      setHousehold(h);
      const snapshot = await driver.fetchSnapshot(h.id);
      setData(snapshot);
      if (user) {
        const boot = await driver.loadBootstrap(user);
        setMembers(boot.members);
      }
      setStatus("ready");
    },
    [driver, user],
  );

  const rotateInviteCode = useCallback(async () => {
    if (!household) throw new Error("Nessuno spazio");
    const code = await driver.rotateInviteCode(household.id);
    setHousehold({ ...household, invite_code: code });
    return code;
  }, [driver, household]);

  const saveProfile = useCallback<StoreValue["saveProfile"]>(
    async (patch) => {
      if (!user) return;
      const updated = await driver.updateProfile(user.id, patch);
      setProfile(updated);
      setMembers((ms) => ms.map((m) => (m.id === updated.id ? updated : m)));
    },
    [driver, user],
  );

  const refresh = useCallback(async () => {
    if (!household) return;
    setData(await driver.fetchSnapshot(household.id));
  }, [driver, household]);

  // --- lookup -------------------------------------------------------------

  const member = useCallback(
    (id: UUID | null | undefined) => (id ? (members.find((m) => m.id === id) ?? null) : null),
    [members],
  );

  const memberName = useCallback(
    (id: UUID | null | undefined) => {
      if (!id) return "Nessuno";
      if (id === user?.id) return profile?.display_name || "Tu";
      return member(id)?.display_name ?? "Partner";
    },
    [member, profile?.display_name, user?.id],
  );

  const category = useCallback(
    (id: UUID | null | undefined) => (id ? (data.categories.find((c) => c.id === id) ?? null) : null),
    [data.categories],
  );

  const categoryBySlug = useCallback(
    (slug: string | null | undefined) =>
      slug ? (data.categories.find((c) => c.slug === slug) ?? null) : null,
    [data.categories],
  );

  const value: StoreValue = {
    status,
    driverKind: driver.kind,
    user,
    profile,
    household,
    members,
    data,
    toasts,
    signIn,
    signUp,
    signOut,
    requestPasswordReset: useCallback(
      async (email: string) => {
        const redirect =
          typeof window !== "undefined" ? `${window.location.origin}/update-password` : "";
        await driver.requestPasswordReset(email, redirect);
      },
      [driver],
    ),
    updatePassword: useCallback((pw: string) => driver.updatePassword(pw), [driver]),
    createHousehold,
    joinHousehold,
    rotateInviteCode,
    saveProfile,
    createTask,
    updateTask,
    toggleTask,
    deleteTask: useCallback((id: UUID) => deleteOptimistic("tasks", id), [deleteOptimistic]),
    createEvent,
    updateEvent,
    deleteEvent: useCallback((id: UUID) => deleteOptimistic("events", id), [deleteOptimistic]),
    createNote,
    updateNote: useCallback(
      (id: UUID, patch: Partial<Note>) => updateOptimistic("notes", id, patch),
      [updateOptimistic],
    ),
    deleteNote: useCallback((id: UUID) => deleteOptimistic("notes", id), [deleteOptimistic]),
    createExpense,
    updateExpense: useCallback(
      (id: UUID, patch: Partial<Expense>) => updateOptimistic("expenses", id, patch),
      [updateOptimistic],
    ),
    deleteExpense: useCallback((id: UUID) => deleteOptimistic("expenses", id), [deleteOptimistic]),
    markNotificationSent: useCallback(
      (id: UUID) => updateOptimistic("notifications", id, { sent_at: nowISO() }),
      [updateOptimistic],
    ),
    refresh,
    pushToast,
    dismissToast,
    memberName,
    member,
    category,
    categoryBySlug,
    driver,
  };

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

/**
 * Driver di sviluppo/demo: tutto in `localStorage`.
 *
 * Serve a due cose concrete:
 *  1. l'app si apre e funziona senza configurare nulla;
 *  2. il realtime è simulato con `BroadcastChannel`, quindi due schede del
 *     browser si comportano davvero come due dispositivi.
 *
 * Non è pensato per l'uso reale in due: per quello serve Supabase.
 */

import { addDays, toISODate, todayISO } from "../date";
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
} from "../types";
import {
  DriverError,
  EMPTY_SNAPSHOT,
  type AuthUser,
  type Bootstrap,
  type ChangeEvent,
  type Driver,
  type Snapshot,
  type TableName,
} from "./driver";

const KEY = "casa:db:v1";
const CHANNEL = "casa:sync:v1";

interface LocalUser {
  id: UUID;
  email: string;
  password: string;
}

interface LocalDb {
  users: LocalUser[];
  sessionUserId: UUID | null;
  profiles: Profile[];
  households: Household[];
  members: { household_id: UUID; user_id: UUID; role: string }[];
  tables: Snapshot;
}

const uid = (): UUID =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `id-${Math.random().toString(36).slice(2)}-${Date.now()}`;

const now = () => new Date().toISOString();

function stamp<T extends object>(row: T): T & { created_at: string; updated_at: string } {
  return { created_at: now(), updated_at: now(), ...row } as T & {
    created_at: string;
    updated_at: string;
  };
}

// ---------------------------------------------------------------------------
//  Dataset di esempio
// ---------------------------------------------------------------------------

function seed(): LocalDb {
  const lucaId = uid();
  const martaId = uid();
  const householdId = uid();
  const t = todayISO();
  const tomorrow = toISODate(addDays(new Date(), 1));
  const in3 = toISODate(addDays(new Date(), 3));
  const yesterday = toISODate(addDays(new Date(), -1));
  const twoDaysAgo = toISODate(addDays(new Date(), -2));

  const profile = (id: UUID, name: string, email: string, accent: string): Profile =>
    stamp({ id, display_name: name, email, accent }) as Profile;

  const cat = (slug: string, name: string, icon: string, color: string, position: number): Category =>
    stamp({
      id: uid(),
      household_id: householdId,
      kind: "expense" as const,
      slug,
      name,
      icon,
      color,
      position,
      created_by: lucaId,
    }) as Category;

  const categories: Category[] = [
    cat("casa", "Casa", "home", "#C2703C", 1),
    cat("alimentari", "Spesa alimentare", "shopping-cart", "#5E8C61", 2),
    cat("ristoranti", "Ristoranti", "utensils", "#C25A38", 3),
    cat("trasporti", "Trasporti", "car", "#4E7C9B", 4),
    cat("shopping", "Shopping", "shopping-bag", "#9B6BA8", 5),
    cat("viaggi", "Viaggi", "plane", "#3F8F86", 6),
    cat("salute", "Salute", "heart-pulse", "#C0453C", 7),
    cat("bollette", "Bollette", "receipt", "#7A6BB5", 8),
    cat("regali", "Regali", "gift", "#D08A5B", 9),
    cat("tempolibero", "Tempo libero", "ticket", "#C99A2E", 10),
    cat("altro", "Altro", "tag", "#8A8175", 11),
  ];
  const bySlug = (slug: string) => categories.find((c) => c.slug === slug)!.id;

  const task = (p: Partial<Task> & { title: string; created_by: UUID }): Task =>
    stamp({
      id: uid(),
      household_id: householdId,
      description: null,
      notes: null,
      due_date: null,
      due_time: null,
      status: "todo" as const,
      priority: "normal" as const,
      category_id: null,
      assigned_to: null,
      reminders: [],
      recurrence: null,
      recurring_item_id: null,
      completed_at: null,
      completed_by: null,
      visibility: "shared" as const,
      ...p,
    }) as Task;

  const tasks: Task[] = [
    task({ title: "Comprare il latte", due_date: t, created_by: martaId, category_id: bySlug("alimentari") }),
    task({
      title: "Chiamare il commercialista",
      due_date: t,
      due_time: "11:00",
      priority: "high",
      reminders: [30],
      assigned_to: lucaId,
      created_by: lucaId,
    }),
    task({ title: "Portare fuori il cane", due_date: t, status: "done", completed_at: now(), completed_by: martaId, created_by: martaId }),
    task({ title: "Prenotare ristorante anniversario", due_date: in3, priority: "high", created_by: lucaId }),
    task({
      title: "Rinnovare assicurazione auto",
      due_date: twoDaysAgo,
      priority: "urgent",
      created_by: martaId,
      category_id: bySlug("trasporti"),
    }),
    task({
      title: "Pagare l'affitto",
      due_date: toISODate(new Date(new Date().getFullYear(), new Date().getMonth() + 1, 1)),
      recurrence: { freq: "monthly", interval: 1, bymonthday: 1 },
      reminders: [1440],
      created_by: lucaId,
      category_id: bySlug("casa"),
    }),
    task({ title: "Annaffiare le piante", due_date: tomorrow, recurrence: { freq: "weekly", interval: 1, byweekday: [new Date().getDay()] }, created_by: martaId }),
  ];

  const event = (p: Partial<EventItem> & { title: string; start_date: string; created_by: UUID }): EventItem =>
    stamp({
      id: uid(),
      household_id: householdId,
      description: null,
      location: null,
      start_time: null,
      end_date: null,
      end_time: null,
      all_day: false,
      attendees: [],
      reminders: [],
      recurrence: null,
      recurring_item_id: null,
      color: null,
      visibility: "shared" as const,
      ...p,
    }) as EventItem;

  const events: EventItem[] = [
    event({
      title: "Cena con Marco",
      start_date: t,
      start_time: "20:00",
      end_time: "22:00",
      location: "Da Vito",
      reminders: [60],
      created_by: lucaId,
    }),
    event({ title: "Palestra", start_date: t, start_time: "18:30", end_time: "19:30", created_by: martaId }),
    event({
      title: "Visita dal dentista",
      start_date: in3,
      start_time: "09:30",
      end_time: "10:15",
      reminders: [1440, 60],
      created_by: martaId,
    }),
  ];

  const expense = (
    amount: number,
    description: string,
    slug: string,
    spent_on: string,
    paid_by: UUID,
    payment_method: Expense["payment_method"] = "card",
  ): Expense =>
    stamp({
      id: uid(),
      household_id: householdId,
      amount_cents: amount,
      currency: "EUR",
      description,
      category_id: bySlug(slug),
      spent_on,
      payment_method,
      paid_by,
      note: null,
      visibility: "shared" as const,
      created_by: paid_by,
    }) as Expense;

  const monthStart = new Date(new Date().getFullYear(), new Date().getMonth(), 1);
  const expenses: Expense[] = [
    expense(6420, "Esselunga", "alimentari", t, lucaId),
    expense(350, "Caffè", "ristoranti", t, martaId, "cash"),
    expense(3500, "Benzina", "trasporti", yesterday, martaId, "card"),
    expense(2890, "Pizza da asporto", "ristoranti", yesterday, lucaId),
    expense(1290, "Farmacia", "salute", twoDaysAgo, martaId),
    expense(85000, "Affitto", "casa", toISODate(monthStart), lucaId, "transfer"),
    expense(7830, "Bolletta luce", "bollette", toISODate(addDays(monthStart, 4)), lucaId, "transfer"),
    expense(4210, "Spesa settimanale", "alimentari", toISODate(addDays(monthStart, 6)), martaId),
    expense(2500, "Cinema", "tempolibero", toISODate(addDays(monthStart, 9)), lucaId),
    expense(11990, "Scarpe", "shopping", toISODate(addDays(monthStart, 11)), martaId),
    // mese precedente, per il confronto
    expense(85000, "Affitto", "casa", toISODate(addDays(monthStart, -25)), lucaId, "transfer"),
    expense(15230, "Spesa", "alimentari", toISODate(addDays(monthStart, -18)), martaId),
    expense(6400, "Ristorante", "ristoranti", toISODate(addDays(monthStart, -12)), lucaId),
    expense(4500, "Benzina", "trasporti", toISODate(addDays(monthStart, -6)), martaId),
  ];

  const note = (body: string, created_by: UUID): Note =>
    stamp({
      id: uid(),
      household_id: householdId,
      body,
      pinned: false,
      archived_at: null,
      converted_kind: null,
      converted_id: null,
      visibility: "shared" as const,
      created_by,
    }) as Note;

  const notes: Note[] = [
    note("Controllare volo Londra", lucaId),
    note("Nuova idea per il negozio: consegna a domicilio il sabato", lucaId),
    note("Prenotare ristorante anniversario", martaId),
  ];

  return {
    users: [
      { id: lucaId, email: "luca@casa.app", password: "demo1234" },
      { id: martaId, email: "marta@casa.app", password: "demo1234" },
    ],
    sessionUserId: lucaId,
    profiles: [
      profile(lucaId, "Luca", "luca@casa.app", "clay"),
      profile(martaId, "Marta", "marta@casa.app", "sage"),
    ],
    households: [
      stamp({
        id: householdId,
        name: "Casa",
        invite_code: "CASA26",
        currency: "EUR",
        created_by: lucaId,
      }) as Household,
    ],
    members: [
      { household_id: householdId, user_id: lucaId, role: "owner" },
      { household_id: householdId, user_id: martaId, role: "member" },
    ],
    tables: { categories, tasks, events, notes, expenses, notifications: [] },
  };
}

// ---------------------------------------------------------------------------
//  Driver
// ---------------------------------------------------------------------------

export class LocalDriver implements Driver {
  readonly kind = "local" as const;
  private channel: BroadcastChannel | null = null;
  private authListeners = new Set<(u: AuthUser | null) => void>();

  private read(): LocalDb {
    if (typeof window === "undefined") return { ...seedEmpty() };
    const raw = window.localStorage.getItem(KEY);
    if (!raw) {
      const fresh = seed();
      window.localStorage.setItem(KEY, JSON.stringify(fresh));
      return fresh;
    }
    try {
      const parsed = JSON.parse(raw) as LocalDb;
      parsed.tables = { ...EMPTY_SNAPSHOT, ...parsed.tables };
      return parsed;
    } catch {
      const fresh = seed();
      window.localStorage.setItem(KEY, JSON.stringify(fresh));
      return fresh;
    }
  }

  private write(db: LocalDb) {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(KEY, JSON.stringify(db));
  }

  private broadcast(ev: ChangeEvent) {
    if (typeof window === "undefined") return;
    this.channel ??= new BroadcastChannel(CHANNEL);
    this.channel.postMessage(ev);
  }

  /** Riporta il localStorage al dataset di esempio. */
  reset() {
    if (typeof window === "undefined") return;
    window.localStorage.removeItem(KEY);
    this.read();
  }

  // --- auth ---------------------------------------------------------------

  async getUser(): Promise<AuthUser | null> {
    const db = this.read();
    if (!db.sessionUserId) return null;
    const u = db.users.find((x) => x.id === db.sessionUserId);
    return u ? { id: u.id, email: u.email } : null;
  }

  onAuthChange(cb: (user: AuthUser | null) => void): () => void {
    this.authListeners.add(cb);
    return () => this.authListeners.delete(cb);
  }

  private emitAuth(user: AuthUser | null) {
    for (const cb of this.authListeners) cb(user);
  }

  async signIn(email: string, password: string): Promise<void> {
    const db = this.read();
    const user = db.users.find((u) => u.email.toLowerCase() === email.trim().toLowerCase());
    if (!user || user.password !== password) {
      throw new DriverError("Invalid login credentials");
    }
    db.sessionUserId = user.id;
    this.write(db);
    this.emitAuth({ id: user.id, email: user.email });
  }

  async signUp(email: string, password: string, displayName: string) {
    const db = this.read();
    const clean = email.trim().toLowerCase();
    if (db.users.some((u) => u.email.toLowerCase() === clean)) {
      throw new DriverError("User already registered");
    }
    if (password.length < 6) throw new DriverError("Password should be at least 6 characters");
    const id = uid();
    db.users.push({ id, email: clean, password });
    db.profiles.push(
      stamp({ id, display_name: displayName || clean.split("@")[0], email: clean, accent: "clay" }) as Profile,
    );
    db.sessionUserId = id;
    this.write(db);
    this.emitAuth({ id, email: clean });
    return { needsConfirmation: false };
  }

  async signOut(): Promise<void> {
    const db = this.read();
    db.sessionUserId = null;
    this.write(db);
    this.emitAuth(null);
  }

  async requestPasswordReset(): Promise<void> {
    // in modalità demo non c'è nessuna email da mandare
  }

  async updatePassword(password: string): Promise<void> {
    const db = this.read();
    const user = db.users.find((u) => u.id === db.sessionUserId);
    if (!user) throw new DriverError("not authenticated");
    user.password = password;
    this.write(db);
  }

  // --- spazio -------------------------------------------------------------

  async loadBootstrap(user: AuthUser): Promise<Bootstrap> {
    const db = this.read();
    const profile =
      db.profiles.find((p) => p.id === user.id) ??
      (stamp({ id: user.id, display_name: user.email.split("@")[0], email: user.email, accent: "clay" }) as Profile);
    const membership = db.members.find((m) => m.user_id === user.id);
    const household = membership
      ? (db.households.find((h) => h.id === membership.household_id) ?? null)
      : null;
    const members = household
      ? db.members
          .filter((m) => m.household_id === household.id)
          .map((m) => db.profiles.find((p) => p.id === m.user_id))
          .filter((p): p is Profile => Boolean(p))
      : [profile];
    return { user, profile, household, members };
  }

  async createHousehold(name: string): Promise<Household> {
    const db = this.read();
    const userId = db.sessionUserId;
    if (!userId) throw new DriverError("not authenticated");
    const household = stamp({
      id: uid(),
      name: name.trim() || "Casa",
      invite_code: inviteCode(),
      currency: "EUR",
      created_by: userId,
    }) as Household;
    db.households.push(household);
    db.members.push({ household_id: household.id, user_id: userId, role: "owner" });
    db.tables.categories.push(...defaultCategories(household.id, userId));
    this.write(db);
    return household;
  }

  async joinHousehold(code: string): Promise<Household> {
    const db = this.read();
    const userId = db.sessionUserId;
    if (!userId) throw new DriverError("not authenticated");
    const household = db.households.find(
      (h) => h.invite_code.toUpperCase() === code.trim().toUpperCase(),
    );
    if (!household) throw new DriverError("invalid_invite_code");
    if (!db.members.some((m) => m.household_id === household.id && m.user_id === userId)) {
      db.members.push({ household_id: household.id, user_id: userId, role: "member" });
    }
    this.write(db);
    return household;
  }

  async rotateInviteCode(householdId: UUID): Promise<string> {
    const db = this.read();
    const h = db.households.find((x) => x.id === householdId);
    if (!h) throw new DriverError("not found");
    h.invite_code = inviteCode();
    this.write(db);
    return h.invite_code;
  }

  async updateProfile(userId: UUID, patch: Partial<Profile>): Promise<Profile> {
    const db = this.read();
    const p = db.profiles.find((x) => x.id === userId);
    if (!p) throw new DriverError("not found");
    Object.assign(p, patch, { updated_at: now() });
    this.write(db);
    return p;
  }

  // --- contenuti ----------------------------------------------------------

  async fetchSnapshot(householdId: UUID): Promise<Snapshot> {
    const db = this.read();
    const userId = db.sessionUserId;
    const visible = <T extends { household_id: UUID; visibility?: string; created_by?: UUID | null }>(
      rows: T[],
    ): T[] =>
      rows.filter(
        (r) =>
          r.household_id === householdId &&
          (r.visibility === undefined || r.visibility === "shared" || r.created_by === userId),
      );

    return {
      categories: visible(db.tables.categories),
      tasks: visible(db.tables.tasks),
      events: visible(db.tables.events),
      notes: visible(db.tables.notes),
      expenses: visible(db.tables.expenses),
      notifications: visible(db.tables.notifications),
    };
  }

  async insert<T extends TableName>(table: T, row: Snapshot[T][number]) {
    const db = this.read();
    const complete = { ...(row as object), created_at: now(), updated_at: now() } as Snapshot[T][number];
    (db.tables[table] as unknown[]).push(complete);
    this.write(db);
    const withId = complete as unknown as { id: UUID } & Record<string, unknown>;
    this.broadcast({ table, type: "INSERT", row: withId });
    return complete;
  }

  async update<T extends TableName>(table: T, id: UUID, patch: Partial<Snapshot[T][number]>) {
    const db = this.read();
    const rows = db.tables[table] as unknown as { id: UUID }[];
    const idx = rows.findIndex((r) => r.id === id);
    if (idx === -1) throw new DriverError("not found");
    const updated = { ...rows[idx], ...patch, updated_at: now() };
    rows[idx] = updated;
    this.write(db);
    this.broadcast({
      table,
      type: "UPDATE",
      row: updated as unknown as { id: UUID } & Record<string, unknown>,
    });
    return updated as Snapshot[T][number];
  }

  async remove(table: TableName, id: UUID): Promise<void> {
    const db = this.read();
    const rows = db.tables[table] as unknown as { id: UUID }[];
    const idx = rows.findIndex((r) => r.id === id);
    if (idx >= 0) rows.splice(idx, 1);
    this.write(db);
    this.broadcast({ table, type: "DELETE", row: { id } });
  }

  async clearNotificationsFor(sourceKind: SourceKind, sourceId: UUID): Promise<void> {
    const db = this.read();
    const keep: NotificationRow[] = [];
    for (const n of db.tables.notifications) {
      if (n.source_kind === sourceKind && n.source_id === sourceId && !n.sent_at) {
        this.broadcast({ table: "notifications", type: "DELETE", row: { id: n.id } });
      } else {
        keep.push(n);
      }
    }
    db.tables.notifications = keep;
    this.write(db);
  }

  subscribe(_householdId: UUID, handler: (ev: ChangeEvent) => void): () => void {
    if (typeof window === "undefined") return () => {};
    const ch = new BroadcastChannel(CHANNEL);
    ch.onmessage = (e) => handler(e.data as ChangeEvent);
    return () => ch.close();
  }
}

function inviteCode(): string {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  return Array.from({ length: 6 }, () => alphabet[Math.floor(Math.random() * alphabet.length)]).join("");
}

function defaultCategories(householdId: UUID, userId: UUID): Category[] {
  const defs: [string, string, string, string][] = [
    ["casa", "Casa", "home", "#C2703C"],
    ["alimentari", "Spesa alimentare", "shopping-cart", "#5E8C61"],
    ["ristoranti", "Ristoranti", "utensils", "#C25A38"],
    ["trasporti", "Trasporti", "car", "#4E7C9B"],
    ["shopping", "Shopping", "shopping-bag", "#9B6BA8"],
    ["viaggi", "Viaggi", "plane", "#3F8F86"],
    ["salute", "Salute", "heart-pulse", "#C0453C"],
    ["bollette", "Bollette", "receipt", "#7A6BB5"],
    ["regali", "Regali", "gift", "#D08A5B"],
    ["tempolibero", "Tempo libero", "ticket", "#C99A2E"],
    ["altro", "Altro", "tag", "#8A8175"],
  ];
  return defs.map(([slug, name, icon, color], i) =>
    stamp({
      id: uid(),
      household_id: householdId,
      kind: "expense" as const,
      slug,
      name,
      icon,
      color,
      position: i + 1,
      created_by: userId,
    }) as Category,
  );
}

function seedEmpty(): LocalDb {
  return {
    users: [],
    sessionUserId: null,
    profiles: [],
    households: [],
    members: [],
    tables: { ...EMPTY_SNAPSHOT },
  };
}

export const DEMO_CREDENTIALS = { email: "luca@casa.app", password: "demo1234" };

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

export interface AuthUser {
  id: UUID;
  email: string;
}

export interface Bootstrap {
  user: AuthUser;
  profile: Profile;
  household: Household | null;
  members: Profile[];
}

export interface Snapshot {
  categories: Category[];
  tasks: Task[];
  events: EventItem[];
  notes: Note[];
  expenses: Expense[];
  notifications: NotificationRow[];
}

export const EMPTY_SNAPSHOT: Snapshot = {
  categories: [],
  tasks: [],
  events: [],
  notes: [],
  expenses: [],
  notifications: [],
};

export type TableName = keyof Snapshot;

export interface ChangeEvent {
  table: TableName;
  type: "INSERT" | "UPDATE" | "DELETE";
  /** Riga nuova (INSERT/UPDATE) oppure `{ id }` per DELETE. */
  row: { id: UUID } & Record<string, unknown>;
}

export interface PushSubscriptionPayload {
  endpoint: string;
  p256dh: string;
  auth: string;
  userAgent?: string;
}

/**
 * Tutto l'accesso ai dati passa da qui. Due implementazioni:
 * `SupabaseDriver` (produzione) e `LocalDriver` (demo su localStorage).
 */
export interface Driver {
  readonly kind: "supabase" | "local";

  // --- autenticazione ---
  getUser(): Promise<AuthUser | null>;
  onAuthChange(cb: (user: AuthUser | null) => void): () => void;
  signIn(email: string, password: string): Promise<void>;
  signUp(
    email: string,
    password: string,
    displayName: string,
  ): Promise<{ needsConfirmation: boolean }>;
  signOut(): Promise<void>;
  requestPasswordReset(email: string, redirectTo: string): Promise<void>;
  updatePassword(password: string): Promise<void>;

  // --- spazio condiviso ---
  loadBootstrap(user: AuthUser): Promise<Bootstrap>;
  createHousehold(name: string): Promise<Household>;
  joinHousehold(code: string): Promise<Household>;
  rotateInviteCode(householdId: UUID): Promise<string>;
  updateProfile(userId: UUID, patch: Partial<Profile>): Promise<Profile>;

  // --- contenuti ---
  fetchSnapshot(householdId: UUID): Promise<Snapshot>;
  insert<T extends TableName>(table: T, row: Snapshot[T][number]): Promise<Snapshot[T][number]>;
  update<T extends TableName>(
    table: T,
    id: UUID,
    patch: Partial<Snapshot[T][number]>,
  ): Promise<Snapshot[T][number]>;
  remove(table: TableName, id: UUID): Promise<void>;
  /** Cancella i promemoria pendenti di un elemento, prima di riscriverli. */
  clearNotificationsFor(sourceKind: SourceKind, sourceId: UUID): Promise<void>;

  // --- realtime ---
  subscribe(householdId: UUID, handler: (ev: ChangeEvent) => void): () => void;

  // --- push (opzionale) ---
  savePushSubscription?(
    userId: UUID,
    householdId: UUID,
    sub: PushSubscriptionPayload,
  ): Promise<void>;
}

export class DriverError extends Error {
  constructor(
    message: string,
    readonly code?: string,
  ) {
    super(message);
    this.name = "DriverError";
  }
}

/** Messaggi d'errore in italiano per i casi che l'utente può incontrare. */
export function friendlyError(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err);
  const map: [RegExp, string][] = [
    [/invalid login credentials/i, "Email o password non corretti."],
    [/email not confirmed/i, "Devi prima confermare l'email che ti abbiamo inviato."],
    [/user already registered/i, "Esiste già un account con questa email."],
    [/password should be at least/i, "La password deve avere almeno 6 caratteri."],
    [/invalid_invite_code/i, "Codice di invito non valido."],
    [/rate limit|too many requests/i, "Troppi tentativi, riprova fra qualche minuto."],
    [/failed to fetch|network/i, "Connessione assente. Riprova quando torni online."],
  ];
  for (const [re, msg] of map) if (re.test(raw)) return msg;
  return raw || "Qualcosa è andato storto.";
}

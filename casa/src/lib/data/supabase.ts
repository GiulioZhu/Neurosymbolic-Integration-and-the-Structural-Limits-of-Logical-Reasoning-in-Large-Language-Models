import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { normalizeTime } from "../date";
import type { Household, Profile, SourceKind, UUID } from "../types";
import {
  DriverError,
  type AuthUser,
  type Bootstrap,
  type ChangeEvent,
  type Driver,
  type PushSubscriptionPayload,
  type Snapshot,
  type TableName,
} from "./driver";

export const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
export const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";
export const supabaseConfigured = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

let client: SupabaseClient | null = null;

export function getSupabase(): SupabaseClient {
  if (!supabaseConfigured) throw new DriverError("Supabase non configurato");
  client ??= createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
  });
  return client;
}

/** Postgres restituisce `time` come `HH:MM:SS`: la UI lavora con `HH:mm`. */
function normalizeRow<T extends Record<string, unknown>>(table: TableName, row: T): T {
  if (table === "tasks" && row.due_time) return { ...row, due_time: normalizeTime(row.due_time as string) };
  if (table === "events") {
    return {
      ...row,
      start_time: normalizeTime((row.start_time as string) ?? null),
      end_time: normalizeTime((row.end_time as string) ?? null),
    };
  }
  return row;
}

function unwrap<T>(res: { data: T | null; error: { message: string; code?: string } | null }): T {
  if (res.error) throw new DriverError(res.error.message, res.error.code);
  if (res.data === null) throw new DriverError("Nessun dato restituito");
  return res.data;
}

export class SupabaseDriver implements Driver {
  readonly kind = "supabase" as const;
  private get sb() {
    return getSupabase();
  }

  // --- auth ---------------------------------------------------------------

  async getUser(): Promise<AuthUser | null> {
    const { data } = await this.sb.auth.getSession();
    const u = data.session?.user;
    return u ? { id: u.id, email: u.email ?? "" } : null;
  }

  onAuthChange(cb: (user: AuthUser | null) => void): () => void {
    const { data } = this.sb.auth.onAuthStateChange((_event, session) => {
      const u = session?.user;
      cb(u ? { id: u.id, email: u.email ?? "" } : null);
    });
    return () => data.subscription.unsubscribe();
  }

  async signIn(email: string, password: string): Promise<void> {
    const { error } = await this.sb.auth.signInWithPassword({ email: email.trim(), password });
    if (error) throw new DriverError(error.message, error.code);
  }

  async signUp(email: string, password: string, displayName: string) {
    const { data, error } = await this.sb.auth.signUp({
      email: email.trim(),
      password,
      options: { data: { display_name: displayName } },
    });
    if (error) throw new DriverError(error.message, error.code);
    return { needsConfirmation: !data.session };
  }

  async signOut(): Promise<void> {
    await this.sb.auth.signOut();
  }

  async requestPasswordReset(email: string, redirectTo: string): Promise<void> {
    const { error } = await this.sb.auth.resetPasswordForEmail(email.trim(), { redirectTo });
    if (error) throw new DriverError(error.message, error.code);
  }

  async updatePassword(password: string): Promise<void> {
    const { error } = await this.sb.auth.updateUser({ password });
    if (error) throw new DriverError(error.message, error.code);
  }

  // --- spazio -------------------------------------------------------------

  async loadBootstrap(user: AuthUser): Promise<Bootstrap> {
    const profileRes = await this.sb.from("profiles").select("*").eq("id", user.id).maybeSingle();
    if (profileRes.error) throw new DriverError(profileRes.error.message);

    // il trigger crea il profilo, ma se per qualsiasi ragione manca lo creiamo qui
    let profile = profileRes.data as Profile | null;
    if (!profile) {
      profile = unwrap(
        await this.sb
          .from("profiles")
          .insert({ id: user.id, display_name: user.email.split("@")[0], email: user.email })
          .select()
          .single(),
      ) as Profile;
    }

    const membership = await this.sb
      .from("household_members")
      .select("household_id, role")
      .eq("user_id", user.id)
      .order("created_at", { ascending: true })
      .limit(1)
      .maybeSingle();
    if (membership.error) throw new DriverError(membership.error.message);

    if (!membership.data) return { user, profile, household: null, members: [profile] };

    const householdId = (membership.data as { household_id: UUID }).household_id;
    const household = unwrap(
      await this.sb.from("households").select("*").eq("id", householdId).single(),
    ) as Household;

    const memberIds = unwrap(
      await this.sb.from("household_members").select("user_id").eq("household_id", householdId),
    ) as { user_id: UUID }[];

    const members = unwrap(
      await this.sb
        .from("profiles")
        .select("*")
        .in("id", memberIds.map((m) => m.user_id)),
    ) as Profile[];

    return { user, profile, household, members };
  }

  async createHousehold(name: string): Promise<Household> {
    const { data, error } = await this.sb.rpc("create_household", { p_name: name });
    if (error) throw new DriverError(error.message, error.code);
    return data as Household;
  }

  async joinHousehold(code: string): Promise<Household> {
    const { data, error } = await this.sb.rpc("join_household", { p_code: code });
    if (error) throw new DriverError(error.message, error.code);
    return data as Household;
  }

  async rotateInviteCode(householdId: UUID): Promise<string> {
    const { data, error } = await this.sb.rpc("rotate_invite_code", { p_household: householdId });
    if (error) throw new DriverError(error.message, error.code);
    return data as string;
  }

  async updateProfile(userId: UUID, patch: Partial<Profile>): Promise<Profile> {
    return unwrap(
      await this.sb.from("profiles").update(patch).eq("id", userId).select().single(),
    ) as Profile;
  }

  // --- contenuti ----------------------------------------------------------

  async fetchSnapshot(householdId: UUID): Promise<Snapshot> {
    const horizon = new Date();
    horizon.setFullYear(horizon.getFullYear() - 1);
    const since = horizon.toISOString().slice(0, 10);

    const [categories, tasks, events, notes, expenses, notifications] = await Promise.all([
      this.sb.from("categories").select("*").eq("household_id", householdId).order("position"),
      this.sb.from("tasks").select("*").eq("household_id", householdId),
      this.sb.from("events").select("*").eq("household_id", householdId).gte("start_date", since),
      this.sb.from("notes").select("*").eq("household_id", householdId).order("created_at", { ascending: false }),
      this.sb.from("expenses").select("*").eq("household_id", householdId).gte("spent_on", since),
      this.sb.from("notifications").select("*").eq("household_id", householdId).is("read_at", null),
    ]);

    for (const res of [categories, tasks, events, notes, expenses, notifications]) {
      if (res.error) throw new DriverError(res.error.message);
    }

    const norm = <T extends TableName>(table: T, rows: unknown[]) =>
      rows.map((r) => normalizeRow(table, r as Record<string, unknown>)) as unknown as Snapshot[T];

    return {
      categories: norm("categories", categories.data ?? []),
      tasks: norm("tasks", tasks.data ?? []),
      events: norm("events", events.data ?? []),
      notes: norm("notes", notes.data ?? []),
      expenses: norm("expenses", expenses.data ?? []),
      notifications: norm("notifications", notifications.data ?? []),
    };
  }

  async insert<T extends TableName>(table: T, row: Snapshot[T][number]) {
    const payload = { ...(row as unknown as Record<string, unknown>) };
    // created_at / updated_at li mette il database
    delete payload.created_at;
    delete payload.updated_at;
    const data = unwrap(await this.sb.from(table).insert(payload).select().single()) as Record<
      string,
      unknown
    >;
    return normalizeRow(table, data) as unknown as Snapshot[T][number];
  }

  async update<T extends TableName>(table: T, id: UUID, patch: Partial<Snapshot[T][number]>) {
    const payload = { ...(patch as unknown as Record<string, unknown>) };
    delete payload.created_at;
    delete payload.updated_at;
    const data = unwrap(
      await this.sb.from(table).update(payload).eq("id", id).select().single(),
    ) as Record<string, unknown>;
    return normalizeRow(table, data) as unknown as Snapshot[T][number];
  }

  async remove(table: TableName, id: UUID): Promise<void> {
    const { error } = await this.sb.from(table).delete().eq("id", id);
    if (error) throw new DriverError(error.message, error.code);
  }

  async clearNotificationsFor(sourceKind: SourceKind, sourceId: UUID): Promise<void> {
    const { error } = await this.sb
      .from("notifications")
      .delete()
      .eq("source_kind", sourceKind)
      .eq("source_id", sourceId)
      .is("sent_at", null);
    if (error) throw new DriverError(error.message, error.code);
  }

  // --- realtime -----------------------------------------------------------

  subscribe(householdId: UUID, handler: (ev: ChangeEvent) => void): () => void {
    const tables: TableName[] = ["tasks", "events", "notes", "expenses", "notifications", "categories"];
    const channel = this.sb.channel(`household:${householdId}`);

    for (const table of tables) {
      channel.on(
        "postgres_changes",
        { event: "*", schema: "public", table, filter: `household_id=eq.${householdId}` },
        (payload) => {
          const type = payload.eventType as ChangeEvent["type"];
          const raw = (type === "DELETE" ? payload.old : payload.new) as Record<string, unknown>;
          if (!raw?.id) return;
          handler({
            table,
            type,
            row: normalizeRow(table, raw) as unknown as { id: UUID } & Record<string, unknown>,
          });
        },
      );
    }

    channel.subscribe();
    return () => {
      void this.sb.removeChannel(channel);
    };
  }

  // --- push ---------------------------------------------------------------

  async savePushSubscription(
    userId: UUID,
    householdId: UUID,
    sub: PushSubscriptionPayload,
  ): Promise<void> {
    const { error } = await this.sb.from("push_subscriptions").upsert(
      {
        user_id: userId,
        household_id: householdId,
        endpoint: sub.endpoint,
        p256dh: sub.p256dh,
        auth: sub.auth,
        user_agent: sub.userAgent ?? null,
      },
      { onConflict: "endpoint" },
    );
    if (error) throw new DriverError(error.message, error.code);
  }
}

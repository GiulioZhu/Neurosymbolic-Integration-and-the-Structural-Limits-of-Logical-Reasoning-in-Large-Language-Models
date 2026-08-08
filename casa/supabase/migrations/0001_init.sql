-- =====================================================================
--  Casa — schema iniziale
--  Postgres / Supabase.  Idempotente: si può rieseguire senza danni.
-- =====================================================================

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------
--  Tipi
-- ---------------------------------------------------------------------
do $$ begin
  create type visibility_t   as enum ('shared', 'private');
exception when duplicate_object then null; end $$;

do $$ begin
  create type task_status_t  as enum ('todo', 'doing', 'done');
exception when duplicate_object then null; end $$;

do $$ begin
  create type priority_t     as enum ('low', 'normal', 'high', 'urgent');
exception when duplicate_object then null; end $$;

do $$ begin
  create type category_kind_t as enum ('expense', 'task');
exception when duplicate_object then null; end $$;

do $$ begin
  create type member_role_t  as enum ('owner', 'member');
exception when duplicate_object then null; end $$;

do $$ begin
  create type source_kind_t  as enum ('task', 'event', 'expense', 'note');
exception when duplicate_object then null; end $$;


-- ---------------------------------------------------------------------
--  Utility: updated_at automatico
-- ---------------------------------------------------------------------
create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;


-- ---------------------------------------------------------------------
--  Profiles  (1:1 con auth.users)
-- ---------------------------------------------------------------------
create table if not exists public.profiles (
  id           uuid primary key references auth.users(id) on delete cascade,
  display_name text        not null default '',
  email        text,
  accent       text        not null default 'clay',   -- colore avatar
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, display_name, email)
  values (
    new.id,
    coalesce(new.raw_user_meta_data ->> 'display_name', split_part(new.email, '@', 1)),
    new.email
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();


-- ---------------------------------------------------------------------
--  Households (spazi condivisi) e membri
-- ---------------------------------------------------------------------
create table if not exists public.households (
  id          uuid primary key default gen_random_uuid(),
  name        text        not null,
  invite_code text        not null unique,
  currency    text        not null default 'EUR',
  created_by  uuid        not null references auth.users(id) on delete cascade,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create table if not exists public.household_members (
  household_id uuid        not null references public.households(id) on delete cascade,
  user_id      uuid        not null references auth.users(id)        on delete cascade,
  role         member_role_t not null default 'member',
  created_at   timestamptz not null default now(),
  primary key (household_id, user_id)
);

create index if not exists household_members_user_idx on public.household_members(user_id);


-- ---------------------------------------------------------------------
--  Helper SECURITY DEFINER usati dalle policy.
--  Servono a evitare la ricorsione infinita delle policy su
--  household_members (una policy che interroga la tabella che protegge).
-- ---------------------------------------------------------------------
create or replace function public.is_household_member(hid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1 from public.household_members m
    where m.household_id = hid and m.user_id = auth.uid()
  );
$$;

create or replace function public.shares_household_with(uid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select uid = auth.uid() or exists (
    select 1
    from public.household_members me
    join public.household_members other using (household_id)
    where me.user_id = auth.uid() and other.user_id = uid
  );
$$;


-- ---------------------------------------------------------------------
--  Categorie  (seed per household, personalizzabili)
-- ---------------------------------------------------------------------
create table if not exists public.categories (
  id           uuid primary key default gen_random_uuid(),
  household_id uuid        not null references public.households(id) on delete cascade,
  kind         category_kind_t not null default 'expense',
  slug         text        not null,
  name         text        not null,
  icon         text        not null default 'tag',
  color        text        not null default '#8A8175',
  position     int         not null default 0,
  created_by   uuid        references auth.users(id) on delete set null,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now(),
  unique (household_id, kind, slug)
);


-- ---------------------------------------------------------------------
--  Serie ricorrenti
--  La regola vive qui; ogni occorrenza generata la referenzia.
--  `rule` = { freq: 'daily'|'weekly'|'monthly'|'custom',
--             interval: int, byweekday: int[], bymonthday: int,
--             until: date|null }
-- ---------------------------------------------------------------------
create table if not exists public.recurring_items (
  id            uuid primary key default gen_random_uuid(),
  household_id  uuid        not null references public.households(id) on delete cascade,
  source_kind   source_kind_t not null,
  rule          jsonb       not null,
  template      jsonb       not null default '{}'::jsonb,
  active        boolean     not null default true,
  created_by    uuid        not null references auth.users(id) on delete cascade,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);


-- ---------------------------------------------------------------------
--  Tasks
-- ---------------------------------------------------------------------
create table if not exists public.tasks (
  id                uuid primary key default gen_random_uuid(),
  household_id      uuid        not null references public.households(id) on delete cascade,
  title             text        not null,
  description       text,
  notes             text,
  due_date          date,
  due_time          time,
  status            task_status_t not null default 'todo',
  priority          priority_t    not null default 'normal',
  category_id       uuid        references public.categories(id) on delete set null,
  assigned_to       uuid        references auth.users(id) on delete set null,
  reminders         int[]       not null default '{}',   -- minuti PRIMA della scadenza
  recurrence        jsonb,                               -- copia denormalizzata della regola
  recurring_item_id uuid        references public.recurring_items(id) on delete set null,
  completed_at      timestamptz,
  completed_by      uuid        references auth.users(id) on delete set null,
  visibility        visibility_t not null default 'shared',
  created_by        uuid        not null references auth.users(id) on delete cascade,
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now()
);

create index if not exists tasks_household_due_idx on public.tasks(household_id, due_date);
create index if not exists tasks_household_status_idx on public.tasks(household_id, status);


-- ---------------------------------------------------------------------
--  Events
-- ---------------------------------------------------------------------
create table if not exists public.events (
  id                uuid primary key default gen_random_uuid(),
  household_id      uuid        not null references public.households(id) on delete cascade,
  title             text        not null,
  description       text,
  location          text,
  start_date        date        not null,
  start_time        time,
  end_date          date,
  end_time          time,
  all_day           boolean     not null default false,
  attendees         uuid[]      not null default '{}',
  reminders         int[]       not null default '{}',
  recurrence        jsonb,
  recurring_item_id uuid        references public.recurring_items(id) on delete set null,
  color             text,
  visibility        visibility_t not null default 'shared',
  created_by        uuid        not null references auth.users(id) on delete cascade,
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now()
);

create index if not exists events_household_start_idx on public.events(household_id, start_date);


-- ---------------------------------------------------------------------
--  Notes / Inbox
-- ---------------------------------------------------------------------
create table if not exists public.notes (
  id             uuid primary key default gen_random_uuid(),
  household_id   uuid        not null references public.households(id) on delete cascade,
  body           text        not null,
  pinned         boolean     not null default false,
  archived_at    timestamptz,
  converted_kind source_kind_t,
  converted_id   uuid,
  visibility     visibility_t not null default 'shared',
  created_by     uuid        not null references auth.users(id) on delete cascade,
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now()
);

create index if not exists notes_household_idx on public.notes(household_id, created_at desc);


-- ---------------------------------------------------------------------
--  Expenses
-- ---------------------------------------------------------------------
create table if not exists public.expenses (
  id             uuid primary key default gen_random_uuid(),
  household_id   uuid        not null references public.households(id) on delete cascade,
  amount_cents   bigint      not null check (amount_cents >= 0),
  currency       text        not null default 'EUR',
  description    text        not null default '',
  category_id    uuid        references public.categories(id) on delete set null,
  spent_on       date        not null default current_date,
  payment_method text        not null default 'card',   -- card | cash | transfer | other
  paid_by        uuid        not null references auth.users(id) on delete cascade,
  note           text,
  visibility     visibility_t not null default 'shared',
  created_by     uuid        not null references auth.users(id) on delete cascade,
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now()
);

create index if not exists expenses_household_date_idx on public.expenses(household_id, spent_on desc);


-- ---------------------------------------------------------------------
--  Notifications  (promemoria materializzati)
-- ---------------------------------------------------------------------
create table if not exists public.notifications (
  id            uuid primary key default gen_random_uuid(),
  household_id  uuid        not null references public.households(id) on delete cascade,
  user_id       uuid        references auth.users(id) on delete cascade, -- null = tutti i membri
  source_kind   source_kind_t not null,
  source_id     uuid        not null,
  title         text        not null,
  body          text,
  scheduled_for timestamptz not null,
  sent_at       timestamptz,
  read_at       timestamptz,
  visibility    visibility_t not null default 'shared',
  created_by    uuid        not null references auth.users(id) on delete cascade,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

create index if not exists notifications_due_idx on public.notifications(scheduled_for) where sent_at is null;
create index if not exists notifications_source_idx on public.notifications(source_kind, source_id);


-- ---------------------------------------------------------------------
--  Push subscriptions (Web Push / VAPID)
-- ---------------------------------------------------------------------
create table if not exists public.push_subscriptions (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid        not null references auth.users(id) on delete cascade,
  household_id uuid        not null references public.households(id) on delete cascade,
  endpoint     text        not null unique,
  p256dh       text        not null,
  auth         text        not null,
  user_agent   text,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);


-- ---------------------------------------------------------------------
--  Trigger updated_at su tutte le tabelle
-- ---------------------------------------------------------------------
do $$
declare t text;
begin
  foreach t in array array[
    'profiles','households','categories','recurring_items',
    'tasks','events','notes','expenses','notifications','push_subscriptions'
  ] loop
    execute format('drop trigger if exists touch_%1$s on public.%1$s', t);
    execute format(
      'create trigger touch_%1$s before update on public.%1$s
       for each row execute function public.touch_updated_at()', t);
  end loop;
end $$;


-- ---------------------------------------------------------------------
--  RPC: creare uno spazio (+ seed categorie) e unirsi a uno spazio
-- ---------------------------------------------------------------------
create or replace function public.generate_invite_code()
returns text
language plpgsql
as $$
declare
  alphabet text := 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; -- niente I,O,0,1
  code text;
  i int;
begin
  loop
    code := '';
    for i in 1..6 loop
      code := code || substr(alphabet, 1 + floor(random() * length(alphabet))::int, 1);
    end loop;
    exit when not exists (select 1 from public.households h where h.invite_code = code);
  end loop;
  return code;
end;
$$;

create or replace function public.create_household(p_name text)
returns public.households
language plpgsql
security definer
set search_path = public
as $$
declare
  h public.households;
  uid uuid := auth.uid();
begin
  if uid is null then
    raise exception 'not authenticated';
  end if;

  insert into public.households (name, invite_code, created_by)
  values (coalesce(nullif(trim(p_name), ''), 'Casa'), public.generate_invite_code(), uid)
  returning * into h;

  insert into public.household_members (household_id, user_id, role)
  values (h.id, uid, 'owner');

  insert into public.categories (household_id, kind, slug, name, icon, color, position, created_by)
  values
    (h.id, 'expense', 'casa',       'Casa',            'home',        '#C2703C',  1, uid),
    (h.id, 'expense', 'alimentari', 'Spesa alimentare','shopping-cart','#5E8C61', 2, uid),
    (h.id, 'expense', 'ristoranti', 'Ristoranti',      'utensils',    '#C25A38',  3, uid),
    (h.id, 'expense', 'trasporti',  'Trasporti',       'car',         '#4E7C9B',  4, uid),
    (h.id, 'expense', 'shopping',   'Shopping',        'shopping-bag','#9B6BA8',  5, uid),
    (h.id, 'expense', 'viaggi',     'Viaggi',          'plane',       '#3F8F86',  6, uid),
    (h.id, 'expense', 'salute',     'Salute',          'heart-pulse', '#C0453C',  7, uid),
    (h.id, 'expense', 'bollette',   'Bollette',        'receipt',     '#7A6BB5',  8, uid),
    (h.id, 'expense', 'regali',     'Regali',          'gift',        '#D08A5B',  9, uid),
    (h.id, 'expense', 'tempolibero','Tempo libero',    'ticket',      '#C99A2E', 10, uid),
    (h.id, 'expense', 'altro',      'Altro',           'tag',         '#8A8175', 11, uid)
  on conflict do nothing;

  return h;
end;
$$;

create or replace function public.join_household(p_code text)
returns public.households
language plpgsql
security definer
set search_path = public
as $$
declare
  h public.households;
  uid uuid := auth.uid();
begin
  if uid is null then
    raise exception 'not authenticated';
  end if;

  select * into h from public.households
  where invite_code = upper(trim(p_code));

  if h.id is null then
    raise exception 'invalid_invite_code';
  end if;

  insert into public.household_members (household_id, user_id, role)
  values (h.id, uid, 'member')
  on conflict do nothing;

  return h;
end;
$$;

create or replace function public.rotate_invite_code(p_household uuid)
returns text
language plpgsql
security definer
set search_path = public
as $$
declare code text;
begin
  if not public.is_household_member(p_household) then
    raise exception 'forbidden';
  end if;
  code := public.generate_invite_code();
  update public.households set invite_code = code where id = p_household;
  return code;
end;
$$;


-- =====================================================================
--  ROW LEVEL SECURITY
--  Regola generale per i contenuti:
--    leggibile  <=>  sono membro dello spazio
--                    AND (la riga è condivisa OR l'ho creata io)
-- =====================================================================

alter table public.profiles           enable row level security;
alter table public.households         enable row level security;
alter table public.household_members  enable row level security;
alter table public.categories         enable row level security;
alter table public.recurring_items    enable row level security;
alter table public.tasks              enable row level security;
alter table public.events             enable row level security;
alter table public.notes              enable row level security;
alter table public.expenses           enable row level security;
alter table public.notifications      enable row level security;
alter table public.push_subscriptions enable row level security;

-- profiles ------------------------------------------------------------
drop policy if exists profiles_select on public.profiles;
create policy profiles_select on public.profiles
  for select using (public.shares_household_with(id));

drop policy if exists profiles_update on public.profiles;
create policy profiles_update on public.profiles
  for update using (id = auth.uid()) with check (id = auth.uid());

drop policy if exists profiles_insert on public.profiles;
create policy profiles_insert on public.profiles
  for insert with check (id = auth.uid());

-- households ----------------------------------------------------------
drop policy if exists households_select on public.households;
create policy households_select on public.households
  for select using (public.is_household_member(id));

drop policy if exists households_update on public.households;
create policy households_update on public.households
  for update using (public.is_household_member(id))
  with check (public.is_household_member(id));

-- l'inserimento passa da create_household() (security definer)

-- household_members ---------------------------------------------------
drop policy if exists members_select on public.household_members;
create policy members_select on public.household_members
  for select using (public.is_household_member(household_id));

drop policy if exists members_delete on public.household_members;
create policy members_delete on public.household_members
  for delete using (user_id = auth.uid());

-- categories ----------------------------------------------------------
drop policy if exists categories_all on public.categories;
create policy categories_all on public.categories
  for all using (public.is_household_member(household_id))
  with check (public.is_household_member(household_id));

-- recurring_items -----------------------------------------------------
drop policy if exists recurring_all on public.recurring_items;
create policy recurring_all on public.recurring_items
  for all using (public.is_household_member(household_id))
  with check (public.is_household_member(household_id) and created_by = auth.uid());

-- contenuti con visibility -------------------------------------------
do $$
declare t text;
begin
  foreach t in array array['tasks','events','notes','expenses','notifications'] loop
    execute format('drop policy if exists %1$s_select on public.%1$s', t);
    execute format($p$
      create policy %1$s_select on public.%1$s
        for select using (
          public.is_household_member(household_id)
          and (visibility = 'shared' or created_by = auth.uid())
        )$p$, t);

    execute format('drop policy if exists %1$s_insert on public.%1$s', t);
    execute format($p$
      create policy %1$s_insert on public.%1$s
        for insert with check (
          public.is_household_member(household_id) and created_by = auth.uid()
        )$p$, t);

    execute format('drop policy if exists %1$s_update on public.%1$s', t);
    execute format($p$
      create policy %1$s_update on public.%1$s
        for update using (
          public.is_household_member(household_id)
          and (visibility = 'shared' or created_by = auth.uid())
        ) with check (
          public.is_household_member(household_id)
        )$p$, t);

    execute format('drop policy if exists %1$s_delete on public.%1$s', t);
    execute format($p$
      create policy %1$s_delete on public.%1$s
        for delete using (
          public.is_household_member(household_id)
          and (visibility = 'shared' or created_by = auth.uid())
        )$p$, t);
  end loop;
end $$;

-- push_subscriptions --------------------------------------------------
drop policy if exists push_all on public.push_subscriptions;
create policy push_all on public.push_subscriptions
  for all using (user_id = auth.uid()) with check (user_id = auth.uid());


-- ---------------------------------------------------------------------
--  Realtime
-- ---------------------------------------------------------------------
do $$
declare t text;
begin
  foreach t in array array[
    'tasks','events','notes','expenses','notifications','categories',
    'households','household_members','profiles'
  ] loop
    begin
      execute format('alter publication supabase_realtime add table public.%I', t);
    exception when duplicate_object then null;
              when undefined_object then null;
    end;
  end loop;
end $$;

-- le righe aggiornate devono includere i valori vecchi, così i client
-- possono rimuovere dalla cache un elemento che diventa privato
alter table public.tasks    replica identity full;
alter table public.events   replica identity full;
alter table public.notes    replica identity full;
alter table public.expenses replica identity full;

# Supabase — come applicare lo schema

Lo schema completo sta in un unico file:
[`migrations/0001_init.sql`](migrations/0001_init.sql). È **idempotente**: puoi
rieseguirlo senza rompere nulla.

## Via dashboard (il modo più rapido)

1. Apri il progetto su [supabase.com](https://supabase.com).
2. **SQL Editor → New query**.
3. Incolla tutto il contenuto di `migrations/0001_init.sql` e premi **Run**.

## Via CLI

```bash
npx supabase link --project-ref <ref-del-progetto>
npx supabase db push
```

## Cosa crea

* **Tabelle** — `profiles`, `households`, `household_members`, `categories`,
  `tasks`, `events`, `notes`, `expenses`, `notifications`, `recurring_items`,
  `push_subscriptions`.
* **Trigger** — creazione automatica del profilo alla registrazione,
  `updated_at` su ogni tabella.
* **RPC** — `create_household()`, `join_household()`, `rotate_invite_code()`
  (`security definer`, perché devono attraversare il confine dello spazio in
  modo controllato).
* **RLS su tutto**, con la stessa regola per i contenuti: sei membro dello
  spazio **e** la riga è condivisa **oppure** l'hai creata tu.
* **Realtime** attivo sulle tabelle di contenuto, con `replica identity full`
  così i client possono togliere dalla cache un elemento diventato privato.

## Impostazioni consigliate del progetto

**Authentication → Providers → Email**: tieni attiva la conferma via email in
produzione (l'app gestisce già la schermata "controlla la posta"). In sviluppo
puoi disattivarla per fare prima.

**Authentication → URL Configuration**: aggiungi agli URL di redirect

```
http://localhost:3000/update-password
https://<il-tuo-dominio>/update-password
```

altrimenti il link di recupero password non torna nell'app.

## Verifica rapida

Dopo aver applicato lo schema:

```sql
-- deve elencare 11 tabelle
select tablename from pg_tables where schemaname = 'public' order by 1;

-- ogni tabella deve avere rowsecurity = true
select tablename, rowsecurity from pg_tables where schemaname = 'public';
```

Poi registra due utenti dall'app: il primo crea lo spazio, il secondo entra con
il codice di invito. Se entrambi vedono le stesse attività, è tutto a posto.

## Reset (solo in sviluppo)

```sql
-- ATTENZIONE: cancella tutti i dati dell'app
drop schema public cascade;
create schema public;
grant usage on schema public to anon, authenticated, service_role;
grant all on all tables in schema public to anon, authenticated, service_role;
```

Poi riesegui la migrazione. Gli account in `auth.users` restano: eliminali dalla
sezione Authentication se vuoi ripartire davvero da zero.

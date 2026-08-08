# Casa — Database

Schema completo in [`supabase/migrations/0001_init.sql`](../supabase/migrations/0001_init.sql).
Qui il modello, le convenzioni e il ragionamento dietro le scelte.

## 1. Convenzioni comuni

Ogni tabella di contenuto porta sempre:

| Colonna | Tipo | Significato |
|---|---|---|
| `id` | `uuid` | chiave primaria, generata dal DB |
| `household_id` | `uuid` | lo spazio condiviso a cui appartiene la riga |
| `created_by` | `uuid` | chi l'ha creata (mostrato nella UI) |
| `created_at` | `timestamptz` | default `now()` |
| `updated_at` | `timestamptz` | aggiornato da trigger `touch_updated_at()` |
| `visibility` | `visibility_t` | `shared` \| `private` |

E dove ha senso: `assigned_to`, `reminders`, `recurrence`, `recurring_item_id`.

## 2. Diagramma

```
auth.users ──1:1── profiles
     │
     └──< household_members >── households
                                    │
        ┌───────────────┬───────────┼──────────────┬──────────────┐
        │               │           │              │              │
    categories      tasks        events          notes        expenses
        │               │           │                             │
        │               └──┬────────┘                             │
        │                  │                                      │
        │           recurring_items                               │
        │                                                         │
        └──────────────────┬──────────────────────────────────────┘
                           │
                    notifications          push_subscriptions
```

## 3. Tabelle

### `profiles`
Estende `auth.users` (che non è modificabile). Creata automaticamente dal trigger
`on_auth_user_created`. Contiene `display_name`, `email`, `accent` (colore
dell'avatar). È l'unica cosa che i due partner vedono l'uno dell'altro.

### `households`
Lo spazio condiviso. `invite_code` è una stringa di 6 caratteri da un alfabeto
senza caratteri ambigui (`I`, `O`, `0`, `1`), rigenerabile con
`rotate_invite_code()`.

### `household_members`
Tabella ponte, PK composta `(household_id, user_id)`, con `role` (`owner`/`member`).
Lo schema supporta N membri e N spazi per utente: l'MVP ne mostra uno solo, ma
non c'è nulla da migrare per aprirlo a più persone.

### `categories`
Seed di 11 categorie di spesa inserite alla creazione dello spazio (Casa, Spesa
alimentare, Ristoranti, Trasporti, Shopping, Viaggi, Salute, Bollette, Regali,
Tempo libero, Altro), ognuna con `icon` e `color`. Sono righe normali, quindi
rinominabili e estendibili. `kind` distingue categorie spesa da categorie task.

### `tasks`
Il cuore. `due_date` e `due_time` sono separati perché "domani" senza orario è un
caso frequentissimo e un `timestamptz` costringerebbe a inventare un'ora.
`status` (`todo`/`doing`/`done`), `priority` (`low`/`normal`/`high`/`urgent`),
`reminders int[]` = minuti *prima* della scadenza, `completed_at`/`completed_by`
per sapere chi ha spuntato cosa.

### `events`
`start_date`/`start_time` + `end_date`/`end_time`, `all_day`, `location`,
`attendees uuid[]`, stessi `reminders` e `recurrence` delle task.

### `notes`
L'Inbox. Solo `body` + `pinned`. Quando una nota diventa altro, si valorizzano
`converted_kind` e `converted_id` e la nota esce dall'Inbox attiva senza perdere
la traccia.

### `expenses`
`amount_cents bigint` — **mai float per il denaro**. `spent_on date`,
`payment_method`, `paid_by` (che può essere diverso da `created_by`: posso
registrare io una spesa pagata da mia moglie), `category_id`.

### `notifications`
Promemoria *materializzati*: una riga per ogni singolo avviso da mandare, con
`scheduled_for`, `sent_at`, `read_at`, e `source_kind`/`source_id` che puntano
alla task o all'evento. Quando si salva un elemento con promemoria, il client
cancella le notifiche pendenti di quella sorgente e le riscrive. Così sia lo
scheduler in-app sia il cron push leggono dalla stessa, semplice, coda.

### `recurring_items`
Registro delle serie: `rule` jsonb + `template`. Ogni occorrenza generata punta
alla serie con `recurring_item_id`, e porta anche una **copia denormalizzata**
della regola in `recurrence`. La duplicazione è voluta: rende la lettura di una
singola task autosufficiente (nessun join per mostrare "ogni lunedì"), mentre la
serie serve per operazioni sull'intero gruppo.

Formato della regola:

```jsonc
{
  "freq": "daily" | "weekly" | "monthly",
  "interval": 1,          // ogni N giorni/settimane/mesi
  "byweekday": [1,3,5],   // 0 = domenica … 6 = sabato (solo weekly)
  "bymonthday": 1,        // giorno del mese (solo monthly)
  "until": "2027-01-01"   // opzionale
}
```

`"Pagare affitto ogni giorno 1 del mese"` →
`{ "freq": "monthly", "interval": 1, "bymonthday": 1 }`.

### `push_subscriptions`
Endpoint Web Push per dispositivo, uno per browser installato.

## 4. Row Level Security

RLS è attiva su **tutte** le tabelle. La regola per i contenuti è una sola,
applicata identicamente a `tasks`, `events`, `notes`, `expenses`, `notifications`:

```sql
using (
  public.is_household_member(household_id)
  and (visibility = 'shared' or created_by = auth.uid())
)
```

Due dettagli che evitano problemi noti:

1. **Niente ricorsione nelle policy.** `is_household_member()` è
   `security definer`: se una policy su `household_members` interrogasse
   `household_members` direttamente, Postgres andrebbe in loop.
2. **`replica identity full`** su tasks/events/notes/expenses, così gli eventi
   realtime di UPDATE portano anche i valori vecchi e un client può rimuovere
   dalla cache un elemento che è appena diventato `private`.

Le operazioni che devono attraversare il confine dello spazio (creare uno spazio,
entrare con un codice) passano da funzioni `security definer` con validazione
esplicita: `create_household()`, `join_household()`, `rotate_invite_code()`.

## 5. Query tipiche

```sql
-- tutto quello che devo caricare all'avvio (uno round-trip per tabella)
select * from tasks    where household_id = $1;
select * from events   where household_id = $1 and start_date >= $2 - interval '1 year';
select * from expenses where household_id = $1 and spent_on  >= $2 - interval '1 year';

-- promemoria da spedire adesso (usato dal cron push)
select * from notifications
where sent_at is null and scheduled_for <= now()
order by scheduled_for limit 200;
```

Il resto (dashboard, calendario, totali, ricerca) è calcolato in memoria sul
client: su questi volumi è più veloce di qualunque query.

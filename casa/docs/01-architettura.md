# Casa — Architettura dell'app

## 1. In una frase

**Casa** è una PWA mobile-first che dà a due persone un unico spazio condiviso per
catturare in pochi secondi task, eventi, note e spese, e rivederli in una dashboard
"Oggi", in un calendario e in un pannello spese.

Il vincolo di progetto che guida ogni decisione: **una nuova task o una nuova spesa
deve entrare in meno di 5 secondi**.

---

## 2. Stack

| Livello | Scelta | Perché |
|---|---|---|
| UI | Next.js 16 (App Router) + React 19 + TypeScript | routing file-based, build statica, ottima PWA |
| Stile | Tailwind CSS v4 con design token CSS | token unici per light/dark, zero runtime |
| Icone | `lucide-react` | set coerente, stroke sottile, "premium" |
| Date | `date-fns` + locale `it` | formattazione italiana affidabile |
| Dati | **Supabase** (Postgres + Auth + Realtime) | auth sicura, RLS, realtime già inclusi |
| Grafici | SVG scritti a mano | nessuna dipendenza pesante, controllo pieno sullo stile |
| Notifiche | Notification API + Service Worker + Web Push (VAPID) | promemoria anche ad app chiusa |

### 2.1 Due driver dati intercambiabili

Il punto delicato di un progetto così è che senza un progetto Supabase attivo l'app
non parte. Per questo tutto l'accesso ai dati passa da una sola interfaccia
(`src/lib/data/driver.ts`) con **due implementazioni**:

```
                    ┌──────────────────────┐
   UI components ──▶│  StoreProvider (ctx) │──▶  Driver (interfaccia)
                    └──────────────────────┘          │
                                              ┌───────┴────────┐
                                              ▼                ▼
                                      SupabaseDriver     LocalDriver
                                      (produzione)       (demo/offline)
```

* **`SupabaseDriver`** — usato quando `NEXT_PUBLIC_SUPABASE_URL` e
  `NEXT_PUBLIC_SUPABASE_ANON_KEY` sono presenti. Auth reale, RLS, realtime.
* **`LocalDriver`** — fallback automatico. Persiste su `localStorage`, simula il
  realtime con `BroadcastChannel` (due schede del browser = due utenti), e carica
  dati di esempio. Serve per provare l'app in 10 secondi e per lo sviluppo offline.

Cambiare driver non richiede di toccare un solo componente.

### 2.2 Perché uno store client-side completo

Il dataset di una coppia è minuscolo (ordine di 10³ righe/anno). L'app quindi
**carica tutto lo spazio condiviso in memoria all'avvio** e lavora su quello:

* ricerca globale istantanea, senza round-trip;
* dashboard, calendario e statistiche sono pure funzioni sullo snapshot;
* le mutazioni sono **ottimistiche**: la UI si aggiorna prima della risposta server;
* il realtime si limita a fare merge dei delta.

È la scelta che rende possibile il vincolo dei 5 secondi.

---

## 3. Struttura delle cartelle

```
casa/
├── docs/                        # questi documenti
├── supabase/
│   ├── migrations/
│   │   └── 0001_init.sql        # schema + RLS + trigger + seed categorie
│   └── README.md                # come applicarlo
├── public/
│   ├── manifest.webmanifest
│   ├── sw.js                    # service worker (cache + push)
│   └── icons/
└── src/
    ├── app/
    │   ├── layout.tsx            # html, provider, registrazione SW
    │   ├── page.tsx              # Home / Oggi
    │   ├── tasks/page.tsx
    │   ├── calendar/page.tsx
    │   ├── expenses/page.tsx
    │   ├── more/page.tsx
    │   ├── inbox/page.tsx
    │   ├── search/page.tsx
    │   ├── stats/page.tsx
    │   ├── settings/page.tsx
    │   ├── (auth)/login|signup|reset|update-password
    │   ├── onboarding/page.tsx   # crea o entra in uno spazio
    │   └── api/
    │       ├── parse/route.ts    # NLU: AI se disponibile, altrimenti regole
    │       └── push/run/route.ts # cron: invia i promemoria scaduti
    ├── components/
    │   ├── shell/                # BottomNav, FAB, AppShell, Header
    │   ├── capture/              # QuickCapture + schermata di conferma
    │   ├── forms/                # TaskForm, EventForm, ExpenseForm, NoteForm
    │   ├── calendar/             # viste Giorno/Settimana/Mese
    │   ├── charts/               # Donut, Bars, Sparkline
    │   └── ui/                   # Sheet, Card, Button, Field, Chip, ...
    └── lib/
        ├── types.ts              # modello di dominio
        ├── data/                 # driver, store, realtime, selettori
        ├── parse.ts              # NLU italiana a regole
        ├── recurrence.ts         # espansione ricorrenze
        ├── reminders.ts          # materializzazione promemoria
        ├── notifications.ts      # permessi + scheduler in-app
        └── date.ts, format.ts, cn.ts
```

---

## 4. Flusso di rendering

1. `layout.tsx` monta `StoreProvider`, che:
   * risolve la sessione (driver);
   * se non c'è → redirect a `/login`;
   * se c'è ma l'utente non appartiene a nessuno spazio → `/onboarding`;
   * altrimenti carica lo snapshot e apre il canale realtime.
2. Ogni pagina consuma lo store con selettori memoizzati (`useToday()`,
   `useTasks(filter)`, `useExpenseTotals(range)`…).
3. Le mutazioni chiamano `store.mutate(...)`: aggiorna subito lo stato locale,
   poi scrive sul driver e riconcilia (o fa rollback con toast in caso di errore).

---

## 5. Sicurezza e privacy

* Auth via Supabase (email + password, reset via email). Nessuna password gestita
  dall'app.
* **Row Level Security su ogni tabella**: si vede una riga solo se
  `household_id` è fra gli spazi dell'utente **e** (`visibility = 'shared'`
  **oppure** `created_by = auth.uid()`).
* La `visibility` è per-riga e si sceglie in fase di creazione ("Solo io" /
  "Condiviso"), quindi l'isolamento è garantito dal database, non dalla UI.
* L'invito al partner avviene con un **codice a 6 caratteri** rigenerabile, oltre
  che per email.

---

## 6. Cosa è nell'MVP e cosa no

**Dentro:** auth + spazio condiviso, dashboard Oggi, Quick Capture con parsing,
task (con ricorrenza), calendario giorno/settimana/mese, promemoria, spese con
grafici, inbox, ricerca globale, riepiloghi giornaliero e settimanale, light/dark,
PWA installabile, realtime.

**Fuori (volutamente):** allegati e foto scontrini, budget per categoria,
liste della spesa collaborative, più di uno spazio per utente nella UI (lo schema
lo supporta già), integrazione con calendari esterni, OCR.

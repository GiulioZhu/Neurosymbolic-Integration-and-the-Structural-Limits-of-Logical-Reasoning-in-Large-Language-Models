# Casa

Lo spazio condiviso per la vita quotidiana di una coppia: **attività, eventi,
note e spese in un posto solo**, su telefono e su desktop.

Progressive Web App in Next.js + TypeScript, con Supabase (Postgres, auth,
realtime) come backend. Tutto in italiano.

Il vincolo che ha guidato ogni scelta: **una nuova task o una nuova spesa deve
entrare in meno di 5 secondi.**

---

## Provala subito

```bash
cd casa
npm install
npm run dev
```

Apri <http://localhost:3000>. **Non serve configurare niente**: senza le
variabili di Supabase l'app parte in *modalità demo* — dati in `localStorage`,
due utenti di esempio (Luca e Marta) e un dataset già popolato.

Per vedere la sincronizzazione fra due persone senza backend: apri l'app in due
schede. Il `LocalDriver` propaga le modifiche con `BroadcastChannel`, quindi si
comportano come due dispositivi.

Credenziali demo: `luca@casa.app` / `demo1234`.

---

## Collegare Supabase (uso reale in due)

1. Crea un progetto su [supabase.com](https://supabase.com).
2. Applica lo schema: vedi [`supabase/README.md`](supabase/README.md).
3. Copia `.env.example` in `.env.local` e riempi:

   ```bash
   NEXT_PUBLIC_SUPABASE_URL=https://xxxx.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
   ```

4. Riavvia `npm run dev`. L'app passa da sola al driver Supabase: auth reale,
   Row Level Security, realtime.

Il primo utente si registra, crea lo spazio e ottiene un **codice di invito a 6
caratteri**; il secondo si registra e lo inserisce. Da quel momento vedono le
stesse cose, con l'indicazione di chi ha creato cosa.

---

## Cosa c'è dentro

| Sezione | Cosa fa |
|---|---|
| **Home / Oggi** | data, saluto, riepilogo (`"Oggi hai 4 attività e 2 eventi"`), attività ed eventi del giorno, ritardi, urgenze, prossima scadenza, spese di oggi e del mese |
| **Quick Capture** | un campo: scrivi in italiano, l'app capisce tipo, data, ora, importo, categoria, ricorrenza e promemoria, e chiede conferma |
| **Attività** | data, ora, priorità, categoria, assegnatario, stato, promemoria, **ricorrenze**, note, privacy |
| **Calendario** | viste Giorno / Settimana / Mese, con eventi, scadenze e spese insieme, distinti per colore |
| **Spese** | inserimento in due tap, totali per periodo/categoria/persona, confronto col mese scorso, grafici |
| **Inbox** | pensieri buttati giù al volo, trasformabili in attività, eventi o spese |
| **Ricerca** | globale e istantanea su tutto |
| **Statistiche** | riepilogo settimanale e andamento a 6 mesi |
| **Impostazioni** | profilo, spazio e invito, notifiche, tema chiaro/scuro |

### Quick Capture — esempi che funzionano

| Scrivi | Ottieni |
|---|---|
| `Comprare latte domani` | Attività · domani |
| `Chiamare commercialista venerdì` | Attività · venerdì |
| `Cena con Marco sabato alle 20` | Evento · sab 20:00–21:00 |
| `Pagato 35€ benzina` | Spesa · 35,00 € · Trasporti · oggi |
| `45 euro cena` | Spesa · 45,00 € · Ristoranti · oggi |
| `Domani alle 17 ricordami di comprare un regalo per Luca` | Attività · domani 17:00 · promemoria |
| `Pagare affitto ogni mese il 1` | Attività ricorrente mensile |

Il parser è **a regole, offline e deterministico** (`src/lib/parse.ts`): risponde
a ogni tasto, senza latenza né costi. Se imposti `ANTHROPIC_API_KEY`, dopo una
breve pausa `/api/parse` chiede anche a Claude una lettura migliore per le frasi
più contorte, valida campo per campo la risposta e, al minimo dubbio, tiene
quella locale. Il campo non smette mai di funzionare.

---

## Notifiche

Due livelli indipendenti:

1. **Sempre attivo** — finché una scheda è aperta, uno scheduler consegna i
   promemoria scaduti con la Notification API. Basta dare il permesso in
   Impostazioni.
2. **Ad app chiusa** (opzionale) — Web Push. Genera le chiavi e configurale:

   ```bash
   npm run vapid   # stampa la coppia da incollare in .env.local
   ```

   Poi fai chiamare `POST /api/push/run` da un cron (Vercel Cron, pg_cron,
   GitHub Actions…) ogni pochi minuti, con
   `Authorization: Bearer $CRON_SECRET`. Servono anche
   `SUPABASE_SERVICE_ROLE_KEY` e `CRON_SECRET`.

Esempio di `vercel.json`:

```json
{ "crons": [{ "path": "/api/push/run", "schedule": "*/5 * * * *" }] }
```

---

## Comandi

```bash
npm run dev        # sviluppo
npm run build      # build di produzione
npm start          # serve la build
npm test           # test del parser, delle ricorrenze e dei selettori
npm run typecheck  # tsc --noEmit
npm run lint       # eslint
npm run vapid      # genera le chiavi per le push
```

---

## Struttura

```
casa/
├── docs/                     # architettura, user flow, database, schermate, design system
├── supabase/migrations/      # schema SQL con RLS, trigger e RPC
├── public/                   # manifest, service worker, icone
├── tests/                    # test unitari (node:test via tsx)
└── src/
    ├── app/                  # pagine (App Router) e API route
    ├── components/           # shell, capture, form, calendario, grafici, UI
    └── lib/
        ├── parse.ts          # NLU italiana
        ├── recurrence.ts     # espansione delle ricorrenze
        ├── data/             # driver, store, selettori
        └── date.ts, format.ts, notifications.ts, theme.ts
```

Documenti di progetto (letti nell'ordine, raccontano il progetto per intero):

1. [Architettura](docs/01-architettura.md)
2. [User flow](docs/02-user-flow.md)
3. [Database](docs/03-database.md)
4. [Design system](docs/04-design-system.md)
5. [Schermate](docs/05-schermate.md)

---

## Scelte che vale la pena conoscere

**Due driver dati.** Tutto l'accesso ai dati passa da un'unica interfaccia
(`src/lib/data/driver.ts`) con due implementazioni: `SupabaseDriver` per la
produzione e `LocalDriver` per la demo. Nessun componente sa quale è in uso.

**Snapshot in memoria.** Il dataset di una coppia è minuscolo, quindi l'app
carica tutto all'avvio e lavora su quello. Ricerca, dashboard, calendario e
statistiche diventano funzioni pure: istantanee, e testabili senza database.

**Mutazioni ottimistiche.** La UI si aggiorna prima della risposta del server e
fa rollback con un avviso se qualcosa va storto.

**La privacy è nel database.** Ogni riga è `shared` o `private`, e le policy RLS
la applicano lato Postgres: la UI non è l'unica difesa.

**Il denaro è in centesimi.** `amount_cents bigint`, mai float.

**Date e ore separate.** "Domani" senza orario è il caso più frequente; un
`timestamptz` costringerebbe a inventare un'ora.

---

## Stato

L'MVP richiesto è completo e verificato: build di produzione pulita, 34 test
verdi, typecheck e lint puliti, e un giro completo dell'interfaccia in Chromium
senza errori in console.

Volutamente **fuori** da questa prima versione: allegati e foto degli scontrini,
budget per categoria, liste della spesa collaborative, sincronizzazione con
calendari esterni, più di uno spazio per utente nella UI (lo schema lo supporta
già).

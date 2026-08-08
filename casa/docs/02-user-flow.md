# Casa — User flow

## 1. Primo accesso (utente A, chi crea lo spazio)

```
/signup  ──▶  inserisce nome, email, password
   │
   ▼
verifica email (se attiva su Supabase)  ──▶  /login
   │
   ▼
/onboarding
   ├── "Crea uno spazio"  ──▶  nome ("Casa", "Family")  ──▶  spazio creato
   │                              │
   │                              ▼
   │                        codice invito a 6 caratteri (es. K7M2QP)
   │                        + campo "invita per email"
   ▼
/  (dashboard Oggi, vuota con empty state guidato)
```

## 2. Ingresso del partner (utente B)

```
/signup ──▶ /onboarding ──▶ "Entra con un codice" ──▶ K7M2QP ──▶ /
```

Da questo momento i due utenti condividono lo stesso `household_id`. Ogni riga
mostra l'avatar di chi l'ha creata.

## 3. Il flusso critico: inserire qualcosa (< 5 secondi)

Due strade, entrambe sempre a portata di pollice.

### 3a. Quick Capture (la via veloce)

```
[Home]  campo "Scrivi qualsiasi cosa…"  (sempre in cima alla dashboard)
   │  l'utente scrive: "Cena con Marco sabato alle 20"
   │
   ▼  (parsing in tempo reale, mentre scrive)
┌───────────────────────────────────────────┐
│  Evento · Cena con Marco                  │
│  sab 14 mar · 20:00–21:00                 │
│  [Evento] [Task] [Nota] [Spesa]  ← switch │
│                        [Modifica] [Salva] │
└───────────────────────────────────────────┘
   │  Invio (o tap su Salva)
   ▼
salvato + toast "Evento creato · Annulla"
```

* Il tipo è **indovinato** ma sempre correggibile con un tap.
* "Modifica" apre il form completo pre-compilato: nessuna informazione va persa.
* Il campo resta focalizzato: si possono inserire più cose di fila.

Esempi che il parser gestisce:

| Testo | Risultato |
|---|---|
| `Comprare latte domani` | Task · scadenza domani |
| `Chiamare commercialista venerdì` | Task · venerdì prossimo |
| `Cena con Marco sabato alle 20` | Evento · sab 20:00–21:00 |
| `Pagato 35€ benzina` | Spesa · 35,00 € · Trasporti · oggi |
| `45 euro cena` | Spesa · 45,00 € · Ristoranti · oggi |
| `Domani alle 17 ricordami di comprare un regalo per Luca` | Task · domani 17:00 · promemoria all'orario |
| `Pagare affitto ogni mese il 1` | Task ricorrente · mensile, giorno 1 |

### 3b. FAB "+" (la via esplicita)

```
[ + ]  (pulsante centrale della bottom nav, rialzato)
   │
   ▼  action sheet a comparsa dal basso
   ├── Nuova attività   ──▶ TaskForm
   ├── Nuovo evento     ──▶ EventForm
   ├── Nuova nota       ──▶ NoteForm
   └── Nuova spesa      ──▶ ExpenseForm  (tastierino numerico subito attivo)
```

La spesa è ottimizzata al massimo: si apre con il focus sull'importo, categoria
e metodo di pagamento hanno il valore usato più di recente, data = oggi.
Due tap e un numero.

## 4. Giornata tipo

```
mattina   apre l'app  ──▶  Home mostra:
                            "Buongiorno Luca. Oggi hai 4 attività e 2 eventi."
                            urgenti · in ritardo · prossima attività · prossimo evento
giorno    Quick Capture al volo, spunta le task con un tap sul cerchio
          la moglie aggiunge "Comprare acqua" ──▶ compare entro ~1s (realtime)
sera      "+ Spesa" ×2, poi guarda il totale di oggi in Home
domenica  /stats  ──▶  riepilogo settimanale
```

## 5. Nota → azione (Inbox)

```
/inbox  ──▶  "Controllare volo Londra"
                │
                ▼  tap su una nota  ──▶  azioni:
                   ├── Trasforma in attività  (apre TaskForm precompilato)
                   ├── Trasforma in evento
                   ├── Trasforma in spesa
                   └── Archivia
```

La nota originale viene marcata `converted_to` così resta la traccia, ma sparisce
dall'Inbox attiva.

## 6. Completamento di una task condivisa

```
utente A tap sul cerchio  ──▶  status = done, completed_at, completed_by = A
                               │
                               ├─ UI di A: barrata + animazione, sparisce dopo 400ms
                               └─ realtime ──▶ UI di B: identica, entro ~1s
                               │
                               └─ se la task è ricorrente:
                                     l'occorrenza corrente si chiude e
                                     ne viene creata una nuova con la data successiva
```

## 7. Promemoria

```
creazione task/evento con promemoria
      │  (all'ora · 10 min · 30 min · 1 ora · 1 giorno · personalizzato)
      ▼
materializzazione righe in `notifications` (scheduled_for calcolato)
      │
      ├── app aperta   ──▶ scheduler in-app: timer + Notification API
      └── app chiusa   ──▶ cron chiama /api/push/run ──▶ Web Push ──▶ SW ──▶ notifica
                            tap sulla notifica ──▶ apre l'app sull'elemento
```

## 8. Privacy per elemento

Ogni form ha un selettore a due stati, sempre visibile, mai nascosto in un menu:

```
   [ 👤 Solo io ]   [ 👥 Condiviso ]        ← default: Condiviso
```

Con "Solo io" la riga non è leggibile dall'altro utente: la restrizione è
applicata dalle policy RLS del database, non solo dalla UI.

## 9. Mappa della navigazione

```
┌──────────────────────────────────────────────────────────────┐
│                        bottom navigation                     │
│  Home      Task      [ + ]      Calendario   Spese    Altro  │
└──────────────────────────────────────────────────────────────┘
     │         │                       │         │        │
     │         │                       │         │        ├── Inbox
     │         │                       │         │        ├── Ricerca
     │         │                       │         │        ├── Statistiche
     │         │                       │         │        └── Impostazioni
     │         │                       │         │              ├── Profilo
     │         │                       │         │              ├── Spazio + invito
     │         │                       │         │              ├── Notifiche
     │         │                       │         │              └── Tema
     │         │                       │         └── liste + grafici + filtri periodo
     │         │                       └── Giorno / Settimana / Mese
     │         └── Oggi · Prossime · In ritardo · Completate · filtri
     └── riepilogo del giorno + quick capture + spese di oggi
```

Sono 5 voci esatte, il "+" è la sesta ma non è una sezione: è un'azione.

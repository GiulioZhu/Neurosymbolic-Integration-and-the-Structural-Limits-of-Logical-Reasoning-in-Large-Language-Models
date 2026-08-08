# Casa — Schermate principali

Wireframe a bassa fedeltà delle sei schermate che portano il peso dell'MVP.
Larghezza di riferimento: 390px (iPhone).

---

## 1. Home / Oggi — `/`

```
┌─────────────────────────────────────────┐
│  sabato 8 agosto            [☾]  [LZ]   │  header: data + tema + avatar
│                                         │
│  Buongiorno, Luca                       │  26px semibold
│  Oggi hai 4 attività e 2 eventi.        │  15px muted
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ ✎  Scrivi qualsiasi cosa…       │    │  QUICK CAPTURE
│  └─────────────────────────────────┘    │  sempre in cima
│                                         │
│  ┌──────────┐┌──────────┐┌──────────┐   │
│  │ IN RIT.  ││ URGENTI  ││ OGGI €   │   │  3 tessere di stato
│  │    1     ││    2     ││  42,50   │   │
│  └──────────┘└──────────┘└──────────┘   │
│                                         │
│  OGGI                        4 attività │
│  ┌─────────────────────────────────┐    │
│  │ ○  Comprare latte               │    │
│  │    Spesa alimentare · LZ        │    │
│  │ ─────────────────────────────── │    │
│  │ ○  Chiamare commercialista  !!  │    │
│  │    11:00 · assegnata a te       │    │
│  │ ─────────────────────────────── │    │
│  │ ✓  Portare fuori il cane        │    │  completata: barrata, tenue
│  └─────────────────────────────────┘    │
│                                         │
│  EVENTI DI OGGI                         │
│  ┌─────────────────────────────────┐    │
│  │ ▍20:00  Cena con Marco          │    │  ▍ = barra colore evento
│  │   20:00–22:00 · Da Vito         │    │
│  └─────────────────────────────────┘    │
│                                         │
│  IN SCADENZA                            │
│  ┌─────────────────────────────────┐    │
│  │ ○  Pagare bollo auto   lun 10   │    │
│  └─────────────────────────────────┘    │
│                                         │
│  SPESE                                  │
│  ┌─────────────────────────────────┐    │
│  │  Oggi              42,50 €      │    │
│  │  Questo mese      812,30 €      │    │
│  │  ▁▂▅▃▇▂▁▄▆▃▂▅  (sparkline mese) │    │
│  │  −8% rispetto a luglio          │    │
│  └─────────────────────────────────┘    │
│                                         │
├─────────────────────────────────────────┤
│  ⌂       ☑      (+)      ▤       ⋯      │
│ Home    Task          Calend.  Spese Altro│
└─────────────────────────────────────────┘
```

Regola della schermata: **tutto ciò che serve alla giornata è sopra la piega o a
un pollice di scroll**. Le sezioni vuote scompaiono, non mostrano "nessun
elemento": una giornata libera deve *sembrare* libera.

---

## 2. Quick Capture — conferma

Non è una pagina, è uno sheet che sale mentre si scrive.

```
┌─────────────────────────────────────────┐
│  Cena con Marco sabato alle 20      ⌫   │  ← testo digitato
│─────────────────────────────────────────│
│                                         │
│   ┌─────────────────────────────────┐   │
│   │ ▍ EVENTO                        │   │
│   │   Cena con Marco                │   │
│   │   sab 14 mar · 20:00 – 21:00    │   │
│   │   Condiviso · creato da te      │   │
│   └─────────────────────────────────┘   │
│                                         │
│   È invece:                             │
│   [ Task ] [ Nota ] [ Spesa ]           │  ← correzione con un tap
│                                         │
│   [   Modifica   ] [     Salva      ]   │
└─────────────────────────────────────────┘
```

Con `45 euro cena` la card diventa:

```
   │ ▍ SPESA                         │
   │   Cena                          │
   │   45,00 € · Ristoranti · oggi   │
   │   Pagato da te · Carta          │
```

---

## 3. Task — `/tasks`

```
┌─────────────────────────────────────────┐
│  Attività                        [⚙]    │
│  [ Oggi ][ Prossime ][ Ritardo ][ Fatte]│  segmented, scorrevole
│  [ Tutte ▾ ][ Chiunque ▾ ][ Priorità ▾ ]│  filtri secondari
│                                         │
│  IN RITARDO                             │
│  ┌─────────────────────────────────┐    │
│  │ ○ Rinnovare assicurazione    !! │    │
│  │   scaduta 2 giorni fa · MZ      │    │
│  └─────────────────────────────────┘    │
│                                         │
│  OGGI · sabato 8                        │
│  ┌─────────────────────────────────┐    │
│  │ ○ Comprare latte                │    │
│  │ ○ Chiamare commercialista 11:00 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  DOMANI                                 │
│  ┌─────────────────────────────────┐    │
│  │ ○ Pagare affitto      ↻ mensile │    │  ↻ = ricorrente
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

Dettaglio task (sheet): titolo · descrizione · data · ora · priorità · categoria ·
assegnatario · stato · promemoria · ricorrenza · note · privacy · "creato da".

---

## 4. Calendario — `/calendar`

```
┌─────────────────────────────────────────┐
│  ‹   Agosto 2026   ›     [G][S][M]      │
│                                         │
│  L   M   M   G   V   S   D              │
│                      1   2              │
│  3   4   5   6   7  (8)  9              │  (8) = oggi, pill accento
│      ·       ··          ·              │  puntini per tipo
│ 10  11  12  13  14  15  16              │
│  ··  ·           ·                      │
│                                         │
│─────────────────────────────────────────│
│  SABATO 8 AGOSTO                        │
│  ┌─────────────────────────────────┐    │
│  │ ▍ 20:00  Cena con Marco     ev  │    │  viola
│  │ ▍ 11:00  Chiamare comm.     ta  │    │  teal
│  │ ▍ ——     Affitto 850 €      sp  │    │  ambra
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

* **Mese**: griglia + puntini colorati per tipo, con la lista del giorno
  selezionato sotto. È la vista che si usa per orientarsi, non per leggere.
* **Settimana**: 7 colonne compatte con le fasce orarie, scroll orizzontale.
* **Giorno**: timeline oraria 6–24 con blocchi posizionati, task senza ora in
  una fascia "Tutto il giorno" in testa.

Legenda sempre visibile: `● evento  ● attività  ● spesa`.

---

## 5. Spese — `/expenses`

```
┌─────────────────────────────────────────┐
│  Spese                                  │
│  [ Oggi ][ Settimana ][ Mese ][ Anno ]  │
│                                         │
│         812,30 €                        │  display 34px
│         agosto · −8% vs luglio          │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │        ╭───────╮                │    │
│  │       │  ◕     │  Alimentari 32%│    │  donut + legenda
│  │       │        │  Casa       24%│    │
│  │        ╰───────╯  Ristoranti 18%│    │
│  └─────────────────────────────────┘    │
│                                         │
│  CHI HA PAGATO                          │
│  ┌─────────────────────────────────┐    │
│  │ LZ  ████████████░░░░  512,30 €  │    │
│  │ MZ  ███████░░░░░░░░░  300,00 €  │    │
│  └─────────────────────────────────┘    │
│                                         │
│  MOVIMENTI                              │
│  ┌─────────────────────────────────┐    │
│  │ 🛒 Esselunga         −64,20 €   │    │
│  │    oggi · Carta · LZ            │    │
│  │ ⛽ Benzina           −35,00 €   │    │
│  │    ieri · Contanti · MZ         │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

Nuova spesa (sheet): si apre con il **tastierino numerico attivo** sull'importo.
Categoria e metodo hanno l'ultimo valore usato. Data = oggi. Salvataggio in due
tap.

---

## 6. Altro — `/more`

```
┌─────────────────────────────────────────┐
│  Altro                                  │
│  ┌─────────────────────────────────┐    │
│  │ 📥  Inbox                    3 ›│    │
│  │ 🔍  Ricerca                    ›│    │
│  │ 📊  Statistiche                ›│    │
│  │ ⚙️  Impostazioni               ›│    │
│  └─────────────────────────────────┘    │
│                                         │
│  QUESTA SETTIMANA                       │
│  ┌─────────────────────────────────┐    │
│  │  12 completate    5 da fare     │    │
│  │  3 eventi         214,80 € spesi│    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**Ricerca** (`/search`): un campo, risultati raggruppati per tipo mentre si
scrive, con evidenziazione del termine. Cercando `assicurazione` compaiono la
task, l'evento di rinnovo, la nota e le spese collegate.

**Statistiche** (`/stats`): riepilogo settimanale (completate, da fare, eventi,
spesa, categorie principali) e andamento a 6 mesi.

**Impostazioni** (`/settings`): profilo, spazio + codice invito con condivisione,
permessi notifiche, tema (Sistema/Chiaro/Scuro), logout.

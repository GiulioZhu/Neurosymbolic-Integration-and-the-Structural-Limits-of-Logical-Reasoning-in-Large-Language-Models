# Casa — Design system

Riferimenti dichiarati: Apple Reminders, Apple Calendar, Notion, Todoist, e le
app di finanza personale moderne. Il denominatore comune è **poco cromo, molta
aria, gerarchia data dalla tipografia**. Casa aggiunge una temperatura calda per
non sembrare uno strumento di lavoro.

## 1. Colore

Base neutra calda (sabbia), non grigia. Un solo accento: terracotta.

### Light

| Token | Valore | Uso |
|---|---|---|
| `--bg` | `#FAF8F5` | sfondo dell'app |
| `--surface` | `#FFFFFF` | card |
| `--surface-2` | `#F3EFE9` | input, chip, stati hover |
| `--border` | `#E9E3DA` | bordi hairline |
| `--text` | `#1B1917` | testo primario |
| `--muted` | `#7C736B` | testo secondario |
| `--faint` | `#A69C93` | metadati, placeholder |
| `--accent` | `#C25A38` | azione primaria, oggi, focus |
| `--accent-soft` | `#FAEDE7` | riempimenti dell'accento |

### Dark

| Token | Valore |
|---|---|
| `--bg` | `#131211` |
| `--surface` | `#1C1A18` |
| `--surface-2` | `#252220` |
| `--border` | `#332E2A` |
| `--text` | `#F5F2ED` |
| `--muted` | `#A39A91` |
| `--faint` | `#7A716A` |
| `--accent` | `#E8825E` |
| `--accent-soft` | `#33221B` |

Il dark non è un'inversione: le superfici restano leggermente calde e i contrasti
sono ridotti per non abbagliare di sera. Il tema segue il sistema, con override
manuale in Impostazioni (`data-theme` sull'elemento `<html>`).

### Colori semantici per tipo di elemento

Servono a distinguere a colpo d'occhio nel calendario e nella ricerca.

| Tipo | Colore | Icona |
|---|---|---|
| Attività | `#3F8F7E` verde-teal | `circle` / `check-circle` |
| Evento | `#7A6BB5` viola tenue | `calendar` |
| Spesa | `#C4873C` ambra | `wallet` |
| Nota | `#8A8175` pietra | `sticky-note` |

Priorità: bassa `--faint`, normale `#4E7C9B`, alta `#D08A24`, urgente `#C0453C`.
La priorità non colora l'intera card — solo un puntino o il bordo sinistro.
Se tutto urla, niente si sente.

## 2. Tipografia

Stack di sistema, così il testo è nativo su ogni piattaforma e non c'è nessun
font da scaricare (primo render immediato, requisito per un'app che si apre 20
volte al giorno):

```css
-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
"Helvetica Neue", Arial, sans-serif
```

Scala (mobile):

| Ruolo | Dimensione / peso / tracking |
|---|---|
| Display (importi grandi) | 34px / 640 / −0.02em, tabular-nums |
| Titolo pagina | 26px / 640 / −0.02em |
| Titolo sezione | 13px / 600 / +0.06em / uppercase / `--muted` |
| Titolo card | 16px / 550 |
| Corpo | 15px / 400 |
| Metadato | 13px / 450 / `--muted` |
| Micro | 11px / 500 / `--faint` |

Numeri sempre `font-variant-numeric: tabular-nums`: le colonne di importi devono
allinearsi.

## 3. Spazio, forma, profondità

* Griglia da **4px**. Padding di pagina 20px, gap fra card 12px, padding card 16px.
* Raggi: card `18px`, input e bottoni `14px`, chip e pill `999px`, FAB cerchio.
* **Nessuna ombra pesante.** Le card si separano con un bordo hairline
  (`1px solid --border`) e, quando servono a galleggiare (sheet, FAB), una sola
  ombra morbida:
  `0 8px 30px rgba(28,20,14,.10)`.
* Superficie mai satura: ogni schermata deve avere almeno il 30% di vuoto.

## 4. Componenti

**Card** — superficie, bordo hairline, raggio 18, padding 16. È l'unico
contenitore. Niente pannelli dentro pannelli.

**Riga elemento (task)** — cerchio di spunta 22px a sinistra (area di tocco 44px),
titolo, sotto una riga di metadati piccoli (ora, categoria, avatar autore).
Tap sul cerchio = completa. Tap sulla riga = apre il dettaglio. Sono due bersagli
diversi e non devono mai sovrapporsi.

**Bottom navigation** — 5 voci, altezza 56px + safe-area inferiore, icona 22px +
etichetta 10px. Voce attiva in `--accent`, con un puntino sotto l'icona.

**FAB "+"** — 56px, centrato e rialzato di 18px sopra la barra, accento pieno,
ombra morbida. È l'elemento più visibile dell'interfaccia: è il gesto che l'app
vuole incoraggiare.

**Sheet** — tutti i form salgono dal basso, angoli superiori 24px, maniglia da
trascinamento, chiusura con swipe/backdrop/Esc. Non si usano mai modali centrate:
su un telefono in una mano il pollice sta in basso.

**Chip / segmented control** — per stati, categorie, viste del calendario.
Selezionato = `--surface` su sfondo `--surface-2` con transizione di 180ms.

**Empty state** — icona tenue, una frase, un'azione. Mai una schermata bianca.

## 5. Movimento

Discreto e breve, serve a spiegare la causalità, non a decorare.

| Gesto | Animazione |
|---|---|
| Apertura sheet | `translateY(100%→0)`, 260ms, `cubic-bezier(.32,.72,0,1)` |
| Completamento task | spunta che si riempie 180ms, riga che sfuma 400ms |
| Comparsa card | fade + `translateY(6px→0)`, 200ms, sfalsata di 30ms |
| Pressione | `scale(.97)`, 120ms |
| Cambio numero | nessuna animazione (i numeri che scorrono si leggono peggio) |

Tutto rispetta `prefers-reduced-motion: reduce` → durate azzerate.

## 6. Regole mobile-first

1. Nulla di interattivo sotto i **44×44px**.
2. Le azioni primarie stanno nella metà **inferiore** dello schermo.
3. `env(safe-area-inset-*)` su header, bottom nav e sheet.
4. Font degli input ≥ 16px, altrimenti iOS fa zoom al focus.
5. Il campo Quick Capture non perde il focus dopo il salvataggio.
6. Da 768px in su il layout diventa a due colonne con sidebar laterale, ma i
   componenti restano gli stessi: nessun secondo design da mantenere.

## 7. Accessibilità

* Contrasto minimo 4.5:1 per il testo, verificato su entrambi i temi.
* Il colore non è mai l'unico veicolo di informazione: tipo ed esito hanno
  sempre anche un'icona o un'etichetta.
* Focus visibile con anello da 2px in `--accent` (non rimosso mai).
* Sheet con `role="dialog"`, `aria-modal`, focus intrappolato, chiusura con Esc.

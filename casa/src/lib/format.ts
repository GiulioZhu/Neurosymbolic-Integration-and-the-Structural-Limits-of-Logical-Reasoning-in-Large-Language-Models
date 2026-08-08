const eur = new Intl.NumberFormat("it-IT", {
  style: "currency",
  currency: "EUR",
  minimumFractionDigits: 2,
});

const eurCompact = new Intl.NumberFormat("it-IT", {
  style: "currency",
  currency: "EUR",
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});

/** 4250 → "42,50 €" */
export function money(cents: number): string {
  return eur.format(cents / 100);
}

/** 4250 → "43 €" — per assi dei grafici e tessere strette. */
export function moneyShort(cents: number): string {
  return eurCompact.format(cents / 100);
}

/** "42,50" o "42.50" o "42" → 4250. Restituisce null se non è un numero. */
export function parseAmountToCents(input: string): number | null {
  const cleaned = input.replace(/[^\d.,-]/g, "").replace(/\.(?=\d{3}\b)/g, "");
  if (!cleaned) return null;
  const normalized = cleaned.replace(",", ".");
  const n = Number(normalized);
  if (!Number.isFinite(n)) return null;
  return Math.round(Math.abs(n) * 100);
}

/** "Luca Zanetti" → "LZ" */
export function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

/** "Luca Zanetti" → "Luca" */
export function firstName(name: string): string {
  return name.trim().split(/\s+/)[0] || name;
}

export function pluralize(n: number, one: string, many: string): string {
  return n === 1 ? one : many;
}

/** "3 attività" / "1 attività" con il numero incluso. */
export function count(n: number, one: string, many: string): string {
  return `${n} ${pluralize(n, one, many)}`;
}

/** Variazione percentuale, con segno e simbolo. `null` se non calcolabile. */
export function percentDelta(current: number, previous: number): string | null {
  if (previous === 0) return current === 0 ? null : null;
  const pct = ((current - previous) / previous) * 100;
  if (!Number.isFinite(pct)) return null;
  const rounded = Math.round(pct);
  if (rounded === 0) return "in linea";
  return `${rounded > 0 ? "+" : "−"}${Math.abs(rounded)}%`;
}

/** Saluto in base all'ora. */
export function greeting(d = new Date()): string {
  const h = d.getHours();
  if (h < 5) return "Buonanotte";
  if (h < 13) return "Buongiorno";
  if (h < 18) return "Buon pomeriggio";
  return "Buonasera";
}

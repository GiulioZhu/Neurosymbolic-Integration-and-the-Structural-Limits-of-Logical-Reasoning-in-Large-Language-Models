"use client";

import { cn } from "@/lib/cn";
import { money, moneyShort } from "@/lib/format";

// ---------------------------------------------------------------------------
//  Donut
// ---------------------------------------------------------------------------

export interface Slice {
  label: string;
  value: number;
  color: string;
}

/**
 * Anello con archi separati da un piccolo spazio. Niente animazioni sul
 * disegno: i numeri devono essere leggibili subito, non danzare.
 */
export function Donut({
  slices,
  size = 132,
  thickness = 14,
  center,
  caption,
}: {
  slices: Slice[];
  size?: number;
  thickness?: number;
  center?: string;
  caption?: string;
}) {
  const total = slices.reduce((a, s) => a + s.value, 0);
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  const gap = total > 0 && slices.length > 1 ? 1.5 : 0;

  let offset = 0;

  return (
    <div className="relative flex-shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={caption}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--surface-2)"
          strokeWidth={thickness}
        />
        {total > 0 &&
          slices.map((s) => {
            const len = (s.value / total) * c;
            const dash = Math.max(0, len - gap);
            const el = (
              <circle
                key={s.label}
                cx={size / 2}
                cy={size / 2}
                r={r}
                fill="none"
                stroke={s.color}
                strokeWidth={thickness}
                strokeLinecap="butt"
                strokeDasharray={`${dash} ${c - dash}`}
                strokeDashoffset={-offset}
                transform={`rotate(-90 ${size / 2} ${size / 2})`}
              />
            );
            offset += len;
            return el;
          })}
      </svg>
      {center && (
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="tnum text-[17px] leading-none font-semibold">{center}</span>
          {caption && <span className="mt-1 text-[11px] text-[var(--faint)]">{caption}</span>}
        </div>
      )}
    </div>
  );
}

export function Legend({ slices, total }: { slices: Slice[]; total: number }) {
  return (
    <ul className="min-w-0 flex-1 space-y-1.5">
      {slices.map((s) => (
        <li key={s.label} className="flex items-center gap-1.5 text-[12.5px]">
          <span
            aria-hidden
            className="h-2.5 w-2.5 flex-shrink-0 rounded-full"
            style={{ background: s.color }}
          />
          <span className="min-w-0 flex-1 truncate text-[var(--muted)]">{s.label}</span>
          <span className="tnum flex-shrink-0 font-medium">{money(s.value)}</span>
          <span className="tnum w-8 flex-shrink-0 text-right text-[11px] text-[var(--faint)]">
            {total ? Math.round((s.value / total) * 100) : 0}%
          </span>
        </li>
      ))}
    </ul>
  );
}

// ---------------------------------------------------------------------------
//  Barra proporzionale (chi ha pagato)
// ---------------------------------------------------------------------------

export function SplitBar({
  parts,
}: {
  parts: { label: string; value: number; color: string }[];
}) {
  const total = parts.reduce((a, p) => a + p.value, 0);
  return (
    <div className="space-y-2.5">
      {parts.map((p) => (
        <div key={p.label}>
          <div className="mb-1 flex items-baseline justify-between text-[13px]">
            <span className="text-[var(--muted)]">{p.label}</span>
            <span className="tnum font-medium">{money(p.value)}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-[var(--surface-2)]">
            <div
              className="h-full rounded-full transition-[width] duration-300"
              style={{ width: `${total ? (p.value / total) * 100 : 0}%`, background: p.color }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
//  Sparkline / barre giornaliere
// ---------------------------------------------------------------------------

export function DayBars({
  values,
  highlightIndex,
  height = 44,
  className,
  ariaLabel,
}: {
  values: number[];
  highlightIndex?: number;
  height?: number;
  className?: string;
  ariaLabel?: string;
}) {
  const max = Math.max(1, ...values);
  return (
    <div
      role="img"
      aria-label={ariaLabel}
      className={cn("flex items-end gap-[2px]", className)}
      style={{ height }}
    >
      {values.map((v, i) => (
        <span
          key={i}
          className="flex-1 rounded-[2px] transition-all"
          style={{
            height: `${Math.max(2, (v / max) * height)}px`,
            background:
              i === highlightIndex ? "var(--accent)" : v > 0 ? "var(--surface-3)" : "var(--surface-2)",
          }}
        />
      ))}
    </div>
  );
}

/** Barre orizzontali con etichetta: usate nelle statistiche mensili. */
export function MonthBars({
  items,
}: {
  items: { label: string; value: number }[];
}) {
  const max = Math.max(1, ...items.map((i) => i.value));
  return (
    <div className="flex h-[110px] items-end gap-2">
      {items.map((it) => (
        <div key={it.label} className="flex min-w-0 flex-1 flex-col items-center gap-1.5">
          <span className="tnum text-[10px] text-[var(--faint)]">
            {it.value > 0 ? moneyShort(it.value) : ""}
          </span>
          <div
            className="w-full rounded-t-[4px] bg-[var(--accent)]/85 transition-all"
            style={{ height: `${Math.max(3, (it.value / max) * 70)}px` }}
          />
          <span className="w-full truncate text-center text-[10px] text-[var(--faint)]">
            {it.label}
          </span>
        </div>
      ))}
    </div>
  );
}

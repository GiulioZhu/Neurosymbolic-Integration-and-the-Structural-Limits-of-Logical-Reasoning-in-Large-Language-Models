"use client";

import { cn } from "@/lib/cn";
import { initials } from "@/lib/format";

const ACCENT_BG: Record<string, string> = {
  clay: "#C25A38",
  sage: "#5E8C61",
  indigo: "#7A6BB5",
  amber: "#C99A2E",
  teal: "#3F8F86",
  rose: "#B5567A",
};

export function Avatar({
  name,
  accent = "clay",
  size = 28,
  title,
}: {
  name: string;
  accent?: string;
  size?: number;
  title?: string;
}) {
  const bg = ACCENT_BG[accent] ?? ACCENT_BG.clay;
  return (
    <span
      title={title ?? name}
      aria-label={title ?? name}
      className="inline-flex flex-shrink-0 items-center justify-center rounded-full font-semibold text-white"
      style={{
        width: size,
        height: size,
        background: bg,
        fontSize: Math.round(size * 0.38),
        letterSpacing: "0.01em",
      }}
    >
      {initials(name)}
    </span>
  );
}

export const AVATAR_ACCENTS = Object.keys(ACCENT_BG);
export { ACCENT_BG };

export function Card({
  children,
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("card overflow-hidden", className)} {...rest}>
      {children}
    </div>
  );
}

export function SectionHeader({
  title,
  right,
  className,
}: {
  title: string;
  right?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("mb-2 flex items-end justify-between px-1", className)}>
      <h2 className="section-title">{title}</h2>
      {right && <span className="text-[12px] text-[var(--faint)]">{right}</span>}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      <div className="mb-3 text-[var(--surface-3)]">{icon}</div>
      <p className="text-[15px] font-medium text-[var(--text)]">{title}</p>
      {description && (
        <p className="mt-1 max-w-[280px] text-[13px] text-[var(--muted)]">{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

/** Tessera numerica della dashboard. */
export function StatTile({
  label,
  value,
  tone = "default",
  onClick,
}: {
  label: string;
  value: string;
  tone?: "default" | "danger" | "accent";
  onClick?(): void;
}) {
  const color =
    tone === "danger" ? "var(--danger)" : tone === "accent" ? "var(--accent)" : "var(--text)";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!onClick}
      className={cn(
        "card press flex-1 px-3 py-2.5 text-left",
        onClick ? "cursor-pointer" : "cursor-default",
      )}
    >
      <div className="text-[11px] font-medium tracking-wide text-[var(--faint)] uppercase">
        {label}
      </div>
      <div className="tnum mt-0.5 text-[19px] leading-tight font-semibold" style={{ color }}>
        {value}
      </div>
    </button>
  );
}

/** Puntino colorato per tipo di elemento. */
export function KindDot({ color, size = 6 }: { color: string; size?: number }) {
  return (
    <span
      aria-hidden
      className="inline-block flex-shrink-0 rounded-full"
      style={{ width: size, height: size, background: color }}
    />
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      role="status"
      aria-label="Caricamento"
      className={cn(
        "inline-block h-5 w-5 animate-spin rounded-full border-2 border-[var(--surface-3)] border-t-[var(--accent)]",
        className,
      )}
    />
  );
}

"use client";

import { cn } from "@/lib/cn";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  badge?: number;
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  className,
  size = "md",
}: {
  options: SegmentedOption<T>[];
  value: T;
  // NoInfer: senza questo, `onChange={setTab}` farebbe dedurre T da
  // Dispatch<SetStateAction<Tab>> e collasserebbe su `string`.
  onChange(v: NoInfer<T>): void;
  className?: string;
  size?: "sm" | "md";
}) {
  return (
    <div
      role="tablist"
      className={cn(
        "no-scrollbar flex gap-1 overflow-x-auto rounded-full bg-[var(--surface-2)] p-1",
        className,
      )}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "press flex-shrink-0 rounded-full font-medium whitespace-nowrap transition-colors",
              size === "sm" ? "px-3 py-1.5 text-[13px]" : "px-3.5 py-2 text-[14px]",
              active
                ? "bg-[var(--surface)] text-[var(--text)] shadow-[var(--shadow-lift)]"
                : "text-[var(--muted)]",
            )}
          >
            {opt.label}
            {opt.badge !== undefined && opt.badge > 0 && (
              <span className={cn("ml-1.5 text-[12px]", active ? "text-[var(--accent)]" : "text-[var(--faint)]")}>
                {opt.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

/** Pillola selezionabile: categorie, priorità, metodi di pagamento. */
export function Chip({
  active,
  onClick,
  children,
  color,
  className,
}: {
  active?: boolean;
  onClick?(): void;
  children: React.ReactNode;
  color?: string;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "press inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[13px] font-medium",
        active
          ? "border-transparent bg-[var(--accent-soft)] text-[var(--accent)]"
          : "border-[var(--border)] bg-[var(--surface-2)] text-[var(--muted)]",
        className,
      )}
      style={active && color ? { background: `${color}1F`, color } : undefined}
    >
      {color && (
        <span
          aria-hidden
          className="h-2 w-2 flex-shrink-0 rounded-full"
          style={{ background: color }}
        />
      )}
      {children}
    </button>
  );
}

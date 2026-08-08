"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { CalendarDays, Home, ListChecks, MoreHorizontal, Plus, Wallet } from "lucide-react";
import { cn } from "@/lib/cn";
import { useComposer } from "@/components/forms/Composer";

const LEFT = [
  { href: "/", label: "Home", icon: Home },
  { href: "/tasks", label: "Task", icon: ListChecks },
] as const;

const RIGHT = [
  { href: "/calendar", label: "Calendario", icon: CalendarDays },
  { href: "/expenses", label: "Spese", icon: Wallet },
  { href: "/more", label: "Altro", icon: MoreHorizontal },
] as const;

const ALL = [...LEFT, ...RIGHT];

function useIsActive() {
  const pathname = usePathname();
  return (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));
}

/**
 * Cinque sezioni e, al centro, il "+": non è una sesta sezione ma l'azione
 * che l'app vuole rendere più facile di tutte. La barra lascia una colonna
 * vuota sotto al pulsante, così non si sovrappone a nessuna voce.
 *
 * Da 768px in su sparisce a favore della sidebar: stessi componenti, stessa
 * gerarchia, solo un'altra disposizione.
 */
export function BottomNav() {
  const isActive = useIsActive();
  const { open } = useComposer();

  const renderItem = (item: (typeof ALL)[number]) => {
    const active = isActive(item.href);
    const Icon = item.icon;
    return (
      <Link
        key={item.href}
        href={item.href}
        aria-current={active ? "page" : undefined}
        className={cn(
          "press relative flex h-[58px] flex-col items-center justify-center gap-1",
          active ? "text-[var(--accent)]" : "text-[var(--faint)]",
        )}
      >
        <Icon size={21} strokeWidth={active ? 2.2 : 1.8} />
        <span className="text-[10px] leading-none font-medium">{item.label}</span>
        {active && <span className="absolute bottom-1.5 h-1 w-1 rounded-full bg-[var(--accent)]" />}
      </Link>
    );
  };

  return (
    <>
      <button
        type="button"
        onClick={() => open({ kind: "picker" })}
        aria-label="Aggiungi qualcosa"
        className={cn(
          "press fixed bottom-[calc(24px+env(safe-area-inset-bottom))] left-1/2 z-40 -translate-x-1/2 md:hidden",
          "flex h-14 w-14 items-center justify-center rounded-full",
          "bg-[var(--accent)] text-[var(--accent-contrast)] shadow-[var(--shadow-soft)]",
        )}
      >
        <Plus size={26} strokeWidth={2.2} />
      </button>

      <nav
        aria-label="Navigazione principale"
        className="safe-bottom fixed inset-x-0 bottom-0 z-30 border-t border-[var(--border)] bg-[var(--surface)]/92 backdrop-blur-xl md:hidden"
      >
        <div className="mx-auto grid max-w-2xl grid-cols-6">
          {LEFT.map(renderItem)}
          <span aria-hidden className="h-[58px]" />
          {RIGHT.map(renderItem)}
        </div>
      </nav>
    </>
  );
}

/** La stessa navigazione, in verticale, per schermi larghi. */
export function SideNav() {
  const isActive = useIsActive();
  const { open } = useComposer();

  return (
    <aside className="sticky top-0 hidden h-dvh w-[236px] flex-shrink-0 flex-col border-r border-[var(--border)] px-4 py-6 md:flex">
      <Link href="/" className="mb-6 flex items-center gap-2.5 px-2">
        <span className="flex h-9 w-9 items-center justify-center rounded-[11px] bg-[var(--accent)] text-[16px] font-bold text-[var(--accent-contrast)]">
          C
        </span>
        <span className="text-[17px] font-semibold tracking-[-0.01em]">Casa</span>
      </Link>

      <button
        type="button"
        onClick={() => open({ kind: "picker" })}
        className="press mb-5 flex h-11 items-center justify-center gap-2 rounded-[var(--radius-field)] bg-[var(--accent)] text-[15px] font-medium text-[var(--accent-contrast)]"
      >
        <Plus size={19} strokeWidth={2.2} />
        Aggiungi
      </button>

      <nav aria-label="Navigazione principale" className="flex flex-col gap-0.5">
        {ALL.map((item) => {
          const active = isActive(item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "press flex items-center gap-3 rounded-[12px] px-3 py-2.5 text-[15px]",
                active
                  ? "bg-[var(--accent-soft)] font-medium text-[var(--accent)]"
                  : "text-[var(--muted)] hover:bg-[var(--surface-2)]",
              )}
            >
              <Icon size={19} strokeWidth={active ? 2.2 : 1.8} />
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}

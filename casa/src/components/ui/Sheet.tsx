"use client";

import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/cn";

export interface SheetProps {
  open: boolean;
  onClose(): void;
  title?: string;
  /** Azione a destra nell'intestazione (di solito "Salva"). */
  action?: React.ReactNode;
  children: React.ReactNode;
  /** `auto` si adatta al contenuto, `tall` occupa quasi tutto lo schermo. */
  height?: "auto" | "tall";
}

/**
 * Pannello che sale dal basso. Su un telefono tenuto in una mano il pollice
 * sta in basso: le modali centrate costringono a cambiare presa.
 */
export function Sheet({ open, onClose, title, action, children, height = "auto" }: SheetProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    previouslyFocused.current = document.activeElement as HTMLElement;
    const original = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
        return;
      }
      if (e.key !== "Tab" || !panelRef.current) return;
      const focusables = panelRef.current.querySelectorAll<HTMLElement>(
        'a[href],button:not([disabled]),textarea,input,select,[tabindex]:not([tabindex="-1"])',
      );
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKey, true);
    // il primo campo prende il focus: si può scrivere senza toccare nulla
    const timer = window.setTimeout(() => {
      const target = panelRef.current?.querySelector<HTMLElement>("[data-autofocus]");
      target?.focus();
    }, 60);

    return () => {
      document.removeEventListener("keydown", onKey, true);
      document.body.style.overflow = original;
      window.clearTimeout(timer);
      previouslyFocused.current?.focus?.();
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center sm:items-center">
      <button
        aria-label="Chiudi"
        onClick={onClose}
        className="animate-fade absolute inset-0 bg-black/35 backdrop-blur-[2px]"
      />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={cn(
          "animate-sheet relative w-full sm:max-w-lg",
          "bg-[var(--surface)] border-t sm:border border-[var(--border)]",
          "rounded-t-[24px] sm:rounded-[24px] shadow-[var(--shadow-soft)]",
          "flex flex-col",
          height === "tall" ? "h-[92dvh] sm:h-[85vh]" : "max-h-[92dvh]",
        )}
      >
        <div className="flex-shrink-0 pt-2">
          <div className="mx-auto h-1 w-9 rounded-full bg-[var(--surface-3)]" />
        </div>

        {(title || action) && (
          <header className="flex flex-shrink-0 items-center gap-2 px-4 py-3">
            <button
              onClick={onClose}
              aria-label="Chiudi"
              className="press -ml-2 flex h-9 w-9 items-center justify-center rounded-full text-[var(--muted)] hover:bg-[var(--surface-2)]"
            >
              <X size={18} />
            </button>
            <h2 className="flex-1 truncate text-[17px] font-semibold">{title}</h2>
            {action}
          </header>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 pb-[max(20px,env(safe-area-inset-bottom))]">
          {children}
        </div>
      </div>
    </div>
  );
}

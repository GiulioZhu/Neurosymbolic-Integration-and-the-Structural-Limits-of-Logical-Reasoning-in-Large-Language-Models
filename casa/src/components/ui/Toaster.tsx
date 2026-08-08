"use client";

import { cn } from "@/lib/cn";
import { useStore } from "@/lib/data/store";

export function Toaster() {
  const { toasts, dismissToast } = useStore();
  if (toasts.length === 0) return null;

  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed inset-x-0 bottom-[calc(84px+env(safe-area-inset-bottom))] md:bottom-8 z-[60] flex flex-col items-center gap-2 px-4"
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          className={cn(
            "animate-rise pointer-events-auto flex w-full max-w-sm items-center gap-3 rounded-full px-4 py-2.5",
            "shadow-[var(--shadow-soft)] backdrop-blur",
            t.tone === "error"
              ? "bg-[var(--danger)] text-white"
              : "bg-[var(--text)] text-[var(--bg)]",
          )}
        >
          <span className="flex-1 truncate text-[14px]">{t.message}</span>
          {t.action && (
            <button
              onClick={() => {
                t.action?.run();
                dismissToast(t.id);
              }}
              className="press flex-shrink-0 text-[14px] font-semibold underline underline-offset-2"
            >
              {t.action.label}
            </button>
          )}
        </div>
      ))}
    </div>
  );
}

"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/cn";
import { useStore } from "@/lib/data/store";
import { Spinner } from "@/components/ui/Bits";
import { Toaster } from "@/components/ui/Toaster";
import { BottomNav, SideNav } from "./BottomNav";
import { ReminderScheduler } from "./ReminderScheduler";

/**
 * Guscio delle pagine autenticate: gestisce il gate di accesso, la barra
 * inferiore, i toast e lo scheduler dei promemoria.
 */
export function AppShell({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const { status } = useStore();
  const router = useRouter();

  useEffect(() => {
    if (status === "anonymous") router.replace("/login");
    if (status === "onboarding") router.replace("/onboarding");
  }, [status, router]);

  if (status !== "ready") {
    return (
      <div className="flex min-h-dvh items-center justify-center">
        <Spinner />
      </div>
    );
  }

  return (
    <div className="min-h-dvh md:flex">
      <SideNav />
      <div className="min-w-0 flex-1">
        <main
          className={cn(
            "mx-auto max-w-2xl px-5 pb-[calc(96px+env(safe-area-inset-bottom))] md:pb-12",
            "safe-top",
            className,
          )}
        >
          {children}
        </main>
      </div>
      <BottomNav />
      <Toaster />
      <ReminderScheduler />
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
}) {
  return (
    <header className="flex items-start justify-between gap-3 pt-5 pb-4">
      <div className="min-w-0">
        <h1 className="text-[26px] leading-tight font-semibold tracking-[-0.02em]">{title}</h1>
        {subtitle && <p className="mt-0.5 text-[14px] text-[var(--muted)]">{subtitle}</p>}
      </div>
      {right && <div className="flex flex-shrink-0 items-center gap-1 pt-1">{right}</div>}
    </header>
  );
}

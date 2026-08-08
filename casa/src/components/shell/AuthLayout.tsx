"use client";

import Link from "next/link";
import { Toaster } from "@/components/ui/Toaster";

/** Guscio delle pagine di accesso: niente barra inferiore, molta aria. */
export function AuthLayout({
  title,
  subtitle,
  children,
  footer,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
}) {
  return (
    <div className="flex min-h-dvh flex-col justify-center px-6 py-10">
      <div className="mx-auto w-full max-w-sm">
        <Link href="/" className="mb-8 flex items-center gap-2.5">
          <span className="flex h-10 w-10 items-center justify-center rounded-[12px] bg-[var(--accent)] text-[18px] font-bold text-[var(--accent-contrast)]">
            C
          </span>
          <span className="text-[18px] font-semibold tracking-[-0.01em]">Casa</span>
        </Link>

        <h1 className="text-[26px] leading-tight font-semibold tracking-[-0.02em]">{title}</h1>
        {subtitle && <p className="mt-1.5 mb-6 text-[15px] text-[var(--muted)]">{subtitle}</p>}
        {!subtitle && <div className="mb-6" />}

        {children}

        {footer && <div className="mt-6 text-center text-[14px] text-[var(--muted)]">{footer}</div>}
      </div>
      <Toaster />
    </div>
  );
}

"use client";

import { forwardRef, useId } from "react";
import { cn } from "@/lib/cn";

const BASE =
  "w-full bg-[var(--surface-2)] border border-[var(--border)] rounded-[var(--radius-field)] " +
  "px-3.5 py-2.5 text-[16px] text-[var(--text)] placeholder:text-[var(--faint)] " +
  "transition-colors focus:border-[var(--accent)] focus:bg-[var(--surface)] outline-none";

export function Label({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) {
  return (
    <label htmlFor={htmlFor} className="mb-1.5 block text-[13px] font-medium text-[var(--muted)]">
      {children}
    </label>
  );
}

export interface FieldProps {
  label?: string;
  hint?: string;
  children: React.ReactNode;
  className?: string;
}

export function Field({ label, hint, children, className }: FieldProps) {
  return (
    <div className={cn("mb-3.5", className)}>
      {label && <Label>{label}</Label>}
      {children}
      {hint && <p className="mt-1.5 text-[12px] text-[var(--faint)]">{hint}</p>}
    </div>
  );
}

export const Input = forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...rest }, ref) {
    return <input ref={ref} className={cn(BASE, className)} {...rest} />;
  },
);

export const Textarea = forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(function Textarea({ className, rows = 3, ...rest }, ref) {
  return <textarea ref={ref} rows={rows} className={cn(BASE, "resize-none", className)} {...rest} />;
});

export const Select = forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(
  function Select({ className, children, ...rest }, ref) {
    return (
      <select ref={ref} className={cn(BASE, "appearance-none pr-8", className)} {...rest}>
        {children}
      </select>
    );
  },
);

/** Riga etichetta + controllo, per i form densi (data, ora, priorità…). */
export function InlineField({
  label,
  children,
  icon,
}: {
  label: string;
  children: React.ReactNode;
  icon?: React.ReactNode;
}) {
  const id = useId();
  return (
    <div className="flex items-center gap-3 border-b border-[var(--border)] py-3 last:border-b-0">
      <div className="flex min-w-[104px] items-center gap-2 text-[14px] text-[var(--muted)]">
        {icon}
        <label htmlFor={id}>{label}</label>
      </div>
      <div className="flex-1 text-right">{children}</div>
    </div>
  );
}

/** Interruttore compatto, usato per privacy e ricorrenze. */
export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange(v: boolean): void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        "press relative h-[26px] w-[44px] flex-shrink-0 rounded-full transition-colors",
        checked ? "bg-[var(--accent)]" : "bg-[var(--surface-3)]",
      )}
    >
      <span
        className={cn(
          "absolute top-[3px] h-5 w-5 rounded-full bg-white shadow-sm transition-transform",
          checked ? "translate-x-[21px]" : "translate-x-[3px]",
        )}
      />
    </button>
  );
}

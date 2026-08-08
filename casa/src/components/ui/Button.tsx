"use client";

import { forwardRef } from "react";
import { cn } from "@/lib/cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-[var(--accent)] text-[var(--accent-contrast)] hover:bg-[var(--accent-hover)] border border-transparent",
  secondary:
    "bg-[var(--surface-2)] text-[var(--text)] hover:bg-[var(--surface-3)] border border-[var(--border)]",
  ghost: "bg-transparent text-[var(--muted)] hover:text-[var(--text)] border border-transparent",
  danger: "bg-transparent text-[var(--danger)] hover:bg-[var(--surface-2)] border border-transparent",
};

const SIZES: Record<Size, string> = {
  sm: "h-9 px-3 text-[13px] rounded-[10px]",
  md: "h-11 px-4 text-[15px] rounded-[var(--radius-field)]",
  lg: "h-13 px-5 text-[16px] rounded-[var(--radius-field)]",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  full?: boolean;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", full, loading, className, children, disabled, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(
        "press inline-flex items-center justify-center gap-2 font-medium select-none",
        "disabled:opacity-50 disabled:pointer-events-none",
        VARIANTS[variant],
        SIZES[size],
        full && "w-full",
        className,
      )}
      {...rest}
    >
      {loading && (
        <span
          aria-hidden
          className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent opacity-70"
        />
      )}
      {children}
    </button>
  );
});

/** Bottone circolare per le icone. 44px = area di tocco minima. */
export const IconButton = forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string }
>(function IconButton({ label, className, children, ...rest }, ref) {
  return (
    <button
      ref={ref}
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        "press inline-flex h-11 w-11 items-center justify-center rounded-full",
        "text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--surface-2)]",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
});

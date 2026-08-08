"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AuthLayout } from "@/components/shell/AuthLayout";
import { Button } from "@/components/ui/Button";
import { Field, Input } from "@/components/ui/Field";
import { useStore } from "@/lib/data/store";
import { friendlyError } from "@/lib/data/driver";
import { DEMO_CREDENTIALS } from "@/lib/data/local";

export default function LoginPage() {
  const { signIn, status, driverKind } = useStore();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (status === "ready") router.replace("/");
    if (status === "onboarding") router.replace("/onboarding");
  }, [status, router]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await signIn(email, password);
      router.replace("/");
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      title="Bentornato"
      subtitle="Entra nel vostro spazio."
      footer={
        <>
          Non hai un account?{" "}
          <Link href="/signup" className="font-medium text-[var(--accent)]">
            Registrati
          </Link>
        </>
      }
    >
      <form onSubmit={submit}>
        <Field label="Email">
          <Input
            type="email"
            autoComplete="email"
            inputMode="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="tu@esempio.it"
          />
        </Field>

        <Field label="Password">
          <Input
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
          />
        </Field>

        {error && <p className="mb-3 text-[13px] text-[var(--danger)]">{error}</p>}

        <Button type="submit" variant="primary" full size="lg" loading={loading}>
          Entra
        </Button>

        <Link
          href="/reset-password"
          className="mt-3 block text-center text-[14px] text-[var(--muted)]"
        >
          Password dimenticata?
        </Link>
      </form>

      {driverKind === "local" && (
        <div className="mt-8 rounded-[var(--radius-card)] border border-dashed border-[var(--border)] p-4 text-[13px] text-[var(--muted)]">
          <p className="mb-2">
            <strong className="text-[var(--text)]">Modalità demo</strong> — nessun Supabase
            configurato, i dati restano in questo browser.
          </p>
          <button
            type="button"
            onClick={() => {
              setEmail(DEMO_CREDENTIALS.email);
              setPassword(DEMO_CREDENTIALS.password);
            }}
            className="press font-medium text-[var(--accent)]"
          >
            Usa l&apos;account di prova ({DEMO_CREDENTIALS.email})
          </button>
        </div>
      )}
    </AuthLayout>
  );
}

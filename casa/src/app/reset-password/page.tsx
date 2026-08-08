"use client";

import { useState } from "react";
import Link from "next/link";
import { MailCheck } from "lucide-react";
import { AuthLayout } from "@/components/shell/AuthLayout";
import { Button } from "@/components/ui/Button";
import { Field, Input } from "@/components/ui/Field";
import { useStore } from "@/lib/data/store";
import { friendlyError } from "@/lib/data/driver";

export default function ResetPasswordPage() {
  const { requestPasswordReset, driverKind } = useStore();
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await requestPasswordReset(email);
      setSent(true);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  if (sent) {
    return (
      <AuthLayout title="Fatto">
        <div className="card p-5 text-center">
          <MailCheck size={30} className="mx-auto mb-3 text-[var(--accent)]" strokeWidth={1.5} />
          <p className="text-[15px]">
            Se esiste un account per <strong>{email}</strong>, riceverai un link per reimpostare la
            password.
          </p>
          {driverKind === "local" && (
            <p className="mt-2 text-[13px] text-[var(--faint)]">
              In modalità demo non viene inviata nessuna email.
            </p>
          )}
          <Link href="/login" className="mt-4 inline-block font-medium text-[var(--accent)]">
            Torna al login
          </Link>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Password dimenticata"
      subtitle="Ti mandiamo un link per sceglierne una nuova."
      footer={
        <Link href="/login" className="font-medium text-[var(--accent)]">
          Torna al login
        </Link>
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

        {error && <p className="mb-3 text-[13px] text-[var(--danger)]">{error}</p>}

        <Button type="submit" variant="primary" full size="lg" loading={loading}>
          Invia il link
        </Button>
      </form>
    </AuthLayout>
  );
}

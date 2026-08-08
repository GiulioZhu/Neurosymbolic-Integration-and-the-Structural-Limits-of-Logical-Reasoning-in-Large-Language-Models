"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { MailCheck } from "lucide-react";
import { AuthLayout } from "@/components/shell/AuthLayout";
import { Button } from "@/components/ui/Button";
import { Field, Input } from "@/components/ui/Field";
import { useStore } from "@/lib/data/store";
import { friendlyError } from "@/lib/data/driver";

export default function SignupPage() {
  const { signUp } = useStore();
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await signUp(email, password, name.trim());
      if (res.needsConfirmation) setSent(true);
      else router.replace("/onboarding");
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  if (sent) {
    return (
      <AuthLayout title="Controlla la posta">
        <div className="card p-5 text-center">
          <MailCheck size={30} className="mx-auto mb-3 text-[var(--accent)]" strokeWidth={1.5} />
          <p className="text-[15px]">
            Abbiamo mandato un link di conferma a <strong>{email}</strong>.
          </p>
          <p className="mt-2 text-[13px] text-[var(--muted)]">
            Aprilo per attivare l&apos;account, poi torna qui.
          </p>
          <Link href="/login" className="mt-4 inline-block font-medium text-[var(--accent)]">
            Vai al login
          </Link>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Creiamo il vostro spazio"
      subtitle="Un account per ciascuno, un posto solo per tutto."
      footer={
        <>
          Hai già un account?{" "}
          <Link href="/login" className="font-medium text-[var(--accent)]">
            Accedi
          </Link>
        </>
      }
    >
      <form onSubmit={submit}>
        <Field label="Come ti chiami">
          <Input
            required
            autoComplete="given-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Luca"
          />
        </Field>

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

        <Field label="Password" hint="Almeno 6 caratteri.">
          <Input
            type="password"
            autoComplete="new-password"
            required
            minLength={6}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
          />
        </Field>

        {error && <p className="mb-3 text-[13px] text-[var(--danger)]">{error}</p>}

        <Button type="submit" variant="primary" full size="lg" loading={loading}>
          Crea account
        </Button>
      </form>
    </AuthLayout>
  );
}

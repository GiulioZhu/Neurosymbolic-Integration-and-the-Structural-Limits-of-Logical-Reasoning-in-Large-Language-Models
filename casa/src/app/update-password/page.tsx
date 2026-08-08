"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { AuthLayout } from "@/components/shell/AuthLayout";
import { Button } from "@/components/ui/Button";
import { Field, Input } from "@/components/ui/Field";
import { useStore } from "@/lib/data/store";
import { friendlyError } from "@/lib/data/driver";

/**
 * Atterraggio del link ricevuto per email. Supabase mette la sessione di
 * recupero nell'URL e il client la raccoglie da solo (`detectSessionInUrl`):
 * qui resta solo da scegliere la nuova password.
 */
export default function UpdatePasswordPage() {
  const { updatePassword, pushToast } = useStore();
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirm) {
      setError("Le due password non coincidono.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      await updatePassword(password);
      pushToast("Password aggiornata");
      router.replace("/");
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout title="Nuova password" subtitle="Scegline una che ricordi.">
      <form onSubmit={submit}>
        <Field label="Password" hint="Almeno 6 caratteri.">
          <Input
            type="password"
            autoComplete="new-password"
            required
            minLength={6}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </Field>

        <Field label="Ripeti la password">
          <Input
            type="password"
            autoComplete="new-password"
            required
            minLength={6}
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
          />
        </Field>

        {error && <p className="mb-3 text-[13px] text-[var(--danger)]">{error}</p>}

        <Button type="submit" variant="primary" full size="lg" loading={loading}>
          Salva
        </Button>
      </form>
    </AuthLayout>
  );
}

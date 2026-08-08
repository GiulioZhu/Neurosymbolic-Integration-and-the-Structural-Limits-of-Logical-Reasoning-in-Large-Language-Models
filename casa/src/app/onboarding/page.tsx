"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Home, KeyRound } from "lucide-react";
import { AuthLayout } from "@/components/shell/AuthLayout";
import { Button } from "@/components/ui/Button";
import { Field, Input } from "@/components/ui/Field";
import { Segmented } from "@/components/ui/Segmented";
import { Spinner } from "@/components/ui/Bits";
import { useStore } from "@/lib/data/store";
import { friendlyError } from "@/lib/data/driver";

export default function OnboardingPage() {
  const { status, createHousehold, joinHousehold, profile } = useStore();
  const router = useRouter();
  const [mode, setMode] = useState<"create" | "join">("create");
  const [name, setName] = useState("Casa");
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (status === "anonymous") router.replace("/login");
    if (status === "ready") router.replace("/");
  }, [status, router]);

  if (status === "loading") {
    return (
      <div className="flex min-h-dvh items-center justify-center">
        <Spinner />
      </div>
    );
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === "create") await createHousehold(name);
      else await joinHousehold(code);
      router.replace("/");
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      title={`Ciao${profile?.display_name ? `, ${profile.display_name}` : ""}`}
      subtitle="Ultimo passaggio: lo spazio che condividerete."
    >
      <Segmented
        className="mb-5"
        value={mode}
        onChange={setMode}
        options={[
          { value: "create", label: "Crea uno spazio" },
          { value: "join", label: "Entra con un codice" },
        ]}
      />

      <form onSubmit={submit}>
        {mode === "create" ? (
          <Field label="Come lo chiamiamo?" hint="Potrai invitare il partner subito dopo.">
            <Input
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Casa"
            />
          </Field>
        ) : (
          <Field label="Codice di invito" hint="Sei caratteri, te lo passa chi ha creato lo spazio.">
            <Input
              required
              value={code}
              onChange={(e) => setCode(e.target.value.toUpperCase())}
              placeholder="K7M2QP"
              maxLength={6}
              autoCapitalize="characters"
              className="text-center text-[22px] font-semibold tracking-[0.3em]"
            />
          </Field>
        )}

        {error && <p className="mb-3 text-[13px] text-[var(--danger)]">{error}</p>}

        <Button type="submit" variant="primary" full size="lg" loading={loading}>
          {mode === "create" ? (
            <>
              <Home size={17} /> Crea lo spazio
            </>
          ) : (
            <>
              <KeyRound size={17} /> Entra
            </>
          )}
        </Button>
      </form>
    </AuthLayout>
  );
}

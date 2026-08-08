"use client";

import { useState } from "react";
import { Bell, Check, Copy, LogOut, Mail, RefreshCw } from "lucide-react";
import { AppShell, PageHeader } from "@/components/shell/AppShell";
import { Avatar, Card, SectionHeader, ACCENT_BG } from "@/components/ui/Bits";
import { Button } from "@/components/ui/Button";
import { Field, Input } from "@/components/ui/Field";
import { Chip, Segmented } from "@/components/ui/Segmented";
import { cn } from "@/lib/cn";
import { useStore } from "@/lib/data/store";
import { useTheme, type ThemePreference } from "@/lib/theme";
import { friendlyError } from "@/lib/data/driver";
import {
  pushConfigured,
  requestPermission,
  subscribeToPush,
  useNotificationPermission,
} from "@/lib/notifications";

export default function SettingsPage() {
  return (
    <AppShell>
      <Settings />
    </AppShell>
  );
}

function Settings() {
  const {
    profile,
    household,
    members,
    user,
    driverKind,
    saveProfile,
    rotateInviteCode,
    signOut,
    pushToast,
    driver,
  } = useStore();
  const { preference, setPreference } = useTheme();

  const [name, setName] = useState(profile?.display_name ?? "");
  const [accent, setAccent] = useState(profile?.accent ?? "clay");
  const [savingProfile, setSavingProfile] = useState(false);
  const [copied, setCopied] = useState(false);
  const permission = useNotificationPermission();

  const dirty = profile && (name !== profile.display_name || accent !== profile.accent);

  const onSaveProfile = async () => {
    setSavingProfile(true);
    try {
      await saveProfile({ display_name: name.trim() || "Io", accent });
      pushToast("Profilo aggiornato");
    } catch (err) {
      pushToast(friendlyError(err), { tone: "error" });
    } finally {
      setSavingProfile(false);
    }
  };

  const copyCode = async () => {
    if (!household) return;
    try {
      await navigator.clipboard.writeText(household.invite_code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      pushToast("Copia non riuscita, seleziona il codice a mano", { tone: "error" });
    }
  };

  const inviteByEmail = () => {
    if (!household) return;
    const subject = encodeURIComponent(`Ti invito su Casa — spazio “${household.name}”`);
    const body = encodeURIComponent(
      `Ciao!\n\nHo creato il nostro spazio condiviso su Casa.\n` +
        `Registrati e inserisci questo codice di invito: ${household.invite_code}\n\n` +
        `${typeof window !== "undefined" ? window.location.origin : ""}/signup\n`,
    );
    window.location.href = `mailto:?subject=${subject}&body=${body}`;
  };

  const enableNotifications = async () => {
    const state = await requestPermission();
    if (state !== "granted") {
      pushToast("Permesso negato: puoi riattivarlo dalle impostazioni del browser", {
        tone: "error",
      });
      return;
    }
    pushToast("Promemoria attivi");

    if (pushConfigured && driver.savePushSubscription && household && user) {
      try {
        const sub = await subscribeToPush();
        if (sub) {
          await driver.savePushSubscription(user.id, household.id, {
            ...sub,
            userAgent: navigator.userAgent,
          });
        }
      } catch {
        /* le notifiche in-app funzionano comunque */
      }
    }
  };

  return (
    <>
      <PageHeader title="Impostazioni" />

      {/* --- profilo --- */}
      <SectionHeader title="Profilo" />
      <Card className="mb-5 p-4">
        <div className="mb-4 flex items-center gap-3">
          <Avatar name={name || "?"} accent={accent} size={48} />
          <div className="min-w-0 flex-1">
            <p className="truncate text-[15px] font-medium">{name || "Senza nome"}</p>
            <p className="truncate text-[13px] text-[var(--muted)]">{user?.email}</p>
          </div>
        </div>

        <Field label="Nome">
          <Input value={name} onChange={(e) => setName(e.target.value)} />
        </Field>

        <Field label="Colore">
          <div className="flex flex-wrap gap-2">
            {Object.entries(ACCENT_BG).map(([key, color]) => (
              <button
                key={key}
                onClick={() => setAccent(key)}
                aria-label={`Colore ${key}`}
                aria-pressed={accent === key}
                className={cn(
                  "press h-8 w-8 rounded-full border-2 transition-transform",
                  accent === key ? "border-[var(--text)]" : "border-transparent",
                )}
                style={{ background: color }}
              />
            ))}
          </div>
        </Field>

        <Button variant="primary" full onClick={onSaveProfile} loading={savingProfile} disabled={!dirty}>
          Salva profilo
        </Button>
      </Card>

      {/* --- spazio --- */}
      <SectionHeader title="Spazio condiviso" />
      <Card className="mb-5 p-4">
        <p className="text-[15px] font-medium">{household?.name}</p>
        <p className="mb-3 text-[13px] text-[var(--muted)]">
          {members.length === 1 ? "Sei da solo qui" : `${members.length} persone`}
        </p>

        <div className="mb-3 flex flex-wrap gap-2">
          {members.map((m) => (
            <span
              key={m.id}
              className="flex items-center gap-2 rounded-full bg-[var(--surface-2)] py-1 pr-3 pl-1"
            >
              <Avatar name={m.display_name} accent={m.accent} size={22} />
              <span className="text-[13px]">{m.display_name}</span>
            </span>
          ))}
        </div>

        <p className="mb-1.5 text-[13px] text-[var(--muted)]">Codice di invito</p>
        <div className="flex items-center gap-2">
          <code className="flex-1 rounded-[var(--radius-field)] bg-[var(--surface-2)] px-3.5 py-2.5 text-center text-[20px] font-semibold tracking-[0.25em]">
            {household?.invite_code}
          </code>
          <Button onClick={copyCode} aria-label="Copia codice">
            {copied ? <Check size={17} /> : <Copy size={17} />}
          </Button>
        </div>

        <div className="mt-2 flex gap-2">
          <Button full onClick={inviteByEmail}>
            <Mail size={16} /> Invita per email
          </Button>
          <Button
            onClick={async () => {
              const code = await rotateInviteCode();
              pushToast(`Nuovo codice: ${code}`);
            }}
            aria-label="Rigenera codice"
          >
            <RefreshCw size={16} />
          </Button>
        </div>
      </Card>

      {/* --- notifiche --- */}
      <SectionHeader title="Notifiche" />
      <Card className="mb-5 p-4">
        <div className="flex items-start gap-3">
          <Bell size={18} className="mt-0.5 flex-shrink-0 text-[var(--muted)]" />
          <div className="min-w-0 flex-1">
            <p className="text-[14px]">
              {permission === "granted"
                ? "I promemoria sono attivi su questo dispositivo."
                : permission === "denied"
                  ? "Le notifiche sono bloccate dal browser."
                  : permission === "unsupported"
                    ? "Questo browser non supporta le notifiche."
                    : "Attiva i promemoria per ricevere gli avvisi."}
            </p>
            {permission === "granted" && !pushConfigured && (
              <p className="mt-1 text-[12px] text-[var(--faint)]">
                Gli avvisi arrivano finché l&apos;app è aperta. Per riceverli anche ad app chiusa
                servono le chiavi VAPID.
              </p>
            )}
          </div>
        </div>
        {permission === "default" && (
          <Button variant="primary" full className="mt-3" onClick={enableNotifications}>
            Attiva i promemoria
          </Button>
        )}
      </Card>

      {/* --- aspetto --- */}
      <SectionHeader title="Aspetto" />
      <Card className="mb-5 p-4">
        <Segmented
          value={preference}
          onChange={(v) => setPreference(v as ThemePreference)}
          options={[
            { value: "system", label: "Sistema" },
            { value: "light", label: "Chiaro" },
            { value: "dark", label: "Scuro" },
          ]}
        />
      </Card>

      {driverKind === "local" && (
        <Card className="mb-5 border-dashed p-4">
          <p className="text-[13px] text-[var(--muted)]">
            <strong className="text-[var(--text)]">Modalità demo.</strong> I dati restano in questo
            browser. Configura Supabase (vedi <code>.env.example</code>) per usare l&apos;app in due,
            con sincronizzazione reale.
          </p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            <Chip
              onClick={() => {
                window.localStorage.removeItem("casa:db:v1");
                window.location.reload();
              }}
            >
              Ripristina i dati di esempio
            </Chip>
          </div>
        </Card>
      )}

      <Button variant="danger" full className="mb-6" onClick={() => void signOut()}>
        <LogOut size={16} /> Esci
      </Button>
    </>
  );
}

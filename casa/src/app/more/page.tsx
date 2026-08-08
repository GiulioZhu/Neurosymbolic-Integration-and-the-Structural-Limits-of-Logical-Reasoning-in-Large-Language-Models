"use client";

import { useMemo } from "react";
import Link from "next/link";
import { BarChart3, ChevronRight, Inbox, Search, Settings } from "lucide-react";
import { AppShell, PageHeader } from "@/components/shell/AppShell";
import { Card, SectionHeader } from "@/components/ui/Bits";
import { useStore } from "@/lib/data/store";
import { money } from "@/lib/format";
import { weekSummary } from "@/lib/data/selectors";

export default function MorePage() {
  return (
    <AppShell>
      <More />
    </AppShell>
  );
}

function More() {
  const { data } = useStore();
  const week = useMemo(() => weekSummary(data), [data]);
  const inboxCount = data.notes.filter((n) => !n.archived_at && !n.converted_kind).length;

  const links = [
    { href: "/inbox", label: "Inbox", icon: Inbox, badge: inboxCount || undefined },
    { href: "/search", label: "Ricerca", icon: Search },
    { href: "/stats", label: "Statistiche", icon: BarChart3 },
    { href: "/settings", label: "Impostazioni", icon: Settings },
  ];

  return (
    <>
      <PageHeader title="Altro" />

      <Card className="mb-6 divide-y divide-[var(--border)]">
        {links.map(({ href, label, icon: Icon, badge }) => (
          <Link
            key={href}
            href={href}
            className="press flex items-center gap-3 px-4 py-3.5 text-[15px]"
          >
            <Icon size={18} className="flex-shrink-0 text-[var(--muted)]" />
            <span className="flex-1">{label}</span>
            {badge !== undefined && (
              <span className="rounded-full bg-[var(--accent-soft)] px-2 py-0.5 text-[12px] font-medium text-[var(--accent)]">
                {badge}
              </span>
            )}
            <ChevronRight size={16} className="text-[var(--faint)]" />
          </Link>
        ))}
      </Card>

      <SectionHeader title="Questa settimana" />
      <Card className="grid grid-cols-2 gap-px overflow-hidden bg-[var(--border)]">
        <Metric label="Completate" value={String(week.completed.length)} />
        <Metric label="Ancora da fare" value={String(week.open.length)} />
        <Metric label="Eventi" value={String(week.events.length)} />
        <Metric label="Spese" value={money(week.spend)} />
      </Card>

      {week.topCategories.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5 px-1">
          {week.topCategories.map((c) => (
            <span
              key={c.slug}
              className="rounded-full px-2.5 py-1 text-[12px]"
              style={{ background: `${c.color}1F`, color: c.color }}
            >
              {c.name} · {money(c.total)}
            </span>
          ))}
        </div>
      )}
    </>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[var(--surface)] px-4 py-3.5">
      <p className="text-[12px] text-[var(--faint)]">{label}</p>
      <p className="tnum mt-0.5 text-[20px] leading-tight font-semibold">{value}</p>
    </div>
  );
}

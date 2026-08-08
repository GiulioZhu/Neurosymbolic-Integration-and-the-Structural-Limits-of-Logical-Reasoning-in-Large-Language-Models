"use client";

import { useMemo } from "react";
import Link from "next/link";
import { AlertTriangle, ArrowRight, CalendarDays, CheckCircle2, Moon, Sun } from "lucide-react";
import { AppShell } from "@/components/shell/AppShell";
import { QuickCapture } from "@/components/capture/QuickCapture";
import { EventRow, RowList, TaskRow } from "@/components/items/Rows";
import { Avatar, Card, EmptyState, SectionHeader, StatTile } from "@/components/ui/Bits";
import { IconButton } from "@/components/ui/Button";
import { DayBars } from "@/components/charts/Charts";
import { useStore } from "@/lib/data/store";
import { useTheme } from "@/lib/theme";
import { formatLongDate, formatMonthName, monthRange } from "@/lib/date";
import { count, firstName, greeting, money, percentDelta } from "@/lib/format";
import { dailyTotals, expensesIn, todayHeadline, todaySummary } from "@/lib/data/selectors";

export default function HomePage() {
  return (
    <AppShell>
      <Dashboard />
    </AppShell>
  );
}

function Dashboard() {
  const { data, profile } = useStore();
  const { resolved, toggle } = useTheme();
  const now = useMemo(() => new Date(), []);
  const summary = useMemo(() => todaySummary(data, now), [data, now]);

  const monthDays = useMemo(() => {
    const range = monthRange(now);
    return dailyTotals(expensesIn(data.expenses, range), range);
  }, [data.expenses, now]);

  const delta = percentDelta(summary.monthSpend, summary.prevMonthSpend);
  const todayIndex = now.getDate() - 1;

  return (
    <>
      <header className="flex items-start justify-between gap-3 pt-5 pb-3">
        <div className="min-w-0">
          <p className="text-[13px] font-medium text-[var(--muted)] first-letter:uppercase">
            {formatLongDate(now)}
          </p>
          <h1 className="mt-0.5 text-[26px] leading-tight font-semibold tracking-[-0.02em]">
            {greeting(now)}, {firstName(profile?.display_name ?? "")}
          </h1>
          <p className="mt-0.5 text-[14px] text-[var(--muted)]">{todayHeadline(summary)}</p>
        </div>
        <div className="flex flex-shrink-0 items-center">
          <IconButton
            label={resolved === "dark" ? "Passa al tema chiaro" : "Passa al tema scuro"}
            onClick={toggle}
          >
            {resolved === "dark" ? <Sun size={18} /> : <Moon size={18} />}
          </IconButton>
          <Link href="/settings" aria-label="Impostazioni" className="press ml-1">
            <Avatar
              name={profile?.display_name ?? "?"}
              accent={profile?.accent ?? "clay"}
              size={32}
            />
          </Link>
        </div>
      </header>

      <div className="mb-4">
        <QuickCapture />
      </div>

      <div className="mb-5 flex gap-2">
        <StatTile
          label="In ritardo"
          value={String(summary.overdue.length)}
          tone={summary.overdue.length ? "danger" : "default"}
        />
        <StatTile
          label="Urgenti"
          value={String(summary.urgent.length)}
          tone={summary.urgent.length ? "accent" : "default"}
        />
        <StatTile label="Oggi" value={money(summary.todaySpend)} />
      </div>

      {summary.overdue.length > 0 && (
        <section className="mb-5">
          <SectionHeader title="In ritardo" right={count(summary.overdue.length, "attività", "attività")} />
          <RowList>
            {summary.overdue.slice(0, 4).map((t) => (
              <TaskRow key={t.id} task={t} showDate />
            ))}
          </RowList>
        </section>
      )}

      <section className="mb-5">
        <SectionHeader
          title="Oggi"
          right={summary.tasks.length > 0 ? `${summary.doneTasks.length}/${summary.tasks.length}` : undefined}
        />
        {summary.tasks.length === 0 ? (
          <Card>
            <EmptyState
              icon={<CheckCircle2 size={30} strokeWidth={1.5} />}
              title="Niente in programma"
              description="Scrivi qualcosa nel campo qui sopra e comparirà qui."
            />
          </Card>
        ) : (
          <RowList>
            {summary.tasks.map((t) => (
              <TaskRow key={t.id} task={t} />
            ))}
          </RowList>
        )}
      </section>

      {summary.events.length > 0 && (
        <section className="mb-5">
          <SectionHeader title="Eventi di oggi" />
          <RowList>
            {summary.events.map((o) => (
              <EventRow key={`${o.event.id}:${o.date}`} event={o.event} date={o.date} />
            ))}
          </RowList>
        </section>
      )}

      {summary.upcoming.length > 0 && (
        <section className="mb-5">
          <SectionHeader
            title="In scadenza"
            right={
              <Link href="/tasks" className="inline-flex items-center gap-1 text-[var(--accent)]">
                Tutte <ArrowRight size={12} />
              </Link>
            }
          />
          <RowList>
            {summary.upcoming.slice(0, 4).map((t) => (
              <TaskRow key={t.id} task={t} showDate />
            ))}
          </RowList>
        </section>
      )}

      {(summary.next || summary.nextEvent) && (
        <section className="mb-5">
          <SectionHeader title="Prossimo" />
          <Card className="divide-y divide-[var(--border)]">
            {summary.next && (
              <div className="flex items-center gap-3 px-4 py-3">
                <CheckCircle2 size={16} className="flex-shrink-0 text-[var(--kind-task)]" />
                <span className="min-w-0 flex-1 truncate text-[14px]">{summary.next.title}</span>
                <span className="flex-shrink-0 text-[13px] text-[var(--muted)]">
                  {summary.next.due_time ?? "senza ora"}
                </span>
              </div>
            )}
            {summary.nextEvent && (
              <div className="flex items-center gap-3 px-4 py-3">
                <CalendarDays size={16} className="flex-shrink-0 text-[var(--kind-event)]" />
                <span className="min-w-0 flex-1 truncate text-[14px]">
                  {summary.nextEvent.event.title}
                </span>
                <span className="flex-shrink-0 text-[13px] text-[var(--muted)]">
                  {summary.nextEvent.event.start_time ?? "tutto il giorno"}
                </span>
              </div>
            )}
          </Card>
        </section>
      )}

      <section className="mb-5">
        <SectionHeader
          title="Spese"
          right={
            <Link href="/expenses" className="inline-flex items-center gap-1 text-[var(--accent)]">
              Dettagli <ArrowRight size={12} />
            </Link>
          }
        />
        <Card className="p-4">
          <div className="flex items-end justify-between">
            <div>
              <p className="text-[12px] text-[var(--faint)]">Oggi</p>
              <p className="tnum text-[22px] leading-tight font-semibold">
                {money(summary.todaySpend)}
              </p>
            </div>
            <div className="text-right">
              <p className="text-[12px] text-[var(--faint)] first-letter:uppercase">
                {formatMonthName(now)}
              </p>
              <p className="tnum text-[22px] leading-tight font-semibold">
                {money(summary.monthSpend)}
              </p>
            </div>
          </div>
          <DayBars
            className="mt-3"
            values={monthDays.map((d) => d.total)}
            highlightIndex={todayIndex}
            ariaLabel="Spese giorno per giorno nel mese corrente"
          />
          {delta && (
            <p className="mt-2 flex items-center gap-1.5 text-[12px] text-[var(--muted)]">
              {delta.startsWith("+") && <AlertTriangle size={12} className="text-[var(--accent)]" />}
              {delta} rispetto allo stesso periodo del mese scorso
            </p>
          )}
        </Card>
      </section>
    </>
  );
}

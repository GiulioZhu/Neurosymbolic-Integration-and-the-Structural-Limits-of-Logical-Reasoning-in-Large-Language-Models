"use client";

import { useMemo } from "react";
import { AppShell, PageHeader } from "@/components/shell/AppShell";
import { Card, SectionHeader } from "@/components/ui/Bits";
import { Donut, Legend, MonthBars } from "@/components/charts/Charts";
import { useStore } from "@/lib/data/store";
import { formatDayMonth, monthRange, formatMonthName } from "@/lib/date";
import { count, money } from "@/lib/format";
import { expensesIn, sum, totalsByCategory, weekSummary } from "@/lib/data/selectors";

export default function StatsPage() {
  return (
    <AppShell>
      <Stats />
    </AppShell>
  );
}

function Stats() {
  const { data } = useStore();
  const now = useMemo(() => new Date(), []);
  const week = useMemo(() => weekSummary(data, now), [data, now]);

  const lastSixMonths = useMemo(() => {
    return Array.from({ length: 6 }, (_, i) => {
      const d = new Date(now.getFullYear(), now.getMonth() - (5 - i), 1);
      const range = monthRange(d);
      return {
        label: formatMonthName(d).slice(0, 3),
        value: sum(expensesIn(data.expenses, range)),
      };
    });
  }, [data.expenses, now]);

  const weekCategories = useMemo(() => {
    const rows = expensesIn(data.expenses, week.range);
    return totalsByCategory(rows, data.categories).slice(0, 5);
  }, [data.categories, data.expenses, week.range]);

  const weekTotal = week.spend;
  const done = week.completed.length;
  const open = week.open.length;
  const completion = done + open > 0 ? Math.round((done / (done + open)) * 100) : 0;

  return (
    <>
      <PageHeader
        title="Statistiche"
        subtitle={`Settimana del ${formatDayMonth(week.range.start)} – ${formatDayMonth(week.range.end)}`}
      />

      <SectionHeader title="Riepilogo settimanale" />
      <Card className="mb-5 p-4">
        <div className="mb-4 flex items-center gap-4">
          <Donut
            size={92}
            thickness={11}
            center={`${completion}%`}
            slices={[
              { label: "Completate", value: done, color: "var(--kind-task)" },
              { label: "Da fare", value: open, color: "var(--surface-3)" },
            ]}
          />
          <div className="min-w-0 flex-1 space-y-1.5 text-[14px]">
            <Line label="Completate" value={String(done)} />
            <Line label="Ancora da fare" value={String(open)} />
            <Line label="Eventi" value={String(week.events.length)} />
            <Line label="Spese" value={money(weekTotal)} />
          </div>
        </div>
        <p className="text-[13px] text-[var(--muted)]">
          {done === 0 && open === 0
            ? "Settimana ancora tutta da scrivere."
            : `Avete chiuso ${count(done, "attività", "attività")} su ${done + open}.`}
        </p>
      </Card>

      {weekCategories.length > 0 && (
        <>
          <SectionHeader title="Dove sono andati i soldi" />
          <Card className="mb-5 flex items-center gap-4 p-4">
            <Donut
              size={110}
              slices={weekCategories.map((c) => ({
                label: c.name,
                value: c.total,
                color: c.color,
              }))}
              center={money(weekTotal)}
            />
            <Legend
              slices={weekCategories.map((c) => ({
                label: c.name,
                value: c.total,
                color: c.color,
              }))}
              total={weekTotal}
            />
          </Card>
        </>
      )}

      <SectionHeader title="Ultimi 6 mesi" />
      <Card className="mb-5 p-4">
        <MonthBars items={lastSixMonths} />
      </Card>
    </>
  );
}

function Line({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="truncate text-[var(--muted)]">{label}</span>
      <span className="tnum font-medium">{value}</span>
    </div>
  );
}

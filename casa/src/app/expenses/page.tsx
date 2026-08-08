"use client";

import { useMemo, useState } from "react";
import { Wallet } from "lucide-react";
import { AppShell, PageHeader } from "@/components/shell/AppShell";
import { ExpenseRow, RowList } from "@/components/items/Rows";
import { Card, EmptyState, SectionHeader } from "@/components/ui/Bits";
import { Segmented } from "@/components/ui/Segmented";
import { Donut, Legend, SplitBar } from "@/components/charts/Charts";
import { useStore } from "@/lib/data/store";
import { dayRange, formatMonthName, monthRange, weekRange, yearRange } from "@/lib/date";
import { money, percentDelta } from "@/lib/format";
import {
  comparableRange,
  expensesIn,
  sum,
  totalsByCategory,
  totalsByPerson,
} from "@/lib/data/selectors";
import { ACCENT_BG } from "@/components/ui/Bits";

type Period = "day" | "week" | "month" | "year";

const PERIOD_LABEL: Record<Period, string> = {
  day: "Oggi",
  week: "Settimana",
  month: "Mese",
  year: "Anno",
};

export default function ExpensesPage() {
  return (
    <AppShell>
      <Expenses />
    </AppShell>
  );
}

function Expenses() {
  const { data, members } = useStore();
  const [period, setPeriod] = useState<Period>("month");
  const now = useMemo(() => new Date(), []);

  const range = useMemo(() => {
    switch (period) {
      case "day":
        return dayRange(now);
      case "week":
        return weekRange(now);
      case "year":
        return yearRange(now);
      default:
        return monthRange(now);
    }
  }, [period, now]);

  const rows = useMemo(() => expensesIn(data.expenses, range), [data.expenses, range]);
  const total = sum(rows);
  const categories = useMemo(() => totalsByCategory(rows, data.categories), [rows, data.categories]);
  const people = useMemo(() => totalsByPerson(rows), [rows]);

  const prevTotal = useMemo(
    () => (period === "month" ? sum(expensesIn(data.expenses, comparableRange(now))) : 0),
    [data.expenses, now, period],
  );
  const delta = period === "month" ? percentDelta(total, prevTotal) : null;

  const slices = categories.slice(0, 6).map((c) => ({ label: c.name, value: c.total, color: c.color }));
  const otherTotal = categories.slice(6).reduce((a, c) => a + c.total, 0);
  if (otherTotal > 0) slices.push({ label: "Altre", value: otherTotal, color: "var(--surface-3)" });

  return (
    <>
      <PageHeader title="Spese" />

      <Segmented
        className="mb-5"
        value={period}
        onChange={setPeriod}
        options={(Object.keys(PERIOD_LABEL) as Period[]).map((p) => ({
          value: p,
          label: PERIOD_LABEL[p],
        }))}
      />

      <div className="mb-6 text-center">
        <p className="tnum text-[38px] leading-none font-semibold tracking-[-0.02em]">
          {money(total)}
        </p>
        <p className="mt-1.5 text-[13px] text-[var(--muted)]">
          <span className="first-letter:uppercase">
            {period === "month" ? formatMonthName(now) : PERIOD_LABEL[period].toLowerCase()}
          </span>
          {delta && <> · {delta} sul mese scorso</>}
          {rows.length > 0 && <> · {rows.length} movimenti</>}
        </p>
      </div>

      {rows.length === 0 ? (
        <Card>
          <EmptyState
            icon={<Wallet size={30} strokeWidth={1.5} />}
            title="Nessuna spesa"
            description="Tocca il + per registrarne una: bastano un numero e una categoria."
          />
        </Card>
      ) : (
        <>
          <section className="mb-5">
            <SectionHeader title="Per categoria" />
            <Card className="flex items-center gap-4 p-4">
              <Donut slices={slices} size={112} thickness={13} center={money(total)} />
              <Legend slices={slices} total={total} />
            </Card>
          </section>

          {people.length > 0 && (
            <section className="mb-5">
              <SectionHeader title="Chi ha pagato" />
              <Card className="p-4">
                <SplitBar
                  parts={people.map((p, i) => {
                    const m = members.find((x) => x.id === p.userId);
                    return {
                      label: m?.display_name ?? "Altro",
                      value: p.total,
                      color: ACCENT_BG[m?.accent ?? ""] ?? ["#C25A38", "#5E8C61", "#7A6BB5"][i % 3],
                    };
                  })}
                />
              </Card>
            </section>
          )}

          <section className="mb-5">
            <SectionHeader title="Movimenti" right={String(rows.length)} />
            <RowList>
              {rows.slice(0, 60).map((e) => (
                <ExpenseRow key={e.id} expense={e} />
              ))}
            </RowList>
          </section>
        </>
      )}
    </>
  );
}

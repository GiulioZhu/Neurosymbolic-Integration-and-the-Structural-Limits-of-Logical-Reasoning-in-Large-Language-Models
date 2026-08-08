"use client";

import { useMemo, useState } from "react";
import { CheckCircle2, SlidersHorizontal } from "lucide-react";
import { AppShell, PageHeader } from "@/components/shell/AppShell";
import { RowList, TaskRow } from "@/components/items/Rows";
import { Card, EmptyState, SectionHeader } from "@/components/ui/Bits";
import { IconButton } from "@/components/ui/Button";
import { Chip, Segmented } from "@/components/ui/Segmented";
import { useStore } from "@/lib/data/store";
import { relativeDay, todayISO } from "@/lib/date";
import {
  byTaskOrder,
  isOpen,
  tasksOverdue,
  tasksSomeday,
  tasksToday,
  tasksUpcoming,
} from "@/lib/data/selectors";
import { PRIORITY_LABEL, type Priority, type Task, type UUID } from "@/lib/types";

type Tab = "today" | "upcoming" | "late" | "done";

export default function TasksPage() {
  return (
    <AppShell>
      <Tasks />
    </AppShell>
  );
}

function Tasks() {
  const { data, members, user } = useStore();
  const [tab, setTab] = useState<Tab>("today");
  const [showFilters, setShowFilters] = useState(false);
  const [assignee, setAssignee] = useState<UUID | "all">("all");
  const [priority, setPriority] = useState<Priority | "all">("all");

  const today = todayISO();

  const filtered = useMemo(() => {
    let rows = data.tasks;
    if (assignee !== "all") rows = rows.filter((t) => t.assigned_to === assignee);
    if (priority !== "all") rows = rows.filter((t) => t.priority === priority);
    return rows;
  }, [assignee, data.tasks, priority]);

  const counts = useMemo(
    () => ({
      today: tasksToday(filtered, today).filter(isOpen).length,
      upcoming: tasksUpcoming(filtered, 30, today).length,
      late: tasksOverdue(filtered, today).length,
    }),
    [filtered, today],
  );

  const groups = useMemo(() => buildGroups(tab, filtered, today), [tab, filtered, today]);
  const empty = groups.every((g) => g.tasks.length === 0);

  return (
    <>
      <PageHeader
        title="Attività"
        right={
          <IconButton
            label="Filtri"
            onClick={() => setShowFilters((v) => !v)}
            className={showFilters ? "text-[var(--accent)]" : undefined}
          >
            <SlidersHorizontal size={18} />
          </IconButton>
        }
      />

      <Segmented
        className="mb-3"
        value={tab}
        onChange={setTab}
        options={[
          { value: "today", label: "Oggi", badge: counts.today },
          { value: "upcoming", label: "Prossime", badge: counts.upcoming },
          { value: "late", label: "In ritardo", badge: counts.late },
          { value: "done", label: "Fatte" },
        ]}
      />

      {showFilters && (
        <div className="animate-rise mb-4 space-y-2.5 rounded-[var(--radius-card)] bg-[var(--surface-2)] p-3">
          <div>
            <p className="mb-1.5 text-[12px] font-medium text-[var(--muted)]">Assegnata a</p>
            <div className="flex flex-wrap gap-1.5">
              <Chip active={assignee === "all"} onClick={() => setAssignee("all")}>
                Chiunque
              </Chip>
              {members.map((m) => (
                <Chip key={m.id} active={assignee === m.id} onClick={() => setAssignee(m.id)}>
                  {m.id === user?.id ? "Io" : m.display_name}
                </Chip>
              ))}
            </div>
          </div>
          <div>
            <p className="mb-1.5 text-[12px] font-medium text-[var(--muted)]">Priorità</p>
            <div className="flex flex-wrap gap-1.5">
              <Chip active={priority === "all"} onClick={() => setPriority("all")}>
                Tutte
              </Chip>
              {(["urgent", "high", "normal", "low"] as Priority[]).map((p) => (
                <Chip key={p} active={priority === p} onClick={() => setPriority(p)}>
                  {PRIORITY_LABEL[p]}
                </Chip>
              ))}
            </div>
          </div>
        </div>
      )}

      {empty ? (
        <Card>
          <EmptyState
            icon={<CheckCircle2 size={30} strokeWidth={1.5} />}
            title={EMPTY_TITLES[tab]}
            description={EMPTY_HINTS[tab]}
          />
        </Card>
      ) : (
        groups
          .filter((g) => g.tasks.length > 0)
          .map((g) => (
            <section key={g.title} className="mb-5">
              <SectionHeader title={g.title} right={String(g.tasks.length)} />
              <RowList>
                {g.tasks.map((t) => (
                  <TaskRow key={t.id} task={t} showDate={g.showDate} />
                ))}
              </RowList>
            </section>
          ))
      )}
    </>
  );
}

const EMPTY_TITLES: Record<Tab, string> = {
  today: "Giornata libera",
  upcoming: "Niente in arrivo",
  late: "Nessun ritardo",
  done: "Ancora niente di completato",
};

const EMPTY_HINTS: Record<Tab, string> = {
  today: "Usa il + o il Quick Capture in Home per aggiungere qualcosa.",
  upcoming: "Le attività dei prossimi 30 giorni compariranno qui.",
  late: "Tutto sotto controllo.",
  done: "Le attività completate restano qui per un po'.",
};

interface Group {
  title: string;
  tasks: Task[];
  showDate?: boolean;
}

function buildGroups(tab: Tab, tasks: Task[], today: string): Group[] {
  if (tab === "today") {
    return [
      { title: "In ritardo", tasks: tasksOverdue(tasks, today), showDate: true },
      { title: "Oggi", tasks: tasksToday(tasks, today) },
      { title: "Senza data", tasks: tasksSomeday(tasks) },
    ];
  }

  if (tab === "upcoming") {
    const upcoming = tasksUpcoming(tasks, 30, today);
    const byDate = new Map<string, Task[]>();
    for (const t of upcoming) {
      const key = t.due_date!;
      byDate.set(key, [...(byDate.get(key) ?? []), t]);
    }
    return [...byDate.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, rows]) => ({ title: capitalize(relativeDay(date)), tasks: rows }));
  }

  if (tab === "late") {
    return [{ title: "In ritardo", tasks: tasksOverdue(tasks, today), showDate: true }];
  }

  return [
    {
      title: "Completate",
      tasks: tasks
        .filter((t) => !isOpen(t))
        .sort((a, b) => (b.completed_at ?? "").localeCompare(a.completed_at ?? ""))
        .slice(0, 50)
        .sort(byTaskOrder),
      showDate: true,
    },
  ];
}

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

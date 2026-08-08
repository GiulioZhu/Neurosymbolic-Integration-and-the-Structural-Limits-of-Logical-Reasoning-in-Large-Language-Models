"use client";

import { useDeferredValue, useMemo, useState } from "react";
import { Search as SearchIcon } from "lucide-react";
import { AppShell, PageHeader } from "@/components/shell/AppShell";
import { EventRow, ExpenseRow, NoteRow, RowList, TaskRow } from "@/components/items/Rows";
import { Card, EmptyState, SectionHeader } from "@/components/ui/Bits";
import { Input } from "@/components/ui/Field";
import { useComposer } from "@/components/forms/Composer";
import { useStore } from "@/lib/data/store";
import { search, type SearchHit } from "@/lib/data/selectors";
import { KIND_LABEL, type SourceKind } from "@/lib/types";

export default function SearchPage() {
  return (
    <AppShell>
      <SearchScreen />
    </AppShell>
  );
}

function SearchScreen() {
  const { data } = useStore();
  const { open } = useComposer();
  const [query, setQuery] = useState("");
  const deferred = useDeferredValue(query);

  const hits = useMemo(() => search(data, deferred), [data, deferred]);

  const grouped = useMemo(() => {
    const map: Record<SourceKind, SearchHit[]> = { task: [], event: [], note: [], expense: [] };
    for (const h of hits) map[h.kind].push(h);
    return map;
  }, [hits]);

  const order: SourceKind[] = ["task", "event", "expense", "note"];
  const hasQuery = deferred.trim().length >= 2;

  return (
    <>
      <PageHeader title="Ricerca" />

      <Input
        autoFocus
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Cerca fra attività, eventi, note e spese"
        aria-label="Cerca"
        className="mb-5"
        type="search"
      />

      {!hasQuery ? (
        <Card>
          <EmptyState
            icon={<SearchIcon size={30} strokeWidth={1.5} />}
            title="Cerca in tutto"
            description="Scrivi almeno due lettere. Prova con “assicurazione”, “cena” o “affitto”."
          />
        </Card>
      ) : hits.length === 0 ? (
        <Card>
          <EmptyState
            icon={<SearchIcon size={30} strokeWidth={1.5} />}
            title={`Nessun risultato per “${deferred.trim()}”`}
            description="Prova con una parola più corta."
          />
        </Card>
      ) : (
        order
          .filter((k) => grouped[k].length > 0)
          .map((k) => (
            <section key={k} className="mb-5">
              <SectionHeader title={KIND_LABEL[k]} right={String(grouped[k].length)} />
              <RowList>
                {grouped[k].map((h) => {
                  switch (h.kind) {
                    case "task":
                      return <TaskRow key={h.item.id} task={h.item} showDate />;
                    case "event":
                      return (
                        <EventRow
                          key={h.item.id}
                          event={h.item}
                          date={h.item.start_date}
                          showDate
                        />
                      );
                    case "expense":
                      return <ExpenseRow key={h.item.id} expense={h.item} />;
                    default:
                      return (
                        <NoteRow
                          key={h.item.id}
                          note={h.item}
                          onAct={() => open({ kind: "note", note: h.item })}
                        />
                      );
                  }
                })}
              </RowList>
            </section>
          ))
      )}
    </>
  );
}

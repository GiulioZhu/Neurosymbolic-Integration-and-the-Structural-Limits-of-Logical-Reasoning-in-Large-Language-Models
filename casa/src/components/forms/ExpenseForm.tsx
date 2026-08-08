"use client";

import { useEffect, useRef, useState } from "react";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Chip } from "@/components/ui/Segmented";
import { Field, Input, Textarea } from "@/components/ui/Field";
import { useStore } from "@/lib/data/store";
import { todayISO } from "@/lib/date";
import { parseAmountToCents } from "@/lib/format";
import { PAYMENT_LABEL, type Expense, type PaymentMethod, type UUID, type Visibility } from "@/lib/types";
import { CategoryPicker, MemberPicker, VisibilityPicker } from "./pickers";

const LAST_USED_KEY = "casa:expense:last";

interface LastUsed {
  category_id?: UUID | null;
  payment_method?: PaymentMethod;
}

function readLastUsed(): LastUsed {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(window.localStorage.getItem(LAST_USED_KEY) ?? "{}") as LastUsed;
  } catch {
    return {};
  }
}

export interface ExpenseFormProps {
  expense?: Expense;
  initial?: Partial<Expense>;
  onDone(): void;
}

/**
 * Il form più usato dell'app: si apre con il cursore sull'importo e con
 * categoria e metodo già impostati sull'ultimo valore usato. Due tap e un
 * numero.
 */
export function ExpenseForm({ expense, initial, onDone }: ExpenseFormProps) {
  const { createExpense, updateExpense, deleteExpense, user, pushToast } = useStore();
  const src = expense ?? initial ?? {};
  // letto una volta all'apertura del form: categoria e metodo partono
  // dall'ultimo valore usato, che è quasi sempre quello giusto
  const [last] = useState<LastUsed>(readLastUsed);

  const [amount, setAmount] = useState(
    src.amount_cents !== undefined ? (src.amount_cents / 100).toFixed(2).replace(".", ",") : "",
  );
  const [description, setDescription] = useState(src.description ?? "");
  const [categoryId, setCategoryId] = useState<UUID | null>(
    src.category_id ?? last.category_id ?? null,
  );
  const [spentOn, setSpentOn] = useState(src.spent_on ?? todayISO());
  const [method, setMethod] = useState<PaymentMethod>(
    src.payment_method ?? last.payment_method ?? "card",
  );
  const [paidBy, setPaidBy] = useState<UUID | null>(src.paid_by ?? user?.id ?? null);
  const [note, setNote] = useState(src.note ?? "");
  const [visibility, setVisibility] = useState<Visibility>(src.visibility ?? "shared");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const amountRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const t = window.setTimeout(() => amountRef.current?.focus(), 80);
    return () => window.clearTimeout(t);
  }, []);

  const cents = parseAmountToCents(amount);

  const save = async () => {
    if (cents === null || cents <= 0) {
      setError("Inserisci un importo.");
      amountRef.current?.focus();
      return;
    }
    setSaving(true);
    const payload = {
      amount_cents: cents,
      description: description.trim(),
      category_id: categoryId,
      spent_on: spentOn,
      payment_method: method,
      paid_by: paidBy ?? user?.id ?? "",
      note: note.trim() || null,
      visibility,
    };
    try {
      if (expense) {
        await updateExpense(expense.id, payload);
        pushToast("Spesa aggiornata");
      } else {
        await createExpense(payload);
        pushToast("Spesa registrata");
      }
      window.localStorage.setItem(
        LAST_USED_KEY,
        JSON.stringify({ category_id: categoryId, payment_method: method }),
      );
      onDone();
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!expense) return;
    await deleteExpense(expense.id);
    pushToast("Spesa eliminata");
    onDone();
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void save();
      }}
      className="pb-4"
    >
      <div className="mb-4 flex items-baseline justify-center gap-1 pt-2">
        <input
          ref={amountRef}
          data-autofocus
          value={amount}
          onChange={(e) => {
            setAmount(e.target.value);
            setError(null);
          }}
          inputMode="decimal"
          placeholder="0,00"
          aria-label="Importo"
          className="tnum w-[190px] border-none bg-transparent text-center text-[42px] leading-none font-semibold tracking-tight text-[var(--text)] placeholder:text-[var(--surface-3)] outline-none"
        />
        <span className="text-[26px] font-medium text-[var(--faint)]">€</span>
      </div>
      {error && <p className="mb-3 text-center text-[13px] text-[var(--danger)]">{error}</p>}

      <Field>
        <Input
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Descrizione (es. Esselunga)"
          aria-label="Descrizione"
        />
      </Field>

      <Field label="Categoria">
        <CategoryPicker value={categoryId} onChange={setCategoryId} allowNone={false} />
      </Field>

      <Field label="Quando">
        <Input
          type="date"
          value={spentOn}
          onChange={(e) => setSpentOn(e.target.value || todayISO())}
          aria-label="Data"
        />
      </Field>

      <Field label="Metodo">
        <div className="flex flex-wrap gap-1.5">
          {(Object.keys(PAYMENT_LABEL) as PaymentMethod[]).map((m) => (
            <Chip key={m} active={method === m} onClick={() => setMethod(m)}>
              {PAYMENT_LABEL[m]}
            </Chip>
          ))}
        </div>
      </Field>

      <Field label="Pagato da">
        <MemberPicker value={paidBy} onChange={setPaidBy} allowNone={false} />
      </Field>

      <Field label="Privacy">
        <VisibilityPicker value={visibility} onChange={setVisibility} />
      </Field>

      <Field label="Nota">
        <Textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2} />
      </Field>

      <div className="flex items-center gap-2 pt-1">
        <Button type="submit" variant="primary" full loading={saving}>
          {expense ? "Salva" : "Registra"}
        </Button>
        {expense && (
          <Button type="button" variant="danger" onClick={remove} aria-label="Elimina">
            <Trash2 size={17} />
          </Button>
        )}
      </div>
    </form>
  );
}

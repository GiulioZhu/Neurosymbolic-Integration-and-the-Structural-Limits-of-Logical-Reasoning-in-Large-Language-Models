"use client";

import { useEffect, useRef } from "react";
import { useStore } from "@/lib/data/store";
import { CHECK_INTERVAL_MS, dueNotifications, showNotification } from "@/lib/notifications";

/**
 * Finché una scheda è aperta questo componente consegna i promemoria scaduti.
 * Non ha UI. È il livello che funziona sempre, anche senza chiavi VAPID: le
 * push servono per quando l'app è chiusa.
 */
export function ReminderScheduler() {
  const { data, markNotificationSent, pushToast } = useStore();
  const delivered = useRef(new Set<string>());

  useEffect(() => {
    let stopped = false;

    const tick = () => {
      if (stopped || document.visibilityState === "hidden") return;
      for (const row of dueNotifications(data.notifications)) {
        if (delivered.current.has(row.id)) continue;
        delivered.current.add(row.id);
        void showNotification(row).then((shown) => {
          // se il permesso non c'è, almeno un toast lo si vede
          if (!shown) pushToast(`Promemoria: ${row.title}`);
        });
        void markNotificationSent(row.id).catch(() => {
          delivered.current.delete(row.id);
        });
      }
    };

    tick();
    const id = window.setInterval(tick, CHECK_INTERVAL_MS);
    document.addEventListener("visibilitychange", tick);
    return () => {
      stopped = true;
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [data.notifications, markNotificationSent, pushToast]);

  return null;
}

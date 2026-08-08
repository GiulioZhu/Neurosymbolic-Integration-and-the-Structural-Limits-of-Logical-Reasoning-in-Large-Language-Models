/**
 * Promemoria lato client.
 *
 * Due livelli, indipendenti:
 *  1. **scheduler in-app** — finché una scheda è aperta, controlla la coda
 *     `notifications` ogni 30 s e mostra gli avvisi scaduti. Funziona sempre,
 *     senza configurare nulla.
 *  2. **Web Push** — se sono configurate le chiavi VAPID, il browser si
 *     registra e `/api/push/run` (chiamato da un cron) recapita gli avvisi
 *     anche ad app chiusa.
 */

import { useSyncExternalStore } from "react";
import type { NotificationRow } from "./types";

export type PermissionState = "unsupported" | "default" | "granted" | "denied";

export function notificationSupport(): PermissionState {
  if (typeof window === "undefined" || !("Notification" in window)) return "unsupported";
  return Notification.permission as PermissionState;
}

const permissionListeners = new Set<() => void>();

export async function requestPermission(): Promise<PermissionState> {
  if (notificationSupport() === "unsupported") return "unsupported";
  const result = await Notification.requestPermission();
  for (const cb of permissionListeners) cb();
  return result as PermissionState;
}

/**
 * Il permesso notifiche è uno stato del browser, non di React: si legge con
 * useSyncExternalStore, che rende corretta anche la prima idratazione.
 */
export function useNotificationPermission(): PermissionState {
  return useSyncExternalStore(
    (cb) => {
      permissionListeners.add(cb);
      return () => permissionListeners.delete(cb);
    },
    notificationSupport,
    () => "default" as PermissionState,
  );
}

/** Mostra l'avviso, preferendo il service worker (sopravvive al cambio pagina). */
export async function showNotification(row: NotificationRow): Promise<boolean> {
  if (notificationSupport() !== "granted") return false;
  const options: NotificationOptions = {
    body: row.body ?? undefined,
    tag: row.id,
    icon: "/icons/icon-192.png",
    badge: "/icons/badge.png",
    data: { sourceKind: row.source_kind, sourceId: row.source_id },
  };

  try {
    const reg = await navigator.serviceWorker?.ready;
    if (reg) {
      await reg.showNotification(row.title, options);
      return true;
    }
  } catch {
    /* si ricade sulla notifica diretta */
  }

  try {
    new Notification(row.title, options);
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
//  Web Push
// ---------------------------------------------------------------------------

export const vapidPublicKey = process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY ?? "";
export const pushConfigured = Boolean(vapidPublicKey);

function urlBase64ToUint8Array(base64: string): Uint8Array {
  const padding = "=".repeat((4 - (base64.length % 4)) % 4);
  const normalized = (base64 + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(normalized);
  return Uint8Array.from([...raw].map((c) => c.charCodeAt(0)));
}

export interface PushKeys {
  endpoint: string;
  p256dh: string;
  auth: string;
}

/** Registra il browser per le push. `null` se non è possibile. */
export async function subscribeToPush(): Promise<PushKeys | null> {
  if (!pushConfigured) return null;
  if (typeof navigator === "undefined" || !("serviceWorker" in navigator)) return null;
  if (!("PushManager" in window)) return null;
  if (notificationSupport() !== "granted") return null;

  const reg = await navigator.serviceWorker.ready;
  const existing = await reg.pushManager.getSubscription();
  const sub =
    existing ??
    (await reg.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(vapidPublicKey) as BufferSource,
    }));

  const json = sub.toJSON() as { endpoint?: string; keys?: { p256dh?: string; auth?: string } };
  if (!json.endpoint || !json.keys?.p256dh || !json.keys?.auth) return null;
  return { endpoint: json.endpoint, p256dh: json.keys.p256dh, auth: json.keys.auth };
}

// ---------------------------------------------------------------------------
//  Scheduler in-app
// ---------------------------------------------------------------------------

export const CHECK_INTERVAL_MS = 30_000;

/** I promemoria da mostrare adesso (scaduti, mai inviati, non troppo vecchi). */
export function dueNotifications(
  rows: NotificationRow[],
  now = new Date(),
  maxAgeMinutes = 120,
): NotificationRow[] {
  const t = now.getTime();
  const floor = t - maxAgeMinutes * 60_000;
  return rows.filter((n) => {
    if (n.sent_at || n.read_at) return false;
    const at = new Date(n.scheduled_for).getTime();
    return at <= t && at >= floor;
  });
}

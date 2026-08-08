"use client";

import { createContext, useCallback, useContext, useEffect, useSyncExternalStore } from "react";

export type ThemePreference = "system" | "light" | "dark";

const KEY = "casa:theme";

interface ThemeValue {
  preference: ThemePreference;
  resolved: "light" | "dark";
  setPreference(p: ThemePreference): void;
  toggle(): void;
}

const ThemeContext = createContext<ThemeValue | null>(null);

export function useTheme(): ThemeValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme va usato dentro <ThemeProvider>");
  return ctx;
}

// ---------------------------------------------------------------------------
//  La preferenza vive in localStorage, cioè fuori da React: la si legge con
//  useSyncExternalStore, che gestisce correttamente anche l'idratazione
//  (sul server risponde "system", poi si riallinea senza warning).
// ---------------------------------------------------------------------------

const listeners = new Set<() => void>();

function subscribePreference(cb: () => void): () => void {
  listeners.add(cb);
  // un'altra scheda che cambia tema aggiorna anche questa
  window.addEventListener("storage", cb);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", cb);
  };
}

function readPreference(): ThemePreference {
  const stored = window.localStorage.getItem(KEY);
  return stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
}

function writePreference(p: ThemePreference) {
  window.localStorage.setItem(KEY, p);
  for (const cb of listeners) cb();
}

const serverPreference = (): ThemePreference => "system";

// --- media query -----------------------------------------------------------

function subscribeSystem(cb: () => void): () => void {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", cb);
  return () => mq.removeEventListener("change", cb);
}

const readSystemDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
const serverSystemDark = () => false;

// ---------------------------------------------------------------------------

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const preference = useSyncExternalStore(subscribePreference, readPreference, serverPreference);
  const systemDark = useSyncExternalStore(subscribeSystem, readSystemDark, serverSystemDark);

  const resolved: "light" | "dark" =
    preference === "dark" || (preference === "system" && systemDark) ? "dark" : "light";

  // unico effetto: sincronizzare il DOM, che è un sistema esterno
  useEffect(() => {
    document.documentElement.dataset.theme = resolved;
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", resolved === "dark" ? "#131211" : "#FAF8F5");
  }, [resolved]);

  const setPreference = useCallback((p: ThemePreference) => writePreference(p), []);
  const toggle = useCallback(
    () => writePreference(resolved === "dark" ? "light" : "dark"),
    [resolved],
  );

  return (
    <ThemeContext.Provider value={{ preference, resolved, setPreference, toggle }}>
      {children}
    </ThemeContext.Provider>
  );
}

/**
 * Applica il tema prima del primo paint, così non si vede il lampo bianco
 * aprendo l'app di sera. Viene iniettato in <head>.
 */
export const themeBootScript = `
(function(){
  try {
    var p = localStorage.getItem('${KEY}') || 'system';
    var dark = p === 'dark' || (p === 'system' && matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  } catch (e) {}
})();
`;

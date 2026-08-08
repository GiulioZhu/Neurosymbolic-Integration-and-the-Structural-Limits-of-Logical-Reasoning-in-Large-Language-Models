/**
 * Genera le icone PNG della PWA da un disegno vettoriale, senza dipendenze:
 * usa il Chromium già presente per rasterizzare un SVG.
 *
 *   node scripts/generate-icons.mjs
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, "../public/icons");

const BG = "#C25A38";
const FG = "#FFF8F4";

/** Il tetto stilizzato di "Casa": una forma sola, riconoscibile a 16px. */
function svg({ size, padding, background }) {
  const s = size;
  const inner = s - padding * 2;
  const x = padding;
  const y = padding;
  const roof = inner * 0.42;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${s}" height="${s}" viewBox="0 0 ${s} ${s}">
  <rect width="${s}" height="${s}" rx="${background ? s * 0.22 : 0}" fill="${BG}"/>
  <path d="
    M ${x + inner / 2} ${y}
    L ${x + inner} ${y + roof}
    L ${x + inner * 0.86} ${y + roof}
    L ${x + inner * 0.86} ${y + inner}
    L ${x + inner * 0.14} ${y + inner}
    L ${x + inner * 0.14} ${y + roof}
    L ${x} ${y + roof}
    Z"
    fill="${FG}"/>
  <circle cx="${x + inner / 2}" cy="${y + inner * 0.72}" r="${inner * 0.1}" fill="${BG}"/>
</svg>`;
}

const TARGETS = [
  { file: "icon-192.png", size: 192, padding: 40, background: true },
  { file: "icon-512.png", size: 512, padding: 106, background: true },
  { file: "maskable-512.png", size: 512, padding: 150, background: true },
  { file: "badge.png", size: 96, padding: 20, background: false },
];

async function main() {
  mkdirSync(outDir, { recursive: true });

  const { chromium } = await import("playwright").catch(() => ({ chromium: null }));
  if (!chromium) {
    // fallback: salva gli SVG, che il manifest può comunque usare
    for (const t of TARGETS) {
      writeFileSync(resolve(outDir, t.file.replace(".png", ".svg")), svg(t));
    }
    console.log("Playwright non disponibile: ho scritto gli SVG in public/icons.");
    return;
  }

  // in ambienti dove il browser è preinstallato altrove, si passa il percorso
  const executablePath = process.env.CHROMIUM_PATH || undefined;
  const browser = await chromium.launch(executablePath ? { executablePath } : {});
  const page = await browser.newPage();

  for (const t of TARGETS) {
    const markup = svg(t);
    await page.setViewportSize({ width: t.size, height: t.size });
    await page.setContent(
      `<html><body style="margin:0;background:transparent">${markup}</body></html>`,
    );
    const buffer = await page.screenshot({ omitBackground: true });
    writeFileSync(resolve(outDir, t.file), buffer);
    console.log(`✓ ${t.file} (${t.size}×${t.size})`);
  }

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

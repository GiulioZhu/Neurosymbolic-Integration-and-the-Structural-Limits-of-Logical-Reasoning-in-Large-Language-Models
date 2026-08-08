import type { Metadata, Viewport } from "next";
import { StoreProvider } from "@/lib/data/store";
import { ThemeProvider, themeBootScript } from "@/lib/theme";
import { ComposerProvider } from "@/components/forms/Composer";
import { ServiceWorker } from "@/components/shell/ServiceWorker";
import "./globals.css";

export const metadata: Metadata = {
  title: "Casa",
  description: "Lo spazio condiviso per la vita quotidiana di una coppia.",
  manifest: "/manifest.webmanifest",
  applicationName: "Casa",
  appleWebApp: { capable: true, statusBarStyle: "default", title: "Casa" },
  icons: {
    icon: [{ url: "/icons/icon-192.png", sizes: "192x192", type: "image/png" }],
    apple: [{ url: "/icons/icon-192.png" }],
  },
};

export const viewport: Viewport = {
  themeColor: "#FAF8F5",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="it" suppressHydrationWarning>
      <head>
        {/* evita il lampo bianco all'apertura con il tema scuro */}
        <script dangerouslySetInnerHTML={{ __html: themeBootScript }} />
      </head>
      <body>
        <ThemeProvider>
          <StoreProvider>
            <ComposerProvider>{children}</ComposerProvider>
          </StoreProvider>
        </ThemeProvider>
        <ServiceWorker />
      </body>
    </html>
  );
}

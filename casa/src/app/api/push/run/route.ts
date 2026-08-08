import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import webpush from "web-push";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Consegna i promemoria scaduti come notifiche push.
 *
 * Va chiamato da uno scheduler (Vercel Cron, pg_cron, GitHub Actions…) ogni
 * minuto o ogni cinque, con l'header `Authorization: Bearer $CRON_SECRET`.
 *
 * Legge la coda `notifications` con la service role key — quindi ignora RLS —
 * e manda ciascun avviso solo ai dispositivi dello stesso household. Un avviso
 * su una riga privata va soltanto a chi l'ha creata.
 */
export async function POST(request: Request) {
  const secret = process.env.CRON_SECRET;
  const auth = request.headers.get("authorization");
  if (!secret || auth !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "Non autorizzato" }, { status: 401 });
  }

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const publicKey = process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY;
  const privateKey = process.env.VAPID_PRIVATE_KEY;

  if (!url || !serviceKey) {
    return NextResponse.json({ error: "Supabase non configurato" }, { status: 503 });
  }
  if (!publicKey || !privateKey) {
    return NextResponse.json({ error: "Chiavi VAPID mancanti" }, { status: 503 });
  }

  webpush.setVapidDetails(process.env.VAPID_SUBJECT ?? "mailto:casa@example.com", publicKey, privateKey);
  const sb = createClient(url, serviceKey, { auth: { persistSession: false } });

  const { data: due, error } = await sb
    .from("notifications")
    .select("*")
    .is("sent_at", null)
    .lte("scheduled_for", new Date().toISOString())
    .order("scheduled_for")
    .limit(200);

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  if (!due || due.length === 0) return NextResponse.json({ sent: 0, pending: 0 });

  const householdIds = [...new Set(due.map((n) => n.household_id as string))];
  const { data: subs } = await sb
    .from("push_subscriptions")
    .select("*")
    .in("household_id", householdIds);

  let sent = 0;
  const staleEndpoints: string[] = [];
  const deliveredIds: string[] = [];

  for (const note of due) {
    const targets = (subs ?? []).filter((s) => {
      if (s.household_id !== note.household_id) return false;
      // riga privata → solo il suo autore
      if (note.visibility === "private" && s.user_id !== note.created_by) return false;
      // notifica indirizzata a una persona precisa
      if (note.user_id && s.user_id !== note.user_id) return false;
      return true;
    });

    const payload = JSON.stringify({
      title: note.title,
      body: note.body ?? "",
      tag: note.id,
      sourceKind: note.source_kind,
      sourceId: note.source_id,
    });

    for (const target of targets) {
      try {
        await webpush.sendNotification(
          {
            endpoint: target.endpoint as string,
            keys: { p256dh: target.p256dh as string, auth: target.auth as string },
          },
          payload,
        );
        sent++;
      } catch (err) {
        const status = (err as { statusCode?: number }).statusCode;
        // 404/410: il browser ha revocato l'iscrizione, va rimossa
        if (status === 404 || status === 410) staleEndpoints.push(target.endpoint as string);
      }
    }

    deliveredIds.push(note.id as string);
  }

  if (deliveredIds.length > 0) {
    await sb
      .from("notifications")
      .update({ sent_at: new Date().toISOString() })
      .in("id", deliveredIds);
  }
  if (staleEndpoints.length > 0) {
    await sb.from("push_subscriptions").delete().in("endpoint", staleEndpoints);
  }

  return NextResponse.json({
    sent,
    processed: deliveredIds.length,
    removedSubscriptions: staleEndpoints.length,
  });
}

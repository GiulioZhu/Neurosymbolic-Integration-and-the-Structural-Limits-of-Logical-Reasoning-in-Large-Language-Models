/**
 * Genera la coppia di chiavi per le notifiche push.
 *
 *   npm run vapid
 *
 * Copia l'output in `.env.local`.
 */

import webpush from "web-push";

const { publicKey, privateKey } = webpush.generateVAPIDKeys();

console.log(`
Chiavi VAPID generate — incollale in .env.local:

NEXT_PUBLIC_VAPID_PUBLIC_KEY=${publicKey}
VAPID_PRIVATE_KEY=${privateKey}
VAPID_SUBJECT=mailto:tu@esempio.it

La chiave privata non va mai esposta al browser né committata.
`);

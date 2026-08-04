const CACHE_NAME = "dragonnest-shell-v2";
const SHELL_ASSETS = [
  "/",
  "/admin",
  "/manifest.webmanifest",
  "/assets/icons/icon.svg",
  "/assets/shared/tokens.css",
  "/assets/shared/base.css",
  "/assets/shared/api.js",
  "/assets/shared/pwa.js",
  "/assets/app/user.css",
  "/assets/app/app.js",
  "/assets/admin/admin.css",
  "/assets/admin/app.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);

  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  if (url.pathname.startsWith("/api/")) {
    return; // live fleet/task data must always go to the network, never cache
  }

  event.respondWith(networkFirst(request));
});

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      await cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;
    if (request.mode === "navigate") {
      return new Response(
        `<!doctype html><html><head><meta charset="utf-8"><title>DragonNest — Offline</title>
        <style>body{font-family:system-ui,sans-serif;background:#f4f6f5;color:#17211b;display:grid;place-items:center;height:100vh;margin:0;text-align:center;padding:24px}</style>
        </head><body><div><h1>You're offline</h1><p>Reconnect to reach the DragonNest Brain.</p></div></body></html>`,
        { headers: { "Content-Type": "text/html" } }
      );
    }
    return new Response("", { status: 504 });
  }
}

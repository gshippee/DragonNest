function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => { /* offline support is best-effort */ });
  });
}

function setupInstallPrompt(buttonId) {
  const button = document.getElementById(buttonId);
  if (!button) return;
  let deferredPrompt = null;
  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    deferredPrompt = event;
    button.hidden = false;
  });
  button.addEventListener("click", async () => {
    if (!deferredPrompt) return;
    button.hidden = true;
    await deferredPrompt.prompt();
    deferredPrompt = null;
  });
  window.addEventListener("appinstalled", () => { button.hidden = true; });
}

function watchOnlineStatus(bannerId) {
  const banner = document.getElementById(bannerId);
  if (!banner) return;
  const update = () => { banner.hidden = navigator.onLine; };
  window.addEventListener("online", update);
  window.addEventListener("offline", update);
  update();
}

registerServiceWorker();

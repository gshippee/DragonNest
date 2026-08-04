const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
const statusClass = (value) => String(value || "neutral").toLowerCase();
const fmt = (value, suffix = "") => (value === -1 || value == null ? "Unknown" : `${value}${suffix}`);
const decimal = (value, digits = 2) => (value === -1 || value == null ? "Unknown" : Number(value).toFixed(digits));

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.json();
}

function toast(message) {
  const node = document.getElementById("toast");
  if (!node) return;
  node.textContent = message;
  node.classList.add("visible");
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => node.classList.remove("visible"), 3200);
}

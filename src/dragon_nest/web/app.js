const state = { devices: [], tasks: [], vectors: [], selectedTask: null };

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const statusClass = (value) => String(value || "neutral").toLowerCase();
const fmt = (value, suffix = "") => value === -1 || value == null ? "Unknown" : `${value}${suffix}`;
const decimal = (value, digits = 2) => value === -1 || value == null ? "Unknown" : Number(value).toFixed(digits);

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  return response.json();
}

async function refresh() {
  try {
    const [health, devices, tasks, events] = await Promise.all([
      api("/api/health"), api("/api/devices"), api("/api/tasks"), api("/api/events?limit=120")
    ]);
    state.devices = devices;
    state.tasks = tasks;
    const origin = $("origin-device");
    const selectedOrigin = origin.value;
    origin.innerHTML = '<option value="">None</option>' + devices.map((device) => `<option value="${esc(device.device_id)}" ${device.device_id === selectedOrigin ? "selected" : ""}>${esc(device.display_name)} (${esc(device.device_id)})</option>`).join("");
    $("brain-dot").classList.add("online");
    $("brain-state").textContent = `${health.brain_id} online`;
    renderDevices(); renderTaskSelect(); renderEvents(events);
    if (state.selectedTask) {
      const updated = tasks.find((task) => task.task_id === state.selectedTask.task_id);
      if (updated) { state.selectedTask = updated; renderTask(updated); }
    }
  } catch (error) {
    $("brain-dot").classList.remove("online");
    $("brain-state").textContent = "Brain unavailable";
  }
  if (window.lucide) lucide.createIcons();
}

function renderDevices() {
  $("device-count").textContent = `${state.devices.length} device${state.devices.length === 1 ? "" : "s"}`;
  $("devices").innerHTML = state.devices.length ? state.devices.map((device) => {
    const h = device.health;
    const models = device.models.map((model) => `<span class="chip">${esc(model.role)} / ${esc(model.model_id)}</span>`).join("");
    return `<article class="device-card ${statusClass(device.status)}">
      <div class="device-title"><div><h3>${esc(device.display_name)}</h3><p>${esc(device.device_id)} · ${esc(device.platform)} · ${device.connected ? "stream connected" : "disconnected"}</p></div><div><span class="status ${statusClass(device.status)}">${esc(device.status)}</span> <button class="icon-button simulate" data-device="${esc(device.device_id)}" title="Simulate device state" aria-label="Simulate ${esc(device.display_name)}"><i data-lucide="gauge"></i></button></div></div>
      <div class="metrics"><div class="metric"><span>Battery</span><strong>${h.battery_pct < 0 ? "Unknown" : `${decimal(h.battery_pct, 0)}%${h.charging ? " charging" : ""}`}</strong></div><div class="metric"><span>Thermal</span><strong>${decimal(h.thermal_level)}</strong></div><div class="metric"><span>Memory</span><strong>${h.available_memory_mb === 0 ? "Unknown" : fmt(h.available_memory_mb, " MB")}</strong></div><div class="metric"><span>Accelerator</span><strong>${h.accelerator_utilization < 0 ? "Unknown" : `${decimal(h.accelerator_utilization * 100, 0)}%`}</strong></div><div class="metric"><span>Network RTT</span><strong>${h.network_rtt_ms < 0 ? "Unknown" : `${decimal(h.network_rtt_ms, 0)} ms`}</strong></div><div class="metric"><span>Active</span><strong>${device.active_tasks.length}</strong></div></div>
      <div class="model-list">${models || '<span class="chip">No advertised models</span>'}</div>
    </article>`;
  }).join("") : '<div class="empty">No registered devices</div>';
  document.querySelectorAll(".simulate").forEach((button) => button.addEventListener("click", () => openSimulation(button.dataset.device)));
}

function renderTaskSelect() {
  const select = $("task-select");
  const current = state.selectedTask?.task_id;
  select.innerHTML = state.tasks.length ? state.tasks.map((task) => `<option value="${esc(task.task_id)}" ${task.task_id === current ? "selected" : ""}>${esc(task.task_id)} · ${esc(task.state)}</option>`).join("") : "<option>No tasks</option>";
  if (!state.selectedTask && state.tasks.length) selectTask(state.tasks[0].task_id);
}

function renderTask(task) {
  const profile = task.profile;
  $("profile-strip").innerHTML = profile ? `<span class="profile-item">Class <strong>${esc(profile.task_class)}</strong></span><span class="profile-item">Confidence <strong>${Math.round(profile.confidence * 100)}%</strong></span><span class="profile-item">Mode <strong>${esc(task.execution_mode)}</strong></span><span class="profile-item">Privacy <strong>${esc(profile.privacy_tier)}</strong></span><span class="profile-item">Reducer <strong>${esc(task.reducer)}</strong></span>${task.origin_device_id ? `<span class="profile-item">Origin <strong>${esc(task.origin_device_id)}</strong></span>` : ""}${task.steering?.enabled ? `<span class="profile-item">Steering <strong>${esc(task.steering.vector_id)} @ ${task.steering.alpha}</strong></span>` : ""}` : "";
  $("route-trace").innerHTML = task.route_reasons?.length ? task.route_reasons.map((reason) => `<li>${esc(reason)}</li>`).join("") : '<li class="empty">No route trace available</li>';
  $("progress").innerHTML = task.progress?.length ? task.progress.map((item) => `<tr><td>${esc(item.id)}</td><td>${esc(item.device_id)}${item.winner ? " · winner" : ""}</td><td>${esc(item.model_id)}</td><td><span class="status ${statusClass(item.state)}">${esc(item.state)}</span></td><td>${fmt(item.latency_ms, " ms")}</td><td>${item.retry_count}</td></tr>`).join("") : '<tr><td colspan="6" class="empty">No shards or pipeline stages</td></tr>';
  $("result-state").className = `status ${statusClass(task.state)}`;
  $("result-state").textContent = task.state;
  $("result-output").textContent = task.result?.output_text || task.error_message || "No result available.";
  const metrics = task.result?.metrics;
  $("result-meta").innerHTML = task.result ? `<span>Device: <strong>${esc(task.result.device_id)}</strong></span><span>Latency: <strong>${task.result.latency_ms} ms</strong></span>${metrics ? `<span>Runtime: <strong>${esc(metrics.runtime_name)} ${esc(metrics.runtime_version)}</strong></span><span>Accelerator: <strong>${esc(metrics.accelerator)}</strong></span>` : ""}` : "";
}

function renderEvents(events) {
  $("events").innerHTML = events.length ? events.map((event) => `<div class="event-row"><span>${Number(event.timestamp).toFixed(2)}</span><span class="event-type">${esc(event.type)}</span><span class="event-subject">${esc(event.subject)}</span><span class="event-message">${esc(event.message)}</span></div>`).join("") : '<div class="empty">No events</div>';
}

function selectTask(taskId) {
  const task = state.tasks.find((item) => item.task_id === taskId);
  if (!task) return;
  state.selectedTask = task; renderTask(task);
}

async function loadVectors() {
  state.vectors = await api("/api/steering-vectors");
  $("vector-id").innerHTML = state.vectors.map((vector) => `<option value="${esc(vector.vector_id)}">${esc(vector.vector_id)}</option>`).join("");
  applyVectorDefaults();
}

function applyVectorDefaults() {
  const vector = state.vectors.find((item) => item.vector_id === $("vector-id").value);
  if (!vector) return;
  $("alpha").min = vector.alpha_min; $("alpha").max = vector.alpha_max; $("alpha").value = vector.default_alpha; $("alpha-value").value = vector.default_alpha;
  $("positions").innerHTML = vector.positions.map((position) => `<option value="${esc(position)}" ${position === vector.default_positions ? "selected" : ""}>${esc(position)}</option>`).join("");
}

async function submitTask(event) {
  event.preventDefault();
  const button = $("submit-button"); button.disabled = true;
  const vector = state.vectors.find((item) => item.vector_id === $("vector-id").value);
  try {
    const response = await api("/api/tasks", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({ request_text: $("request").value, preferred_mode: $("preferred-mode").value, execution_mode: $("execution-mode").value, origin_device_id: $("origin-device").value, reducer: $("reducer").value, steering: { enabled: $("steering-enabled").checked, vector_id: vector?.vector_id || "", model_family: vector?.model_family || "", target_layer: vector?.default_layer || 0, alpha: Number($("alpha").value), positions: $("positions").value } }) });
    toast(response.success ? `Task ${response.task_id} succeeded` : `${response.error_code}: ${response.error_message}`);
    await refresh(); selectTask(response.task_id);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

function openSimulation(deviceId) {
  const device = state.devices.find((item) => item.device_id === deviceId); if (!device) return;
  $("simulation-device").value = deviceId; $("sim-thermal").value = device.health.thermal_level; $("sim-load").value = device.health.accelerator_utilization; $("sim-rtt").value = device.health.network_rtt_ms; $("sim-offline").checked = device.status === "OFFLINE"; updateSimulationOutputs(); $("simulation-dialog").showModal();
}

function updateSimulationOutputs() { $("sim-thermal-value").value = $("sim-thermal").value; $("sim-load-value").value = $("sim-load").value; }

async function applySimulation(event) {
  event.preventDefault();
  try { await api(`/api/devices/${encodeURIComponent($("simulation-device").value)}/simulate`, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({ thermal_level: Number($("sim-thermal").value), accelerator_utilization: Number($("sim-load").value), network_rtt_ms: Number($("sim-rtt").value), offline: $("sim-offline").checked }) }); $("simulation-dialog").close(); await refresh(); }
  catch (error) { toast(error.message); }
}

function toast(message) { const node = $("toast"); node.textContent = message; node.classList.add("visible"); clearTimeout(window.toastTimer); window.toastTimer = setTimeout(() => node.classList.remove("visible"), 3200); }

document.addEventListener("DOMContentLoaded", async () => {
  $("task-form").addEventListener("submit", submitTask); $("task-select").addEventListener("change", (event) => selectTask(event.target.value)); $("steering-enabled").addEventListener("change", (event) => $("steering-controls").hidden = !event.target.checked); $("vector-id").addEventListener("change", applyVectorDefaults); $("alpha").addEventListener("input", () => $("alpha-value").value = $("alpha").value); $("refresh-events").addEventListener("click", refresh); $("simulation-form").addEventListener("submit", applySimulation); $("sim-cancel").addEventListener("click", () => $("simulation-dialog").close()); $("sim-thermal").addEventListener("input", updateSimulationOutputs); $("sim-load").addEventListener("input", updateSimulationOutputs);
  await loadVectors(); await refresh(); setInterval(refresh, 1000);
});

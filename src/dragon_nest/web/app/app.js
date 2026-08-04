const STORAGE_KEY = "dragonnest.myDevice";
const state = { myDevice: loadMyDevice(), vectors: [], enrollment: null };

const $ = (id) => document.getElementById(id);

function loadMyDevice() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); }
  catch { return null; }
}

function saveMyDevice(value) {
  state.myDevice = value;
  if (value) localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
  else localStorage.removeItem(STORAGE_KEY);
  updateVisibility();
}

function updateVisibility() {
  const associated = Boolean(state.myDevice);
  $("enrollment-card").hidden = associated;
  $("device-card").hidden = !associated;
  $("profile-card").hidden = !associated;
  $("submit-card").hidden = !associated;
}

async function refresh() {
  try {
    const health = await api("/api/health");
    $("brain-dot").classList.add("online");
    $("brain-state").textContent = `${health.brain_id} online`;
  } catch (error) {
    $("brain-dot").classList.remove("online");
    $("brain-state").textContent = "Brain unavailable";
  }
  if (state.myDevice) await refreshMyDevice();
  if (window.lucide) lucide.createIcons();
}

async function refreshMyDevice() {
  try {
    const devices = await api("/api/devices");
    const device = devices.find((item) => item.device_id === state.myDevice.device_id);
    if (!device) return;
    renderDevice(device);
  } catch (error) { /* device data unavailable while offline; keep last render */ }
}

function renderDevice(device) {
  const h = device.health;
  $("device-subtitle").textContent = `${device.display_name} · ${device.platform} · ${device.connected ? "connected" : "disconnected"}`;
  $("device-status").className = `status-pill ${statusClass(device.status)}`;
  $("device-status").textContent = device.status;
  $("device-metrics").innerHTML = `
    <div class="metric"><span>Battery</span><strong>${h.battery_pct < 0 ? "Unknown" : `${decimal(h.battery_pct, 0)}%${h.charging ? " charging" : ""}`}</strong></div>
    <div class="metric"><span>Thermal</span><strong>${decimal(h.thermal_level)}</strong></div>
    <div class="metric"><span>Memory</span><strong>${h.available_memory_mb === 0 ? "Unknown" : fmt(h.available_memory_mb, " MB")}</strong></div>
    <div class="metric"><span>Network RTT</span><strong>${h.network_rtt_ms < 0 ? "Unknown" : `${decimal(h.network_rtt_ms, 0)} ms`}</strong></div>
  `;
}

async function loadVectors() {
  state.vectors = await api("/api/steering-vectors");
  $("profile-vector").innerHTML = '<option value="">None</option>' + state.vectors.map((vector) => `<option value="${esc(vector.vector_id)}">${esc(vector.vector_id)}</option>`).join("");
}

async function loadProfile() {
  if (!state.myDevice) return;
  try {
    const profiles = await api("/api/personal-profiles");
    const profile = profiles.find((item) => item.profile_id === state.myDevice.profile_id);
    if (!profile) return;
    $("profile-name").value = profile.person_name;
    $("profile-mode").value = profile.preferred_mode;
    $("profile-vector").value = profile.steering_vector_id;
    $("profile-alpha").value = profile.steering_alpha;
    $("profile-alpha-value").value = profile.steering_alpha;
    $("profile-positions").value = profile.steering_positions;
    $("profile-notes").value = profile.notes;
  } catch (error) { toast(error.message); }
}

async function saveProfile(event) {
  event.preventDefault();
  const button = $("profile-save"); button.disabled = true;
  try {
    await api(`/api/personal-profiles/${encodeURIComponent(state.myDevice.profile_id)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        person_name: $("profile-name").value.trim(),
        preferred_mode: $("profile-mode").value,
        steering_vector_id: $("profile-vector").value,
        steering_alpha: Number($("profile-alpha").value),
        steering_positions: $("profile-positions").value,
        notes: $("profile-notes").value.trim(),
      }),
    });
    toast("Profile saved");
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

async function submitTask(event) {
  event.preventDefault();
  const button = $("submit-button"); button.disabled = true;
  $("result-card").hidden = false;
  $("result-state").className = "status-pill running";
  $("result-state").textContent = "Running";
  $("result-output").textContent = "Waiting for response…";
  $("result-meta").innerHTML = "";
  try {
    const response = await api("/api/tasks", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_text: $("request").value,
        preferred_mode: $("preferred-mode").value,
        origin_device_id: state.myDevice.device_id,
        use_profile_steering: $("profile-steering").checked,
      }),
    });
    const task = await api(`/api/tasks/${encodeURIComponent(response.task_id)}`);
    renderResult(task);
  } catch (error) {
    $("result-state").className = "status-pill failed";
    $("result-state").textContent = "Failed";
    $("result-output").textContent = error.message;
    toast(error.message);
  } finally { button.disabled = false; }
}

function renderResult(task) {
  $("result-state").className = `status-pill ${statusClass(task.state)}`;
  $("result-state").textContent = task.state;
  $("result-output").textContent = task.result?.output_text || task.error_message || "No result available.";
  const metrics = task.result?.metrics;
  $("result-meta").innerHTML = task.result ? `<span>Handled by: <strong>${esc(task.result.device_id)}</strong></span><span>Latency: <strong>${task.result.latency_ms} ms</strong></span>${metrics ? `<span>Runtime: <strong>${esc(metrics.runtime_name)} ${esc(metrics.runtime_version)}</strong></span>` : ""}` : "";
}

function forgetDevice() {
  saveMyDevice(null);
  toast("Device forgotten on this browser");
}

function openEnrollment() {
  $("enrollment-host").value = window.location.hostname || "127.0.0.1";
  $("enrollment-port").value = "50051";
  $("enrollment-person").value = "";
  $("enrollment-device-name").value = "";
  resetEnrollment();
  $("enrollment-dialog").showModal();
}

function resetEnrollment() {
  clearInterval(window.__enrollmentTimer);
  state.enrollment = null;
  $("enrollment-settings").hidden = false;
  $("enrollment-code").hidden = true;
  $("enrollment-create").hidden = false;
  $("enrollment-create").disabled = false;
  $("enrollment-status").textContent = "Waiting for scan";
  $("enrollment-expiry").textContent = "";
  $("enrollment-qr").removeAttribute("src");
}

async function createEnrollment(event) {
  event.preventDefault();
  const button = $("enrollment-create"); button.disabled = true;
  try {
    const session = await api("/api/enrollment-sessions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        brain_host: $("enrollment-host").value.trim(),
        brain_port: Number($("enrollment-port").value),
        use_tls: false,
        ttl_seconds: 300,
        person_name: $("enrollment-person").value.trim(),
        device_name: $("enrollment-device-name").value.trim(),
      }),
    });
    state.enrollment = session;
    $("enrollment-settings").hidden = true;
    $("enrollment-code").hidden = false;
    $("enrollment-create").hidden = true;
    $("enrollment-qr").src = `${session.qr_url}?t=${Date.now()}`;
    updateEnrollmentStatus(session);
    window.__enrollmentTimer = setInterval(pollEnrollment, 1000);
  } catch (error) { toast(error.message); button.disabled = false; }
}

async function pollEnrollment() {
  if (!state.enrollment) return;
  try {
    const session = await api(`/api/enrollment-sessions/${encodeURIComponent(state.enrollment.session_id)}`);
    state.enrollment = session; updateEnrollmentStatus(session);
    if (session.status === "CLAIMED") {
      clearInterval(window.__enrollmentTimer);
      saveMyDevice({ device_id: session.claimed_device_id, profile_id: session.profile_id });
      $("enrollment-dialog").close();
      resetEnrollment();
      await loadProfile();
      await refresh();
      toast("Device connected");
    } else if (session.status !== "PENDING") {
      clearInterval(window.__enrollmentTimer);
    }
  } catch (error) { clearInterval(window.__enrollmentTimer); toast(error.message); }
}

function updateEnrollmentStatus(session) {
  const remaining = Math.max(0, Math.ceil(session.expires_at - Date.now() / 1000));
  if (session.status === "PENDING") {
    $("enrollment-status").textContent = "Waiting for scan";
    $("enrollment-expiry").textContent = `Expires in ${remaining}s`;
  } else {
    $("enrollment-status").textContent = session.status;
    $("enrollment-expiry").textContent = "";
  }
}

async function closeEnrollment() {
  clearInterval(window.__enrollmentTimer);
  if (state.enrollment?.status === "PENDING") {
    try { await api(`/api/enrollment-sessions/${encodeURIComponent(state.enrollment.session_id)}`, { method: "DELETE" }); }
    catch (error) { toast(error.message); }
  }
  $("enrollment-dialog").close(); resetEnrollment();
}

document.addEventListener("DOMContentLoaded", async () => {
  updateVisibility();
  $("add-device").addEventListener("click", openEnrollment);
  $("enrollment-form").addEventListener("submit", createEnrollment);
  $("enrollment-close").addEventListener("click", closeEnrollment);
  $("enrollment-cancel").addEventListener("click", closeEnrollment);
  $("forget-device").addEventListener("click", forgetDevice);
  $("profile-form").addEventListener("submit", saveProfile);
  $("profile-alpha").addEventListener("input", () => $("profile-alpha-value").value = $("profile-alpha").value);
  $("task-form").addEventListener("submit", submitTask);
  watchOnlineStatus("offline-banner");
  setupInstallPrompt("install-button");
  await loadVectors();
  await loadProfile();
  await refresh();
  setInterval(refresh, 4000);
});

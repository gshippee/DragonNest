const STORAGE_KEY = "dragonnest.myDevice";
const STYLE_VECTOR = "concise-vs-verbose-layer-7";
const state = { myDevice: loadMyDevice(), enrollment: null };
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
    await api("/api/health");
    $("brain-dot").classList.add("online");
    $("brain-state").textContent = "Ready";
  } catch (error) {
    $("brain-dot").classList.remove("online");
    $("brain-state").textContent = "Unavailable";
  }
  if (state.myDevice) await refreshMyDevice();
  if (window.lucide) lucide.createIcons();
}

async function refreshMyDevice() {
  try {
    const devices = await api("/api/devices");
    const device = devices.find((item) => item.device_id === state.myDevice.device_id);
    if (device) renderDevice(device);
  } catch (error) { /* Keep the last friendly state when offline. */ }
}

function renderDevice(device) {
  const ready = device.connected && ["HEALTHY", "DEGRADED"].includes(device.status);
  $("device-subtitle").textContent = ready
    ? "Your device is available for requests."
    : "Your device is reconnecting. You can try again in a moment.";
  $("device-status").className = `status-pill ${ready ? "healthy" : "neutral"}`;
  $("device-status").textContent = ready ? "Ready" : "Connecting";
}

function profileStyle(profile) {
  if (profile.steering_vector_id !== STYLE_VECTOR) return "balanced";
  return profile.steering_alpha < 0 ? "concise" : "detailed";
}

async function loadProfile() {
  if (!state.myDevice?.profile_id) return;
  try {
    const profiles = await api("/api/personal-profiles");
    const profile = profiles.find((item) => item.profile_id === state.myDevice.profile_id);
    if (!profile) return;
    $("profile-name").value = profile.person_name;
    $("profile-mode").value = profile.preferred_mode;
    $("profile-style").value = profileStyle(profile);
  } catch (error) { toast("Your preferences are temporarily unavailable."); }
}

function steeringForStyle(style) {
  if (style === "concise") return { steering_vector_id: STYLE_VECTOR, steering_alpha: -2, steering_positions: "last" };
  if (style === "detailed") return { steering_vector_id: STYLE_VECTOR, steering_alpha: 2, steering_positions: "last" };
  return { steering_vector_id: "", steering_alpha: 0, steering_positions: "last" };
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
        allow_remote_vector: false,
        notes: "",
        ...steeringForStyle($("profile-style").value),
      }),
    });
    toast("Preferences saved");
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

async function submitTask(event) {
  event.preventDefault();
  const button = $("submit-button"); button.disabled = true;
  $("result-card").hidden = false;
  $("result-state").className = "status-pill running";
  $("result-state").textContent = "Thinking";
  $("result-output").textContent = "Working on it...";
  try {
    const response = await api("/api/tasks", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_text: $("request").value,
        preferred_mode: "auto",
        origin_device_id: state.myDevice.device_id,
        use_profile_steering: true,
      }),
    });
    const task = await api(`/api/tasks/${encodeURIComponent(response.task_id)}`);
    renderResult(task);
  } catch (error) {
    $("result-state").className = "status-pill failed";
    $("result-state").textContent = "Try again";
    $("result-output").textContent = "DragonNest could not complete that request. Please try again.";
  } finally { button.disabled = false; }
}

function renderResult(task) {
  const succeeded = task.state === "SUCCEEDED";
  $("result-state").className = `status-pill ${succeeded ? "healthy" : "failed"}`;
  $("result-state").textContent = succeeded ? "Done" : "Try again";
  $("result-output").textContent = task.result?.output_text
    || "DragonNest could not complete that request. Please try again.";
}

function forgetDevice() {
  saveMyDevice(null);
  toast("You can connect another device now.");
}

function openEnrollment() {
  resetEnrollment();
  $("enrollment-dialog").showModal();
  createEnrollment();
}

function resetEnrollment() {
  clearInterval(window.__enrollmentTimer);
  state.enrollment = null;
  $("enrollment-status").textContent = "Preparing code";
  $("enrollment-expiry").textContent = "";
  $("enrollment-qr").removeAttribute("src");
}

async function createEnrollment() {
  try {
    const session = await api("/api/enrollment-sessions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        brain_host: window.location.hostname || "127.0.0.1",
        brain_port: 50051,
        use_tls: window.location.protocol === "https:",
        ttl_seconds: 300,
      }),
    });
    state.enrollment = session;
    $("enrollment-qr").src = `${session.qr_url}?t=${Date.now()}`;
    updateEnrollmentStatus(session);
    window.__enrollmentTimer = setInterval(pollEnrollment, 1000);
  } catch (error) {
    $("enrollment-status").textContent = "Could not create a code";
    toast(error.message);
  }
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
      toast("Your device is connected.");
    } else if (session.status !== "PENDING") {
      clearInterval(window.__enrollmentTimer);
    }
  } catch (error) { clearInterval(window.__enrollmentTimer); toast(error.message); }
}

function updateEnrollmentStatus(session) {
  const remaining = Math.max(0, Math.ceil(session.expires_at - Date.now() / 1000));
  if (session.status === "PENDING") {
    $("enrollment-status").textContent = "Scan this code with your phone";
    $("enrollment-expiry").textContent = `Expires in ${remaining}s`;
  } else {
    $("enrollment-status").textContent = "Connecting your device";
    $("enrollment-expiry").textContent = "";
  }
}

async function closeEnrollment() {
  clearInterval(window.__enrollmentTimer);
  if (state.enrollment?.status === "PENDING") {
    try { await api(`/api/enrollment-sessions/${encodeURIComponent(state.enrollment.session_id)}`, { method: "DELETE" }); }
    catch (error) { /* Closing should still work if the Brain is unavailable. */ }
  }
  $("enrollment-dialog").close(); resetEnrollment();
}

document.addEventListener("DOMContentLoaded", async () => {
  updateVisibility();
  $("add-device").addEventListener("click", openEnrollment);
  $("enrollment-close").addEventListener("click", closeEnrollment);
  $("enrollment-cancel").addEventListener("click", closeEnrollment);
  $("forget-device").addEventListener("click", forgetDevice);
  $("profile-form").addEventListener("submit", saveProfile);
  $("task-form").addEventListener("submit", submitTask);
  watchOnlineStatus("offline-banner");
  setupInstallPrompt("install-button");
  await loadProfile();
  await refresh();
  setInterval(refresh, 4000);
});

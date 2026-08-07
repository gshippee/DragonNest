const STORAGE_KEY = "dragonnest.myDevice";
const STYLE_VECTOR = "concise-vs-verbose-layer-7";
const state = { myDevice: loadMyDevice(), enrollment: null };
// Synthesized audio for the response currently on screen. Kept so replaying an
// answer costs nothing -- synthesis runs on the NPU and takes real seconds.
const speech = { text: "", url: "", audio: null };
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
  clearSpeech();
  $("speak-button").disabled = true;
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
  const output = task.result?.output_text
    || "DragonNest could not complete that request. Please try again.";
  $("result-state").className = `status-pill ${succeeded ? "healthy" : "failed"}`;
  $("result-state").textContent = succeeded ? "Done" : "Try again";
  $("result-output").textContent = output;
  if (speech.text !== output) clearSpeech();
  $("speak-button").disabled = !succeeded;
}

function setSpeakIcon(name) {
  // lucide swaps <i data-lucide> for an <svg> in place, so changing the icon
  // means restoring the placeholder and re-running the converter.
  $("speak-button").innerHTML = `<i data-lucide="${name}"></i>`;
  if (window.lucide) lucide.createIcons();
}

function stopSpeech() {
  if (speech.audio) {
    speech.audio.pause();
    speech.audio = null;
  }
  $("speak-button").classList.remove("busy");
  setSpeakIcon("volume-2");
}

function clearSpeech() {
  stopSpeech();
  if (speech.url) URL.revokeObjectURL(speech.url);
  speech.url = "";
  speech.text = "";
}

async function requestSpeech(text) {
  // Not the shared api() helper: this response is a .wav body, not JSON.
  const response = await fetch("/api/speech", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    // 409 is the NPU being busy with the language model, not a failure --
    // speech yields to it rather than competing for the same DSP session.
    let detail = { 409: "DragonNest is thinking. Try the speaker again in a moment.",
                   503: "This DragonNest host isn't set up for speech yet." }[response.status]
                 || "DragonNest could not read that aloud.";
    try { detail = (await response.json()).detail || detail; } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.blob();
}

function playSpeech() {
  const button = $("speak-button");
  button.classList.remove("busy");
  setSpeakIcon("square");
  const audio = new Audio(speech.url);
  speech.audio = audio;
  audio.addEventListener("ended", stopSpeech);
  audio.addEventListener("error", () => { toast("That audio could not be played."); stopSpeech(); });
  audio.play().catch(() => { toast("Your browser blocked playback."); stopSpeech(); });
}

async function speakResult() {
  const button = $("speak-button");
  if (speech.audio) { stopSpeech(); return; }  // the button stops playback mid-clip
  const text = $("result-output").textContent.trim();
  if (!text) return;
  if (speech.url && speech.text === text) { playSpeech(); return; }

  button.disabled = true;
  button.classList.add("busy");
  setSpeakIcon("loader");  // "loader", not "loader-2": the latter is a deprecated lucide alias
  try {
    const blob = await requestSpeech(text);
    if (speech.url) URL.revokeObjectURL(speech.url);
    speech.url = URL.createObjectURL(blob);
    speech.text = text;
    playSpeech();
  } catch (error) {
    toast(error.message);
    stopSpeech();
  } finally {
    button.disabled = false;
  }
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
  $("enrollment-host-row").hidden = true;
  $("enrollment-host").innerHTML = "";
}

const LOOPBACK_HOSTNAMES = new Set(["localhost", "127.0.0.1", "::1", ""]);

// Candidate Brain addresses a phone might need to reach, most likely first.
// window.location.hostname is included when it isn't loopback because it's
// the address that's known to work right now (it's how this page loaded).
async function candidateBrainHosts() {
  const pageHost = window.location.hostname;
  const candidates = LOOPBACK_HOSTNAMES.has(pageHost) ? [] : [pageHost];
  try {
    const info = await api("/api/server-info");
    for (const address of info.lan_addresses || []) {
      if (!candidates.includes(address)) candidates.push(address);
    }
  } catch { /* server-info unavailable; fall back to what we already have */ }
  if (!candidates.length) candidates.push(pageHost || "127.0.0.1");
  return candidates;
}

async function createEnrollment() {
  try {
    const hosts = await candidateBrainHosts();
    const picker = $("enrollment-host");
    picker.innerHTML = hosts.map((host) => `<option value="${esc(host)}">${esc(host)}</option>`).join("");
    $("enrollment-host-row").hidden = hosts.length < 2;
    await refreshEnrollmentQr();
  } catch (error) {
    $("enrollment-status").textContent = "Could not create a code";
    toast(error.message);
  }
}

async function refreshEnrollmentQr() {
  try {
    const session = await api("/api/enrollment-sessions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        brain_host: $("enrollment-host").value || (await candidateBrainHosts())[0],
        brain_port: 50051,
        use_tls: window.location.protocol === "https:",
        ttl_seconds: 300,
      }),
    });
    state.enrollment = session;
    $("enrollment-qr").src = `${session.qr_url}?t=${Date.now()}`;
    updateEnrollmentStatus(session);
    clearInterval(window.__enrollmentTimer);
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
  $("enrollment-host").addEventListener("change", refreshEnrollmentQr);
  $("forget-device").addEventListener("click", forgetDevice);
  $("profile-form").addEventListener("submit", saveProfile);
  $("task-form").addEventListener("submit", submitTask);
  $("speak-button").addEventListener("click", speakResult);
  watchOnlineStatus("offline-banner");
  setupInstallPrompt("install-button");
  await loadProfile();
  await refresh();
  setInterval(refresh, 4000);
});

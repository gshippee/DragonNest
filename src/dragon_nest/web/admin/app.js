const state = { devices: [], tasks: [], vectors: [], selectedTask: null, followLatest: true, enrollment: null, behaviorProfiles: [], lastPlan: null, provisioning: [] };

const $ = (id) => document.getElementById(id);

async function refresh() {
  try {
    const [health, devices, tasks, events, provisioning] = await Promise.all([
      api("/api/health"), api("/api/devices"), api("/api/tasks"), api("/api/events?limit=120"), api("/api/provisioning")
    ]);
    state.devices = devices;
    const selection = DragonNestAdminSelection.reconcileSelection(
      tasks, state.selectedTask?.task_id || "", state.followLatest);
    state.tasks = selection.tasks;
    state.provisioning = provisioning;
    for (const id of ["origin-device", "behavior-origin"]) {
      const origin = $(id);
      const selectedOrigin = origin.value;
      origin.innerHTML = '<option value="">None</option>' + devices.map((device) => `<option value="${esc(device.device_id)}" ${device.device_id === selectedOrigin ? "selected" : ""}>${esc(device.display_name)} (${esc(device.device_id)})</option>`).join("");
      if (!selectedOrigin && devices.length === 1) origin.value = devices[0].device_id;
    }
    renderProvisioning();
    $("brain-dot").classList.add("online");
    $("brain-state").textContent = `${health.brain_id} online`;
    renderDevices(); renderLiveRequests(); renderEvents(events); renderFabricSummary();
    const updated = state.tasks.find((task) => task.task_id === selection.selectedTaskId);
    state.selectedTask = updated || null;
    if (updated) { renderTask(updated); renderLiveRequests(); }
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
    const personal = device.personal_profile;
    const inventory = device.hardware || {};
    const advertised = new Set((device.models || []).map((model) => model.model_id));
    const modelInventory = (device.models || []).map((model) => `<li><span class="artifact-dot ${model.warm ? "warm" : "installed"}"></span><span><strong>${esc(model.model_id)}</strong><small>${esc(model.runtime || "runtime unknown")} · ${esc((model.accelerators || []).join("+") || "accelerator unknown")}</small></span><em>${model.warm ? "warm" : "installed · cold"}</em></li>`).join("");
    const deployedInventory = (device.deployments || []).filter((item) => item.state !== "absent" && !advertised.has(item.artifact_id)).map((item) => `<li><span class="artifact-dot ${esc(item.state)}"></span><span><strong>${esc(item.artifact_id)}</strong><small>deployed artifact</small></span><em>${esc(item.state)}</em></li>`).join("");
    const runtimeNames = [...new Set((device.models || []).map((model) => model.runtime).filter(Boolean))];
    const memoryLabel = h.available_memory_mb === 0 ? "Unknown" : fmt(h.available_memory_mb, " MB");
    const simulatedFields = device.simulated_fields || [];
    const memoryIsSimulated = simulatedFields.includes("available_memory_mb")
      || (device.simulated_constraint && simulatedFields.length === 0);
    const memoryFlag = memoryIsSimulated ? '<span class="simulated-tag">SIMULATED</span>' : "";
    const hardwareLabel = [inventory.soc_model, inventory.npu_status ? `NPU ${inventory.npu_status}` : ""].filter(Boolean).join(" · ") || device.platform;
    return `<article class="device-card ${statusClass(device.status)}">
      <div class="device-title"><div><h3>${esc(device.display_name)}</h3><p>${esc(device.device_id)} · ${esc(hardwareLabel)}</p></div><div><span class="status-pill ${statusClass(device.status)}">${esc(device.status)}</span> <button class="icon-btn simulate" data-device="${esc(device.device_id)}" title="Simulate device state" aria-label="Simulate ${esc(device.display_name)}"><i data-lucide="gauge"></i></button><button class="icon-btn remove-device" data-device="${esc(device.device_id)}" title="Remove device" aria-label="Remove ${esc(device.display_name)}"><i data-lucide="trash-2"></i></button></div></div>
      <div class="device-facts"><div><span>RAM available</span><strong>${memoryLabel} ${memoryFlag}</strong></div><div><span>Thermal</span><strong>${h.thermal_level < 0 ? "Unknown" : decimal(h.thermal_level)}</strong></div><div><span>CPU / GPU / NPU</span><strong>${utilizationTriplet(h)}</strong></div><div><span>Active / runtime</span><strong>${device.active_tasks.length} / ${esc(runtimeNames.join(", ") || "idle")}</strong></div></div>
      <div class="artifact-heading"><span>Available models & artifacts</span><span>${(device.models || []).length}</span></div>
      <ul class="artifact-inventory">${modelInventory}${deployedInventory}${!modelInventory && !deployedInventory ? '<li class="empty">No advertised models</li>' : ""}</ul>
      ${personal ? `<div class="device-owner">Profile: ${esc(personal.person_name)}</div>` : ""}
    </article>`;
  }).join("") : '<div class="empty">No registered devices</div>';
  document.querySelectorAll(".simulate").forEach((button) => button.addEventListener("click", () => openSimulation(button.dataset.device)));
  document.querySelectorAll(".remove-device").forEach((button) => button.addEventListener("click", () => removeDevice(button.dataset.device)));
}

function renderFabricSummary() {
  const healthy = state.devices.filter((device) => device.connected && device.status === "HEALTHY").length;
  const models = new Set(state.devices.flatMap((device) => (device.models || []).map((model) => model.model_id)));
  $("fabric-summary").textContent = `${state.devices.length} devices · ${models.size} active models · ${healthy} healthy`;
}

function renderLiveRequests() {
  $("follow-state").textContent = state.followLatest ? "Following latest" : `Pinned to ${state.selectedTask?.task_id || "request"}`;
  $("follow-state").classList.toggle("pinned", !state.followLatest);
  $("follow-latest").hidden = state.followLatest;
  $("recent-requests").innerHTML = state.tasks.length ? state.tasks.map((task) => {
    const destination = requestDestination(task);
    const selected = task.task_id === state.selectedTask?.task_id;
    return `<button type="button" class="request-row ${selected ? "selected" : ""}" data-task="${esc(task.task_id)}"><time>${formatTaskTime(task.created_at)}</time><span class="request-copy"><strong>${esc(shortPrompt(task.request))}</strong><small>${esc(deviceName(task.origin_device_id) || "Admin")} → ${esc(destination)}</small></span><span class="request-mode">${esc(preferenceLabel(task.preferred_mode))} · ${esc(task.execution_mode)}</span><span class="status-pill ${statusClass(task.state)}">${esc(task.state)}</span><span class="request-latency">${task.result?.latency_ms ? `${task.result.latency_ms} ms` : ""}</span></button>`;
  }).join("") : '<div class="empty">No requests yet</div>';
  document.querySelectorAll(".request-row").forEach((button) => button.addEventListener("click", () => selectTask(button.dataset.task, true)));
}

function renderTask(task) {
  const profile = task.profile;
  $("selected-prompt").innerHTML = `<span>User request</span><strong>${esc(task.request)}</strong>`;
  $("profile-strip").innerHTML = profile ? `<span class="profile-item">Compute preference <strong>${esc(preferenceLabel(task.preferred_mode))}</strong></span><span class="profile-item">Profile requested <strong>${esc(profileLabel(task.behavior_profile_id || "balanced"))}</strong></span><span class="profile-item">Profile realization <strong>${esc(realizationLabel(task.profile_realization))}</strong></span><span class="profile-item">Classification <strong>${esc(profile.task_class)} / ${esc(profile.complexity)}</strong></span><span class="profile-item">Model selected <strong>${esc(task.result?.metrics?.model_id || task.progress?.[0]?.model_id || "pending")}</strong></span><span class="profile-item">Artifact <strong>${esc(task.selected_artifact_id || "pending")}</strong></span><span class="profile-item">Execution topology <strong>${esc(task.execution_mode)}</strong></span><span class="profile-item">Confidence <strong>${Math.round(profile.confidence * 100)}%</strong></span><span class="profile-item">Privacy <strong>${esc(profile.privacy_tier)}</strong></span><span class="profile-item">Reducer <strong>${esc(task.reducer)}</strong></span>${task.origin_device_id ? `<span class="profile-item">Origin <strong>${esc(task.origin_device_id)}</strong></span>` : ""}${task.steering?.enabled ? `<span class="profile-item">Runtime steering <strong>${esc(task.steering.vector_id)}</strong></span>` : ""}` : "";
  $("route-trace").innerHTML = task.route_reasons?.length ? task.route_reasons.map((reason) => `<li>${esc(reason)}</li>`).join("") : '<li class="empty">No route trace available</li>';
  $("route-topology").innerHTML = renderTopology(task);
  $("progress").textContent = JSON.stringify(task.progress || []);
  $("result-state").className = `status-pill ${statusClass(task.state)}`;
  $("result-state").textContent = task.state;
  $("result-output").textContent = task.result?.output_text || task.error_message || "No result available.";
  const metrics = task.result?.metrics;
  $("result-meta").innerHTML = task.result ? `<span>Device: <strong>${esc(task.result.device_id)}</strong></span><span>Latency: <strong>${task.result.latency_ms} ms</strong></span>${metrics ? `<span>Runtime: <strong>${esc(metrics.runtime_name)} ${esc(metrics.runtime_version)}</strong></span><span>Accelerator: <strong>${esc(metrics.accelerator)}</strong></span>` : ""}` : "";
}

function renderTopology(task) {
  if (task.progress?.length && task.execution_mode === "layer_pipeline") {
    const groups = [];
    for (const stage of task.progress) {
      const latest = groups[groups.length - 1];
      if (!latest || latest.deviceId !== stage.device_id) groups.push({ deviceId: stage.device_id, stages: [] });
      groups[groups.length - 1].stages.push(stage);
    }
    return `<div class="pipeline-label">Qwen3-1.7B · one-cut pipeline</div><div class="pipeline-route">${groups.map((group, index) => `${index ? '<span class="route-arrow">→<small>hidden boundary</small></span>' : ""}<div class="stage-device"><strong>${esc(deviceName(group.deviceId) || group.deviceId)}</strong><div class="stage-blocks">${group.stages.map((stage) => `<span title="${esc(stage.model_id)}">${esc(stageLabel(stage.id))}</span>`).join("")}</div></div>`).join("")}</div>`;
  }
  const origin = deviceName(task.origin_device_id) || task.origin_device_id || "Application";
  const result = task.result;
  const destination = deviceName(result?.device_id) || result?.device_id || "Awaiting route";
  const metrics = result?.metrics;
  const rejected = (task.route_reasons || []).find((reason) => /rejected|no eligible compatible local capacity|insufficient|exceeds available|below .*memory/i.test(reason));
  return `<div class="single-route"><div class="route-node origin"><span>Origin</span><strong>${esc(origin)}</strong></div><div class="route-arrow">→${rejected ? `<small>${esc(rejected)}</small>` : ""}</div><div class="route-node destination"><span>Selected execution</span><strong>${esc(destination)}</strong><small>${esc(metrics?.model_id || "model pending")}${metrics ? ` · ${esc(metrics.runtime_name)} · ${esc(metrics.accelerator)}` : ""}</small></div></div>`;
}

function renderEvents(events) {
  $("events").innerHTML = events.length ? events.map((event) => `<div class="event-row"><span>${Number(event.timestamp).toFixed(2)}</span><span class="event-type">${esc(event.type)}</span><span class="event-subject">${esc(event.subject)}</span><span class="event-message">${esc(event.message)}</span></div>`).join("") : '<div class="empty">No events</div>';
}

function renderVectors() {
  $("vector-list").innerHTML = state.vectors.length ? state.vectors.map((vector) => `<div class="vector-item"><strong>${esc(vector.vector_id)}</strong><span>${esc(vector.model_family)} · alpha ${vector.alpha_min} to ${vector.alpha_max} (default ${vector.default_alpha}) · ${esc(vector.positions.join(", "))}</span></div>`).join("") : '<div class="empty">No steering vectors configured</div>';
}

function deviceName(deviceId) {
  return state.devices.find((device) => device.device_id === deviceId)?.display_name || "";
}

function preferenceLabel(value) {
  const labels = { auto: "Auto", local: "Local", elastic: "Elastic", quality: "Quality", private: "Private", fast: "Fast", parallel: "Parallel" };
  return labels[value] || value || "Auto";
}

function profileLabel(value) {
  const labels = { balanced: "Balanced", concise: "Concise", detailed: "Detailed" };
  return labels[value] || value || "Balanced";
}

function realizationLabel(value) {
  const labels = {
    none: "Base model",
    baked_profile: "Baked activation profile",
    prompt_profile: "Prompt conditioning",
    runtime_vector: "Runtime activation steering",
  };
  return labels[value] || value || "Base model";
}

function shortPrompt(value) {
  const normalized = String(value || "").replace(/\s+/g, " ").trim();
  return normalized.length > 64 ? `${normalized.slice(0, 61)}...` : normalized || "Untitled request";
}

function formatTaskTime(timestamp) {
  if (!timestamp) return "--:--:--";
  return new Date(Number(timestamp) * 1000).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function stageLabel(stageId) {
  const match = String(stageId || "").match(/(?:stage[-_ ]?|s)(\d+)/i);
  return match ? `S${match[1]}` : stageId;
}

function requestDestination(task) {
  if (task.progress?.length) {
    const groups = [];
    for (const item of task.progress) {
      const label = deviceName(item.device_id) || item.device_id;
      const latest = groups[groups.length - 1];
      if (!latest || latest.label !== label) groups.push({ label, stages: [] });
      groups[groups.length - 1].stages.push(stageLabel(item.id));
    }
    return groups.map((group) => `${group.label}${group.stages.length ? ` [${group.stages.join(",")}]` : ""}`).join(" → ");
  }
  const result = task.result;
  const destination = deviceName(result?.device_id) || result?.device_id || "routing";
  const model = result?.metrics?.model_id;
  return model ? `${destination} / ${model}` : destination;
}

function utilizationTriplet(health) {
  const value = (amount) => (amount == null || amount < 0 ? "?" : `${decimal(amount * 100, 0)}%`);
  return `${value(health.cpu_utilization)} / ${value(health.gpu_utilization)} / ${value(health.npu_utilization)}`;
}

function selectTask(taskId, pin = false) {
  const task = state.tasks.find((item) => item.task_id === taskId);
  if (!task) return;
  if (pin) state.followLatest = false;
  state.selectedTask = task; renderTask(task); renderLiveRequests();
}

function resumeFollowLatest() {
  state.followLatest = true;
  if (state.tasks.length) selectTask(state.tasks[0].task_id);
  else renderLiveRequests();
}

async function loadVectors() {
  state.vectors = await api("/api/steering-vectors");
  $("vector-id").innerHTML = state.vectors.map((vector) => `<option value="${esc(vector.vector_id)}">${esc(vector.vector_id)}</option>`).join("");
  $("enrollment-vector").innerHTML = '<option value="">None</option>' + state.vectors.map((vector) => `<option value="${esc(vector.vector_id)}">${esc(vector.vector_id)}</option>`).join("");
  applyVectorDefaults();
  renderVectors();
}

function applyVectorDefaults() {
  const vector = state.vectors.find((item) => item.vector_id === $("vector-id").value);
  if (!vector) return;
  $("alpha").min = vector.alpha_min; $("alpha").max = vector.alpha_max; $("alpha").value = vector.default_alpha; $("alpha-value").value = vector.default_alpha;
  $("positions").innerHTML = vector.positions.map((position) => `<option value="${esc(position)}" ${position === vector.default_positions ? "selected" : ""}>${esc(position)}</option>`).join("");
}

function applyEnrollmentVectorDefaults() {
  const vector = state.vectors.find((item) => item.vector_id === $("enrollment-vector").value);
  if (!vector) { $("enrollment-alpha").removeAttribute("min"); $("enrollment-alpha").removeAttribute("max"); $("enrollment-alpha").value = "0"; return; }
  $("enrollment-alpha").min = vector.alpha_min; $("enrollment-alpha").max = vector.alpha_max; $("enrollment-alpha").value = vector.default_alpha;
  $("enrollment-positions").innerHTML = vector.positions.map((position) => `<option value="${esc(position)}" ${position === vector.default_positions ? "selected" : ""}>${esc(position)}</option>`).join("");
}

async function submitTask(event) {
  event.preventDefault();
  const button = $("submit-button"); button.disabled = true;
  const vector = state.vectors.find((item) => item.vector_id === $("vector-id").value);
  try {
    const response = await api("/api/tasks", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({ request_text: $("request").value, preferred_mode: $("preferred-mode").value, execution_mode: $("execution-mode").value, origin_device_id: $("origin-device").value, reducer: $("reducer").value, use_profile_steering: $("profile-steering").checked, steering: { enabled: $("steering-enabled").checked, vector_id: vector?.vector_id || "", model_family: vector?.model_family || "", target_layer: vector?.default_layer || 0, alpha: Number($("alpha").value), positions: $("positions").value } }) });
    toast(response.success ? `Task ${response.task_id} succeeded` : `${response.error_code}: ${response.error_message}`);
    await refresh(); selectTask(response.task_id);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

function openSimulation(deviceId) {
  const device = state.devices.find((item) => item.device_id === deviceId); if (!device) return;
  $("simulation-device").value = deviceId; $("sim-thermal").value = device.health.thermal_level; $("sim-load").value = device.health.accelerator_utilization; $("sim-rtt").value = device.health.network_rtt_ms; $("sim-battery").value = Math.max(0, device.health.battery_pct); $("sim-memory").value = device.health.available_memory_mb; $("sim-steering").checked = device.runtime_steering_enabled !== false; $("sim-offline").checked = device.status === "OFFLINE";
  $("sim-artifacts").innerHTML = (device.deployments || []).length ? "<span class=\"sim-artifacts-title\">Artifact deployment states</span>" + device.deployments.map((item) => `<label class="field sim-artifact-row">${esc(item.artifact_id)}<select data-artifact="${esc(item.artifact_id)}" data-original="${esc(item.state)}">${["absent", "installed", "warm"].map((option) => `<option value="${option}" ${option === item.state ? "selected" : ""}>${option}</option>`).join("")}</select></label>`).join("") : "";
  updateSimulationOutputs(); $("simulation-dialog").showModal();
}

function updateSimulationOutputs() { $("sim-thermal-value").value = $("sim-thermal").value; $("sim-load-value").value = $("sim-load").value; }

async function applySimulation(event) {
  event.preventDefault();
  const artifactStates = {};
  document.querySelectorAll("#sim-artifacts select").forEach((select) => {
    if (select.value !== select.dataset.original) artifactStates[select.dataset.artifact] = select.value;
  });
  const payload = { thermal_level: Number($("sim-thermal").value), accelerator_utilization: Number($("sim-load").value), network_rtt_ms: Number($("sim-rtt").value), battery_pct: Number($("sim-battery").value), available_memory_mb: Number($("sim-memory").value), runtime_steering_enabled: $("sim-steering").checked, offline: $("sim-offline").checked };
  if (Object.keys(artifactStates).length) payload.artifact_states = artifactStates;
  try { await api(`/api/devices/${encodeURIComponent($("simulation-device").value)}/simulate`, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(payload) }); $("simulation-dialog").close(); await refresh(); }
  catch (error) { toast(error.message); }
}

async function loadBehaviorProfiles() {
  state.behaviorProfiles = await api("/api/behavior-profiles");
  $("behavior-profile").innerHTML = '<option value="">None</option>' + state.behaviorProfiles.map((profile) => `<option value="${esc(profile.profile_id)}">${esc(profile.display_name)} (${esc(profile.fallback_policy)})</option>`).join("");
}

function behaviorPayload() {
  return {
    request_text: $("behavior-request").value,
    base_model_family: $("behavior-family").value,
    behavior_profile_id: $("behavior-profile").value,
    estimated_input_tokens: Number($("behavior-input-tokens").value) || 256,
    estimated_output_tokens: Number($("behavior-output-tokens").value) || 128,
    privacy: $("behavior-privacy").value,
    latency_preference: $("behavior-latency").value,
    origin_device_id: $("behavior-origin").value,
    fallback_policy_override: $("behavior-fallback").value
  };
}

function renderPlan(plan) {
  state.lastPlan = plan;
  const summary = [];
  summary.push(`<span class="profile-item">Profile <strong>${esc(plan.behavior_profile || "none")}</strong></span>`);
  if (plan.fallback_policy) summary.push(`<span class="profile-item">Fallback policy <strong>${esc(plan.fallback_policy)}</strong></span>`);
  if (plan.chosen) {
    summary.push(`<span class="profile-item">Chosen <strong>${esc(plan.chosen.device_id)} / ${esc(plan.chosen.artifact_id)}</strong></span>`);
    summary.push(`<span class="profile-item">Realized via <strong>${esc(plan.chosen.realization_mode)}</strong></span>`);
  } else {
    summary.push(`<span class="profile-item">Result <strong>${esc(plan.error_code || "no route")}</strong></span>`);
    if (plan.provisioning_hint) summary.push(`<button id="provision-hint" class="btn-secondary" type="button" data-profile="${esc(plan.provisioning_hint)}"><i data-lucide="hammer"></i><span>Provision '${esc(plan.provisioning_hint)}'</span></button>`);
  }
  $("behavior-summary").innerHTML = summary.join("");
  const hint = $("provision-hint");
  if (hint) hint.addEventListener("click", () => provisionProfile(hint.dataset.profile));
  $("behavior-explanation").innerHTML = plan.explanation.map((line) => `<li>${esc(line)}</li>`).join("");
  $("behavior-candidates").innerHTML = plan.candidates.length ? plan.candidates.map((candidate) => {
    const chosen = plan.chosen && candidate.device_id === plan.chosen.device_id && candidate.artifact_id === plan.chosen.artifact_id && candidate.realization_mode === plan.chosen.realization_mode;
    const verdict = candidate.feasible
      ? (chosen ? '<span class="status-pill healthy">chosen</span>' : '<span class="status-pill neutral">feasible</span>')
      : `<span class="status-pill failed">rejected</span><div class="reason-list">${candidate.rejection_reasons.map((reason) => esc(reason)).join("<br>")}</div>`;
    const cost = candidate.cost ? `${Math.round(candidate.cost.total_ms)} ms` : "—";
    const memory = candidate.memory ? ` · ${candidate.memory.total_mb}/${candidate.memory.available_mb} MB${candidate.memory.estimated_fields.length ? " (est)" : ""}` : "";
    return `<tr class="${chosen ? "chosen" : ""}"><td>${esc(candidate.device_id)}</td><td>${esc(candidate.artifact_id)}</td><td>${esc(candidate.realization_mode)}</td><td>${esc(candidate.deployment_state)}</td><td>${cost}${memory}</td><td>${verdict}</td></tr>`;
  }).join("") : '<tr><td colspan="6" class="empty">No candidates generated</td></tr>';
  if (window.lucide) lucide.createIcons();
}

async function previewBehaviorRoute() {
  const button = $("behavior-preview"); button.disabled = true;
  try { renderPlan(await api("/api/route-plan", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(behaviorPayload()) })); }
  catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

async function executeBehaviorTask(event) {
  event.preventDefault();
  const button = $("behavior-execute"); button.disabled = true;
  try {
    const response = await api("/api/behavior-tasks", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({ ...behaviorPayload(), timeout_ms: 30000 }) });
    if (response.route_plan) renderPlan(response.route_plan);
    toast(response.success ? `Task ${response.task_id} succeeded on ${response.device_id}` : `${response.error_code}: ${response.error_message || "no feasible deployment"}`);
    await refresh(); selectTask(response.task_id);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

async function provisionProfile(profileId) {
  const profile = state.behaviorProfiles.find((item) => item.profile_id === profileId);
  const baked = profile?.realizations?.find((realization) => realization.baked_artifact_id);
  if (!baked) { toast(`Profile ${profileId} declares no bake target`); return; }
  const deviceId = $("behavior-origin").value || state.devices[0]?.device_id;
  if (!deviceId) { toast("No device available for provisioning"); return; }
  try {
    await api("/api/provisioning", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({ profile_id: profileId, device_id: deviceId, artifact_id: baked.baked_artifact_id }) });
    toast(`Provisioning ${baked.baked_artifact_id} on ${deviceId}`);
    await refresh();
  } catch (error) { toast(error.message); }
}

function renderProvisioning() {
  const jobs = state.provisioning || [];
  $("provisioning-count").textContent = `${jobs.length} job${jobs.length === 1 ? "" : "s"}`;
  $("provisioning-jobs").innerHTML = jobs.length ? jobs.map((job) => {
    const terminal = job.state === "warm" || job.state === "failed";
    return `<tr><td>${esc(job.job_id)}</td><td>${esc(job.profile_id)}</td><td>${esc(job.device_id)}</td><td>${esc(job.artifact_id)}</td><td><span class="status-pill ${job.state === "warm" ? "healthy" : job.state === "failed" ? "failed" : "running"}">${esc(job.state)}</span></td><td>${esc(job.detail)}</td><td>${terminal ? "" : `<button class="btn-secondary advance-job" data-job="${esc(job.job_id)}" type="button">Advance</button>`}</td></tr>`;
  }).join("") : '<tr><td colspan="7" class="empty">No provisioning jobs</td></tr>';
  document.querySelectorAll(".advance-job").forEach((button) => button.addEventListener("click", async () => {
    try { await api(`/api/provisioning/${encodeURIComponent(button.dataset.job)}/advance`, { method: "POST" }); await refresh(); }
    catch (error) { toast(error.message); }
  }));
}

function openEndpointDialog() {
  $("endpoint-device-id").value = "";
  $("endpoint-display-name").value = "";
  $("endpoint-base-url").value = "";
  $("endpoint-admin-token").value = sessionStorage.getItem("endpointAdminToken") || "";
  $("endpoint-provider").value = "dragonnest";
  $("endpoint-credential-env").value = "";
  $("endpoint-profile-context").checked = false;
  $("endpoint-model-id").value = "endpoint-model";
  $("endpoint-model-family").value = "endpoint";
  $("endpoint-model-role").value = "general";
  $("endpoint-task-classes").value = "chat_qa";
  $("endpoint-max-context").value = "4096";
  $("endpoint-total-memory").value = "0";
  $("endpoint-auto-discover").checked = true;
  $("endpoint-submit").disabled = false;
  applyProviderMode();
  $("endpoint-dialog").showModal();
}

const ENDPOINT_PROVIDER_COPY = {
  dragonnest: {
    credentialLabel: "Credential environment variable",
    credentialPlaceholder: "ENDPOINT_API_TOKEN",
    modelIdLabel: "Model ID",
    modelIdPlaceholder: "",
  },
  openai_chat: {
    credentialLabel: "API key environment variable",
    credentialPlaceholder: "CIRRASCALE_API_KEY",
    modelIdLabel: "Model IDs (comma-separated)",
    modelIdPlaceholder: "Llama-3.1-8B, Llama-3.1-70B",
  },
};

function applyProviderMode() {
  const provider = $("endpoint-provider").value;
  const isOpenAi = provider === "openai_chat";
  const copy = ENDPOINT_PROVIDER_COPY[provider];
  $("endpoint-credential-env-field").firstChild.textContent = copy.credentialLabel;
  $("endpoint-credential-env").placeholder = copy.credentialPlaceholder;
  $("endpoint-model-id-field").firstChild.textContent = copy.modelIdLabel;
  $("endpoint-model-id").placeholder = copy.modelIdPlaceholder;
  $("endpoint-auto-discover-row").hidden = isOpenAi;
  $("endpoint-fetch").hidden = isOpenAi;
  for (const id of ["endpoint-model-family-field", "endpoint-model-role-field", "endpoint-task-classes-field"]) {
    $(id).hidden = isOpenAi;
  }
  if (isOpenAi) {
    $("endpoint-auto-discover").checked = false;
    $("endpoint-model-id").value = "";
  }
  toggleEndpointMode();
}

function endpointHeaders() {
  const input = $("endpoint-admin-token");
  const token = input?.value || sessionStorage.getItem("endpointAdminToken") || "";
  if (input?.value) sessionStorage.setItem("endpointAdminToken", input.value);
  return {"Content-Type": "application/json", "Authorization": `Bearer ${token}`};
}

function toggleEndpointMode() {
  $("endpoint-manual-fields").hidden = $("endpoint-auto-discover").checked;
}

function applyDiscoveredInfo(info) {
  if (info.display_name && !$("endpoint-display-name").value.trim()) $("endpoint-display-name").value = info.display_name;
  if (info.total_memory_mb) $("endpoint-total-memory").value = info.total_memory_mb;
  const models = info.models || [];
  if (models.length) {
    const first = models[0];
    $("endpoint-model-id").value = first.model_id || "";
    $("endpoint-model-family").value = first.model_family || "";
    $("endpoint-model-role").value = first.role || "";
    $("endpoint-task-classes").value = (first.task_classes || []).join(", ");
    $("endpoint-max-context").value = first.max_context_tokens || 0;
  }
  toast(models.length
    ? `Found ${models.length} model${models.length === 1 ? "" : "s"}: ${models.map((m) => m.model_id).join(", ")}`
    : "Endpoint reachable, but reported no models via /info.");
}

async function fetchEndpointDetails() {
  if ($("endpoint-provider").value === "openai_chat") {
    toast("Auto-discovery isn't supported for OpenAI-compatible endpoints; enter model IDs manually");
    return;
  }
  const baseUrl = $("endpoint-base-url").value.trim();
  if (!baseUrl) { toast("Enter an endpoint URL first"); return; }
  const button = $("endpoint-fetch"); button.disabled = true;
  try {
    const info = await api("/api/rest-devices/discover", {
      method: "POST", headers: endpointHeaders(),
      body: JSON.stringify({ base_url: baseUrl, credential_env: $("endpoint-credential-env").value.trim() }),
    });
    applyDiscoveredInfo(info);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

function endpointModelsPayload(provider) {
  const maxContextTokens = Number($("endpoint-max-context").value) || 0;
  if (provider === "openai_chat") {
    return $("endpoint-model-id").value.split(",").map((item) => item.trim()).filter(Boolean).map((modelId) => ({
      model_id: modelId,
      model_family: "openai-compatible",
      role: "general",
      task_classes: ["chat_qa"],
      max_context_tokens: maxContextTokens,
      warm: true,
      quality_score: 0.6
    }));
  }
  if ($("endpoint-auto-discover").checked) return [];
  return [{
    model_id: $("endpoint-model-id").value.trim(),
    model_family: $("endpoint-model-family").value.trim(),
    role: $("endpoint-model-role").value.trim(),
    task_classes: $("endpoint-task-classes").value.split(",").map((item) => item.trim()).filter(Boolean),
    max_context_tokens: maxContextTokens,
    warm: true,
    quality_score: 0.6
  }];
}

async function registerEndpoint(event) {
  event.preventDefault();
  const button = $("endpoint-submit"); button.disabled = true;
  const deviceId = $("endpoint-device-id").value.trim();
  const provider = $("endpoint-provider").value;
  try {
    const models = endpointModelsPayload(provider);
    if (provider === "openai_chat" && !models.length) throw new Error("Enter at least one model ID");
    await api("/api/rest-devices", {
      method: "POST", headers: endpointHeaders(),
      body: JSON.stringify({
        device_id: deviceId,
        display_name: $("endpoint-display-name").value.trim(),
        provider,
        base_url: $("endpoint-base-url").value.trim(),
        credential_env: $("endpoint-credential-env").value.trim(),
        allow_profile_context: $("endpoint-profile-context").checked,
        total_memory_mb: Number($("endpoint-total-memory").value) || 0,
        models
      })
    });
    $("endpoint-dialog").close();
    toast(`Endpoint ${deviceId} registered`);
    await refresh();
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}

async function removeDevice(deviceId) {
  const device = state.devices.find((item) => item.device_id === deviceId);
  const name = device?.display_name || deviceId;
  if (!window.confirm(`Remove ${name} from this DragonNest fabric? The device must be enrolled again before it can rejoin.`)) return;
  try { await api(`/api/devices/${encodeURIComponent(deviceId)}`, {method: "DELETE"}); toast(`${name} removed`); await refresh(); }
  catch (error) { toast(error.message); }
}

function openEnrollment() {
  $("enrollment-host").value = window.location.hostname || "127.0.0.1";
  $("enrollment-port").value = "50051";
  $("enrollment-tls").checked = false;
  $("enrollment-person").value = "";
  $("enrollment-device-name").value = "";
  $("enrollment-mode").value = "auto";
  $("enrollment-vector").value = "";
  $("enrollment-alpha").value = "0";
  $("enrollment-positions").value = "last";
  $("enrollment-notes").value = "";
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
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ brain_host: $("enrollment-host").value.trim(), brain_port: Number($("enrollment-port").value), use_tls: $("enrollment-tls").checked, ttl_seconds: 300, person_name: $("enrollment-person").value.trim(), device_name: $("enrollment-device-name").value.trim(), preferred_mode: $("enrollment-mode").value, steering_vector_id: $("enrollment-vector").value, steering_alpha: Number($("enrollment-alpha").value), steering_positions: $("enrollment-positions").value, notes: $("enrollment-notes").value.trim() })
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
    if (session.status !== "PENDING") { clearInterval(window.__enrollmentTimer); await refresh(); }
  } catch (error) { clearInterval(window.__enrollmentTimer); toast(error.message); }
}

function updateEnrollmentStatus(session) {
  const remaining = Math.max(0, Math.ceil(session.expires_at - Date.now() / 1000));
  if (session.status === "CLAIMED") {
    $("enrollment-status").textContent = `Enrolled ${session.claimed_device_id}`;
    $("enrollment-expiry").textContent = "Connected credential issued";
  } else if (session.status === "PENDING") {
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
    try { await api(`/api/enrollment-sessions/${encodeURIComponent(state.enrollment.session_id)}`, {method: "DELETE"}); }
    catch (error) { toast(error.message); }
  }
  $("enrollment-dialog").close(); resetEnrollment();
}

document.addEventListener("DOMContentLoaded", async () => {
  $("task-form").addEventListener("submit", submitTask);
  $("behavior-form").addEventListener("submit", executeBehaviorTask);
  $("behavior-preview").addEventListener("click", previewBehaviorRoute);
  $("follow-latest").addEventListener("click", resumeFollowLatest);
  $("steering-enabled").addEventListener("change", (event) => $("steering-controls").hidden = !event.target.checked);
  $("vector-id").addEventListener("change", applyVectorDefaults);
  $("alpha").addEventListener("input", () => $("alpha-value").value = $("alpha").value);
  $("refresh-events").addEventListener("click", refresh);
  $("simulation-form").addEventListener("submit", applySimulation);
  $("sim-cancel").addEventListener("click", () => $("simulation-dialog").close());
  $("sim-thermal").addEventListener("input", updateSimulationOutputs);
  $("sim-load").addEventListener("input", updateSimulationOutputs);
  $("add-device").addEventListener("click", openEnrollment);
  $("add-endpoint").addEventListener("click", openEndpointDialog);
  $("endpoint-form").addEventListener("submit", registerEndpoint);
  $("endpoint-close").addEventListener("click", () => $("endpoint-dialog").close());
  $("endpoint-cancel").addEventListener("click", () => $("endpoint-dialog").close());
  $("endpoint-fetch").addEventListener("click", fetchEndpointDetails);
  $("endpoint-auto-discover").addEventListener("change", toggleEndpointMode);
  $("endpoint-provider").addEventListener("change", applyProviderMode);
  $("enrollment-form").addEventListener("submit", createEnrollment);
  $("enrollment-close").addEventListener("click", closeEnrollment);
  $("enrollment-cancel").addEventListener("click", closeEnrollment);
  $("enrollment-vector").addEventListener("change", applyEnrollmentVectorDefaults);
  watchOnlineStatus("offline-banner");
  setupInstallPrompt("install-button");
  await loadVectors(); await loadBehaviorProfiles(); await refresh(); setInterval(refresh, 1000);
});

const state = { report: { devices: [], capabilities: [], pipelines: [], task_classes: {} } };

const $ = (id) => document.getElementById(id);

const BOTTLENECK_LABEL = {
  compute: "Compute-bound",
  memory: "Memory-bound",
  communication: "Communication-bound",
  balanced: "Balanced",
};

const CHART = { width: 720, height: 420, padLeft: 54, padRight: 20, padTop: 20, padBottom: 54 };

async function refresh() {
  try {
    const [health, report] = await Promise.all([api("/api/health"), api("/api/regimes")]);
    state.report = report;
    $("brain-dot").classList.add("online");
    $("brain-state").textContent = `${health.brain_id} online`;
  } catch (error) {
    $("brain-dot").classList.remove("online");
    $("brain-state").textContent = "Brain unavailable";
  }
  $("device-count").textContent = `${state.report.devices.length} device${state.report.devices.length === 1 ? "" : "s"}`;
  renderChart(state.report.devices);
  renderTaskClasses(state.report.task_classes);
  renderCapabilityTable(state.report.capabilities);
  renderPipelineTable(state.report.pipelines);
  if (window.lucide) lucide.createIcons();
}

function renderLegend() {
  const items = [
    ["compute", "Compute-bound"],
    ["memory", "Memory-bound"],
    ["communication", "Communication-bound"],
    ["balanced", "Balanced — no axis is currently binding"],
  ];
  $("regime-legend").innerHTML = items.map(([key, label]) => `<span class="regime-legend-item"><span class="regime-swatch ${key}"></span>${esc(label)}</span>`).join("");
}

function clamp01(value) { return Math.max(0, Math.min(1, value)); }

function renderChart(devices) {
  const { width, height, padLeft, padRight, padTop, padBottom } = CHART;
  const x0 = padLeft, x1 = width - padRight, y0 = padTop, y1 = height - padBottom;
  const toX = (compute) => x0 + clamp01(compute) * (x1 - x0);
  const toY = (memory) => y1 - clamp01(memory) * (y1 - y0);
  const threshold = 0.65;

  const parts = [];
  parts.push(`<rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" fill="none" class="chart-grid-line"></rect>`);
  parts.push(`<line x1="${toX(threshold)}" y1="${y0}" x2="${toX(threshold)}" y2="${y1}" class="chart-threshold-line"></line>`);
  parts.push(`<line x1="${x0}" y1="${toY(threshold)}" x2="${x1}" y2="${toY(threshold)}" class="chart-threshold-line"></line>`);
  parts.push(`<text x="${toX(threshold) + 4}" y="${y0 + 10}" class="chart-zone-label">balanced &ge; ${threshold}</text>`);
  parts.push(`<text x="${x0 + 4}" y="${toY(threshold) - 4}" class="chart-zone-label">balanced &ge; ${threshold}</text>`);
  parts.push(`<text x="${(x0 + x1) / 2}" y="${height - 14}" text-anchor="middle" class="chart-axis-label">Compute headroom →</text>`);
  parts.push(`<text x="${padLeft - 40}" y="${(y0 + y1) / 2}" text-anchor="middle" transform="rotate(-90 ${padLeft - 40} ${(y0 + y1) / 2})" class="chart-axis-label">Memory headroom →</text>`);

  devices.forEach((device) => {
    const cx = toX(device.compute);
    const cy = toY(device.memory);
    const r = 7 + clamp01(device.communication) * 13;
    const hitR = Math.max(r + 10, 16);
    const bottleneck = BOTTLENECK_LABEL[device.bottleneck] ? device.bottleneck : "balanced";
    parts.push(`<g>
      <circle class="regime-node-hit" data-device="${esc(device.device_id)}" cx="${cx}" cy="${cy}" r="${hitR}" tabindex="0" role="img" aria-label="${esc(device.display_name)}: ${esc(BOTTLENECK_LABEL[bottleneck])}"></circle>
      <circle class="regime-node-mark ${bottleneck}" cx="${cx}" cy="${cy}" r="${r}"></circle>
      <text class="regime-node-label" x="${cx + r + 6}" y="${cy + 4}">${esc(device.display_name || device.device_id)}</text>
    </g>`);
  });

  $("regime-chart").innerHTML = parts.join("");
  attachChartInteractions(devices);
}

function attachChartInteractions(devices) {
  const byId = new Map(devices.map((device) => [device.device_id, device]));
  const tooltip = $("regime-tooltip");
  document.querySelectorAll(".regime-node-hit").forEach((node) => {
    const device = byId.get(node.dataset.device);
    if (!device) return;
    const show = (clientX, clientY) => {
      const bottleneck = BOTTLENECK_LABEL[device.bottleneck] ? device.bottleneck : "balanced";
      tooltip.innerHTML = `<strong></strong><span class="tooltip-bottleneck ${bottleneck}"></span><div class="tooltip-reason"></div>`;
      tooltip.querySelector("strong").textContent = device.display_name || device.device_id;
      tooltip.querySelector(".tooltip-bottleneck").textContent = BOTTLENECK_LABEL[bottleneck];
      const reasonNode = tooltip.querySelector(".tooltip-reason");
      reasonNode.appendChild(document.createTextNode(
        `Compute ${Math.round(device.compute * 100)}% · Memory ${Math.round(device.memory * 100)}% · Communication ${Math.round(device.communication * 100)}%`
      ));
      device.reasons.forEach((reason) => {
        const line = document.createElement("div");
        line.textContent = reason;
        reasonNode.appendChild(line);
      });
      tooltip.hidden = false;
      const left = Math.min(clientX + 14, window.innerWidth - 272);
      const top = Math.min(clientY + 14, window.innerHeight - 140);
      tooltip.style.left = `${Math.max(8, left)}px`;
      tooltip.style.top = `${Math.max(8, top)}px`;
    };
    const hide = () => { tooltip.hidden = true; };
    node.addEventListener("pointermove", (event) => show(event.clientX, event.clientY));
    node.addEventListener("pointerenter", (event) => show(event.clientX, event.clientY));
    node.addEventListener("pointerleave", hide);
    node.addEventListener("focus", () => {
      const rect = node.getBoundingClientRect();
      show(rect.left, rect.top);
    });
    node.addEventListener("blur", hide);
  });
}

function renderTaskClasses(taskClasses) {
  const entries = Object.entries(taskClasses || {});
  $("task-class-grid").innerHTML = entries.length ? entries.map(([taskClass, buckets]) => `
    <article class="task-class-card">
      <h3>${esc(taskClass)}</h3>
      <div class="task-class-row"><span class="row-label">Achievable on</span>${buckets.achievable_on.length ? buckets.achievable_on.map((id) => `<span class="chip achievable">${esc(id)}</span>`).join("") : '<span class="chip">none</span>'}</div>
      <div class="task-class-row"><span class="row-label">Blocked on</span>${buckets.blocked_on.length ? buckets.blocked_on.map((id) => `<span class="chip blocked">${esc(id)}</span>`).join("") : '<span class="chip">none</span>'}</div>
    </article>`).join("") : '<div class="empty">No advertised task classes</div>';
}

function renderCapabilityTable(capabilities) {
  $("capability-table").innerHTML = capabilities.length ? capabilities.map((row) => `
    <tr>
      <td>${esc(row.device_id)}</td>
      <td>${esc(row.model_id)}</td>
      <td>${esc(row.task_classes.join(", "))}</td>
      <td><span class="status-pill ${row.achievable ? "healthy" : "unhealthy"}">${row.achievable ? "Achievable" : "Blocked"}</span></td>
      <td>${row.limiting_factor ? esc(BOTTLENECK_LABEL[row.limiting_factor] || row.limiting_factor) : "—"}</td>
      <td>${esc(row.detail)}</td>
    </tr>`).join("") : '<tr><td colspan="6" class="empty">No registered models</td></tr>';
}

function renderPipelineTable(pipelines) {
  $("pipeline-table").innerHTML = pipelines.length ? pipelines.map((row) => `
    <tr>
      <td>${esc(row.pipeline_id)}</td>
      <td>${esc(row.left_device_id)} → ${esc(row.right_device_id)}</td>
      <td>${row.combined_rtt_ms < 0 ? "Unknown" : `${decimal(row.combined_rtt_ms, 0)} ms`}</td>
      <td><span class="status-pill ${row.achievable ? "healthy" : "unhealthy"}">${row.achievable ? "Achievable" : "Blocked"}</span></td>
      <td>${row.limiting_factor ? esc(BOTTLENECK_LABEL[row.limiting_factor] || row.limiting_factor) : "—"}</td>
    </tr>`).join("") : '<tr><td colspan="5" class="empty">No split-layer pipeline pairs found</td></tr>';
}

document.addEventListener("DOMContentLoaded", async () => {
  renderLegend();
  await refresh();
  setInterval(refresh, 2000);
});

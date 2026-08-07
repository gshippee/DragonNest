# DragonNest Demo Runbook

A heterogeneous Snapdragon AI-fabric demo: a Snapdragon X Elite laptop and a
Samsung Galaxy S25 Ultra register with the DragonNest Brain, report live
telemetry, and the Brain routes behavior-aware requests to concrete
executable deployments — visibly, with explanations, fallbacks, and
provisioning.

The smoke run and simulated agents below need **no Qualcomm SDKs**. They
exercise the real Brain/control-plane code with mock executors. The separate
X Elite Genie command uses the hardware manifest and only advertises a model
after its local bundle passes path and checksum validation.

## 0. One-time setup (Windows PowerShell)

```powershell
cd DragonNest
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q     # expect: all tests pass
```

## 1. Zero-hardware smoke run (recommended first)

```powershell
.venv\Scripts\python.exe scripts\demo_scenarios.py
```

This starts an in-process Brain plus simulated `x-elite-01` and
`s25-ultra-01` agents and walks scenarios A–G below, printing every route
explanation and PASS/FAIL. It must end with `All scenarios passed.`

## 2. Live demo: Brain + two agents + dashboard

Terminal 1 — the Brain (gRPC on 50051, dashboard on 8080):

```powershell
.venv\Scripts\python.exe scripts\run_brain.py
```

Terminal 2 — simulated/control-plane X Elite agent:

```powershell
.venv\Scripts\python.exe scripts\run_agent.py --device-id x-elite-01 --fabric configs\demo-fleet.yaml
```

This command registers the deterministic `demo-fleet.yaml` X Elite device and
runs its portable mock artifacts. It does **not** claim Genie or NPU execution.

The real X Elite worker is physically verified with Qwen3-4B / Genie / HTP,
including `HardwareRuntimeAdapter`, Brain-to-Agent gRPC, and the machine's real
LAN interface. Set the same enrollment token as the Brain and run the pinned,
one-command launcher on the X Elite:

```powershell
$env:DRAGONNEST_ENROLLMENT_TOKEN = "<same token as Brain>"
.\scripts\run_xelite_worker.ps1 -Brain <brain-host>:50051
```

The launcher refuses ambiguous or checksum-mismatched bundles, then advertises
`ModelCapability.model_id=qwen3-4b-genie` as installed/cold with runtime
steering disabled. The detailed sanitized proof is in
`docs/results/xelite_worker_status.md`. Only the genuinely separate-host
desktop-Brain round trip remains to be physically verified.

Terminal 3 — the Galaxy S25 Ultra agent (simulated locally; on the real
phone install the Android Agent APK and point it at the Brain's LAN address):

```powershell
.venv\Scripts\python.exe scripts\run_agent.py --device-id s25-ultra-01 --fabric configs\demo-fleet.yaml
```

Open <http://127.0.0.1:8080/admin>.

**Expected registration flow:** both devices appear as cards within ~2
seconds, status HEALTHY, with SoC identity chips ("Snapdragon X Elite
X1E-80-100", "Qualcomm SM8750 Snapdragon 8 Elite"), live memory / thermal /
battery / RTT (updated every heartbeat; on Windows, memory and battery are
real native probes), steering-realization chips (`runtime_vector`,
`baked_profile`, `prompt_profile`), and per-artifact deployment chips
(`small-chat-v1 · warm`, …).

> Note: agents heartbeat *real* host telemetry. For a deterministic
> on-stage flow, pin values through the per-device **Simulate** dialog
> (gauge icon on each card) — that is also how every scenario below is
> triggered.

## 3. Scenario walkthrough (Behavior Routing panel, section 04)

All scenarios use the **Behavior Routing** band: choose model family
`mock`, fill the fields listed, press **Preview route** (no execution) or
**Route & execute**. The candidate table shows every (device, artifact,
realization) considered, with rejection reasons and cost breakdowns; the
explanation list is the scheduler's own narrative.

### A. Warm-device preference
- Profile: None. Defaults everywhere. Preview.
- **Expect:** `x-elite-01 / small-chat-v1 (warm)` chosen with cold-load 0;
  `s25-ultra-01 / small-chat-v1` feasible but ~1.5 s more expensive
  (cold load) — visible as an "Alternative" line.

### B. Behavior locality
- Profile: `Concise`. Preview.
- **Expect:** two feasible realizations — runtime vector on the phone
  (it advertises steering support) and the baked artifact
  `small-chat-v1-concise-baked` warm on the laptop. The baked laptop
  deployment wins on cost; the runtime alternative appears with its price
  delta and the explanation names both realizations.

### C. Thermal reroute
- Simulate on `x-elite-01`: set artifact `small-chat-v1` → `absent`
  (Simulate dialog → Artifact deployment states). Preview (no profile):
  **phone wins**.
- Simulate on `s25-ultra-01`: Thermal → `0.92`. Preview again:
  **laptop wins**, phone rejected with `device health is UNHEALTHY
  (thermal=0.92 …)`.
- Reset: thermal back to `0.3`, artifact back to `warm`.

### D. Memory rejection
- Simulate on `s25-ultra-01`: Available memory → `1000` MB.
- Preview with Input tokens `2600`, Output tokens `400`.
- **Expect:** phone rejected **before dispatch** with
  `projected memory 2078 MB exceeds available 1000 MB (fixed 512 +
  artifact 900 + KV cache 282 + margin 384)`; components are flagged as
  estimates; laptop chosen. Reset memory to `6144`.

### E. Runtime steering unavailable
- Simulate on `s25-ultra-01`: uncheck **Runtime steering supported**.
- Profile `Concise`, Preview → chosen realization is `baked_profile` on the
  laptop and the explanation says the preferred `runtime_vector` fell back
  under `allow_baked_equivalent` *without changing the profile*.
- Set Fallback override → `Exact only`, Preview → **no route**:
  `BEHAVIOR_UNAVAILABLE`. Re-enable runtime steering.

### F. Failure recovery
- Easiest live version: submit **Route & execute** (no profile), then kill
  the laptop agent (Ctrl+C in terminal 2) mid-run, or pre-arm
  `scripts/demo_scenarios.py` scenario F which forces a disconnect exactly
  when the task lands.
- **Expect:** the laptop attempt is fenced as `DEVICE_OFFLINE`, the task
  retries once on `s25-ultra-01` and succeeds; the Routing Trace and task
  attempts show both attempts; a late laptop result cannot overwrite the
  accepted one (result fencing).

### G. Missing profile → provisioning
- Profile: `Family Assistant`, Preview.
- **Expect:** `BEHAVIOR_UNAVAILABLE` — the profile's only realization is an
  unbuilt bake target — plus a **Provision 'family-assistant'** button.
- Click it (target defaults to the origin/first device). The Provisioning
  band (section 05) shows the job; press **Advance** repeatedly:
  `missing → build_queued → compiling → validating → ready_remote →
  downloading → installed → warm`. Every detail line is prefixed `[mock]` —
  the UI never claims a real AI Hub compile ran.
- Preview the profile again: it now routes to
  `family-assistant-v0-baked (warm)`.

## 4. Classic execution modes (unchanged)

The Task Submission band (section 03) still drives the original paths:
single, data-parallel fanout/replica race, and the fixed qwen3-0.6b
layer-pipeline template (`part-a` on the phone, `part-b` on the laptop),
including origin-preference, private mode, and disconnect recovery.
`scripts/demo_grpc.py` remains a self-contained check for those.

## 5. Fallback commands when physical hardware is unavailable

| Need | Command |
| --- | --- |
| Full scripted demo, no hardware, no dashboard | `.venv\Scripts\python.exe scripts\demo_scenarios.py` |
| Same fleet in the dashboard without a phone | run both `run_agent.py` commands on the laptop (as in section 2) |
| Classic modes smoke test | `.venv\Scripts\python.exe scripts\demo_grpc.py` |
| Whole test suite | `.venv\Scripts\python.exe -m pytest -q` |

## 6. Real-hardware notes

- **X Elite laptop:** use `pc-01` and `scripts/run_xelite_worker.ps1` for the
  real worker; `x-elite-01` is the simulated/control-plane demo identity.
  Windows memory/battery telemetry is native. Real Qwen3-4B Genie/HTP
  execution is physically verified; runtime steering remains correctly off.
- **Galaxy S25 Ultra:** build/install the Android Agent
  (`scripts/build_android.sh`), enroll via the dashboard **Add device** QR,
  and it registers over the LAN with real inventory (SoC `SM8750`) and
  telemetry. Mock execution works out of the box; NPU execution needs the
  Genie JNI bundle described in `android-agent/README.md`.
- Remaining physical Android execution/provisioning obligations are specified
  in `docs/HARDWARE_CONTRACT.md`.

## 7. True cross-host X Elite proof

This is the remaining physical claim: the Brain/application runs on a separate
desktop while `pc-01` executes Qwen3-4B through Genie/HTP on the X Elite.

Desktop terminal 1:

```powershell
$env:DRAGONNEST_ENROLLMENT_TOKEN = "<random-shared-token>"
.\scripts\run_demo_brain.ps1
```

If the variable is omitted, the launcher generates and prints a random token.
It binds gRPC/dashboard to the LAN, prints candidate LAN addresses, disables
the unrelated HTTP endpoint pool, refuses `dev-token`, and does not alter
Windows Firewall.

X Elite terminal (no LLM interaction):

```powershell
git pull
$env:DRAGONNEST_ENROLLMENT_TOKEN = "<same-token>"
.\scripts\run_xelite_worker.ps1 -Brain <desktop-lan-ip>:50051
```

Desktop terminal 2, after `pc-01` appears in the dashboard:

```powershell
.\.venv\Scripts\python.exe scripts\validate_remote_xelite.py `
  --brain-http http://127.0.0.1:8080 `
  --device-id pc-01 `
  --model-id qwen3-4b-genie `
  --runs 3 `
  --calibrate-memory `
  --output "$env:TEMP\dragonnest-crosshost-xelite.json"
```

The harness waits for an eligible, reachable gRPC worker; requires the exact
installed/cold model and binary artifact IDs; rejects runtime-steering claims;
submits through `/api/behavior-tasks`; and accepts only a scheduler-selected
`pc-01` result whose execution metrics report `qwen3-4b-genie`, `genie`, and
`htp`. The proof contains task/attempt IDs, route/explanation, latencies,
telemetry summaries, and output SHA-256—never raw output, tokens, paths, or
credentials.

Brain telemetry is sampled throughout each call. The proof marks memory
capture reliable only if the Brain exposes the active task and at least two
distinct in-task heartbeat updates. A false value means the observed minimum
may have missed a faster transient; do not change `estimated_memory_mb` from
that run alone. For optional 100 ms host-side sampling, start this in a second
X Elite terminal before rerunning the desktop harness:

```powershell
.\scripts\sample_xelite_memory.ps1 `
  -DurationSeconds 180 `
  -Output "$env:TEMP\dragonnest-xelite-memory-samples.json"
```

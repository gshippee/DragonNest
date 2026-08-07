# DragonNest Demo Runbook

A heterogeneous Snapdragon AI-fabric demo: a Snapdragon X Elite laptop and a
Samsung Galaxy S25 Ultra register with the DragonNest Brain, report live
telemetry, and the Brain routes behavior-aware requests to concrete
executable deployments — visibly, with explanations, fallbacks, and
provisioning.

The stage topology is not a separate desktop Brain. The Snapdragon X Elite
laptop runs both the Brain and its local `pc-01` Genie/HTP Device Agent. The
Galaxy S25 Ultra runs PersonaCare plus the Android Agent and sends every user
request to that laptop Brain over the LAN.

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

## 2. Physical stage demo: phone RAM pressure reroutes to X Elite

On the Snapdragon X Elite laptop, from the repository root:

```powershell
.\scripts\run_xelite_demo.ps1
```

This opens two visible windows: a LAN-visible Brain/dashboard with a 75-second
default task timeout, and `pc-01` connected to it over loopback using the
physically verified `qwen3-4b-genie` bundle through Genie/HTP. It prints the
exact `<laptop-lan-ip>:50051` address for PersonaCare and the dashboard URL.
The launcher generates a shared nontrivial enrollment token when one is not
already set. It does not modify Windows Firewall.

### Enroll PersonaCare

1. Open the printed `http://<laptop-lan-ip>:8080/admin` URL.
2. Choose **Add device**, use `<laptop-lan-ip>` and port `50051`, and create
   the enrollment QR.
3. On the S25, open PersonaCare, choose **Scan enrollment QR**, and scan it.
4. Wait until both the phone and `pc-01` are connected in the dashboard.

### Exact two-request procedure

Preconditions:

- X Elite Brain and the real `pc-01` worker are connected.
- S25 PersonaCare is enrolled.
- Compute preference is **Auto**.
- Persona is **Balanced** and no steering is active.
- PersonaCare **Demo controls** has simulated low RAM disabled/reset.

Request 1:

```text
What is the capital of Japan?
```

Use this deliberately simple wording so the classifier stays in `single`
mode. Avoid `compare`, `analyze`, numbered lists, semicolons, and `and also`,
which can select another execution mode. Expect origin `phone-01`, execution
mode `single`, and selected device `phone-01`.

Next open PersonaCare **Demo controls**, enable simulated low RAM, and set
available RAM to exactly `64 MB`. The Android Agent schedules an immediate
heartbeat on its existing gRPC connection. Wait until the dashboard phone
card shows 64 MB; this should be nearly immediate.

Request 2: send another simple chat question with the same controls. Expect:

- `phone-01` rejected because 64 MB is below its model minimum;
- selected device `pc-01`;
- selected model `qwen3-4b-genie`;
- runtime `genie`, accelerator `htp`;
- PersonaCare displays `Ran on <X Elite laptop display name>`.

This complete two-request transition has passed on the physical S25 and X
Elite stage setup. Repeat it as a demo procedure; it is no longer an
unverified topology claim.

The current thin APK executes Request 1 with `android-mock-v1`. This proves
the phone-origin routing and live RAM-pressure transition, not phone NPU
execution. Do not claim phone NPU execution until a real Android runtime
artifact is integrated and physically verified.

Choosing **Local** sends `preferred_mode=local`. Under 64 MB that request must
fail on the phone rather than fall back to the laptop. Legacy callers that send
`preferred_mode=private` retain the same hard origin-only placement semantics.

## 3. Control-plane fallback: Brain + two simulated agents + dashboard

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
LAN interface. For the stage topology use section 2. The lower-level worker
command remains available for diagnostics:

```powershell
$env:DRAGONNEST_ENROLLMENT_TOKEN = "<same token as Brain>"
.\scripts\run_xelite_worker.ps1 -Brain <brain-host>:50051
```

The launcher refuses ambiguous or checksum-mismatched bundles, then advertises
`ModelCapability.model_id=qwen3-4b-genie` as installed/cold with runtime
steering disabled. The detailed sanitized proof is in
`docs/results/xelite_worker_status.md`. The hackathon demo does not depend on
a separate desktop Brain.

Terminal 3 — the Galaxy S25 Ultra agent (simulated locally; on the real
phone install the Android Agent APK and point it at the Brain's LAN address):

```powershell
.venv\Scripts\python.exe scripts\run_agent.py --device-id s25-ultra-01 --fabric configs\demo-fleet.yaml
```

Open <http://127.0.0.1:8080/admin>. The primary control-room view contains
only **Device Registry**, **Live Requests**, and **Selected Request**. Live
Requests follows the newest PersonaCare or admin request by default. Clicking
an older request pins it across refreshes; choose **Follow latest** to resume.
Each device card exposes installed/cold/warm model inventory and labels a
RAM override as **SIMULATED**. The trash button removes a device from the
fabric; that device must be explicitly enrolled again before it can rejoin.
Manual submission, behavior routing, steering, provisioning, and endpoint
administration remain under **Advanced**; the raw event stream is under the
collapsed **Event log** diagnostics panel.

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

## 4. Scenario walkthrough (Advanced -> Behavior Routing)

Expand **Advanced** and use **Behavior Routing**: choose model family
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
  retries once on `s25-ultra-01` and succeeds; the Selected Request route explanation and task
  attempts show both attempts; a late laptop result cannot overwrite the
  accepted one (result fencing).

### G. Missing profile → provisioning
- Profile: `Family Assistant`, Preview.
- **Expect:** `BEHAVIOR_UNAVAILABLE` — the profile's only realization is an
  unbuilt bake target — plus a **Provision 'family-assistant'** button.
- Click it (target defaults to the origin/first device). The **Provisioning**
  area under Advanced shows the job; press **Advance** repeatedly:
  `missing → build_queued → compiling → validating → ready_remote →
  downloading → installed → warm`. Every detail line is prefixed `[mock]` —
  the UI never claims a real AI Hub compile ran.
- Preview the profile again: it now routes to
  `family-assistant-v0-baked (warm)`.

## 5. Classic execution modes (unchanged)

**Advanced -> Manual Task Submission** still drives the original paths:
single, data-parallel fanout/replica race, and the fixed qwen3-0.6b
layer-pipeline template (`part-a` on the phone, `part-b` on the laptop),
including origin-preference, private mode, and disconnect recovery.
`scripts/demo_grpc.py` remains a self-contained check for those.

## 6. Fallback commands when physical hardware is unavailable

| Need | Command |
| --- | --- |
| Full scripted demo, no hardware, no dashboard | `.venv\Scripts\python.exe scripts\demo_scenarios.py` |
| Same fleet in the dashboard without a phone | run both `run_agent.py` commands on the laptop (as in section 3) |
| Classic modes smoke test | `.venv\Scripts\python.exe scripts\demo_grpc.py` |
| Whole test suite | `.venv\Scripts\python.exe -m pytest -q` |

## 7. Real-hardware notes

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

## 8. Act 3 control plane: variable four-stage split

The recovered `qwen3-1.7b-w4a16-demo-v1` control plane now reconstructs S0-S3,
routes one laptop-prefix/phone-suffix cut with cumulative stage memory, performs
explicit prefill/decode passes, and cleans stage-local sessions. Desktop tests
prove the 2+2, 3+1, and 4+0 cuts through the public gRPC path. This remains
mock/control-plane evidence until both targets execute their real QNN contexts.

PersonaCare exposes compute intent independently from its behavior persona:

- **Auto** uses the deterministic task classifier. High-complexity requests use
  the pipeline only when its complete executable chain is advertised; otherwise
  Auto falls back to the best feasible full model.
- **Local** is hard origin-only single-model execution.
- **Elastic** explicitly requests `qwen3-1.7b-w4a16-demo-v1` and returns
  `ELASTIC_UNAVAILABLE` rather than silently substituting the 4B model.
- **Quality** selects the highest-quality feasible full model and never treats
  pipeline stages as complete models.

The exact artifact staging, evidence boundary, and acceptance commands are in
`docs/QWEN3_1_7B_HANDOFF.md`. These preferences make the control-plane decision
inspectable; they do not upgrade the 1.7B pipeline to physical evidence.

The eventual UI milestone is: phone request -> Brain selects complex quality
mode -> fixed S0-S3 cut -> prompt pass -> repeated decode passes -> final text
returned to PersonaCare.

Keep the four S25 context binaries outside the APK. With one debuggable S25
attached to the X Elite artifact-cache host, stage them with:

```powershell
.\scripts\deploy_s25_demo_artifacts.ps1 -CacheRoot C:\DragonNestArtifacts
```

The wrapper verifies all four source hashes before its first phone mutation,
checks that exactly one expected S25 and the debuggable app are present, copies
through a temporary ADB directory into `files/dragonnest-models/`, verifies the
installed hashes, writes the runtime manifest, and restarts PersonaCare. Until
the direct JNI QNN session/graph/KV binding passes physical validation it ends
with `ARTIFACTS INSTALLED` / `RUNTIME NOT YET EXECUTABLE`, and the Agent does
not advertise those stages. The two 0.6B bundles are optional: missing bytes
are reported and any unchecksummed bytes are refused rather than guessed.

The planner/router also already contain `data_parallel` support. New parallel
architecture is deferred until phone-to-phone, phone-low-RAM-to-X-Elite, and
the fixed phone-plus-X-Elite pipeline have passed in that order.

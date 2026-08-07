# DragonNest

**A trusted fabric of Snapdragon devices that share one AI workload.**

You register your laptop and phone once. From then on, every request you make
is routed to whichever device can answer it best *right now* — accounting for
battery, heat, memory, network, and privacy — and DragonNest shows you exactly
why it chose that device.

```
                 ┌──────────────────────────────────────┐
   User  ───────▶│  BRAIN  (laptop)                     │
   web app       │  classify → plan → route → explain   │
   :8080/        │  dashboard :8080  ·  gRPC :50051     │
                 └───────┬──────────────────┬───────────┘
                         │                  │
                 ┌───────▼──────┐   ┌───────▼──────┐
                 │ Agent pc-01  │   │ Agent phone  │   ← devices report live
                 │ X Elite/HTP  │   │ S25 / APK    │     telemetry + capability
                 └──────────────┘   └──────────────┘
```

Three things make it more than a load balancer:

| Capability | What it means |
|---|---|
| **Health-aware routing** | Reroutes around thermal, battery, memory, and disconnect events mid-flight, with a one-retry fallback. |
| **Behavior steering** | Answer style (concise / balanced / detailed) is a routed capability — runtime steering vector, baked profile, or prompt profile, with declared fallbacks. |
| **Split execution** | One task can run on one device, shard across several, or run as a layer pipeline with hidden-state boundaries handed device to device. |

---

## Start here

| I am… | Go to |
|---|---|
| Setting up the fabric, running the Brain, adding devices | [Admin guide](#admin-guide) |
| Just want to connect my phone and ask questions | [User guide](#user-guide) |
| Evaluating / demoing this in 5 minutes | [Zero-hardware demo](#2-verify-the-install-no-hardware-needed) |
| Looking for design detail | [Documentation map](#documentation-map) |

---

## Dependencies

**Required for everything in the Admin and User guides:** Python 3.10 or newer
on Windows, macOS, or Linux. Nothing else. `pip install -e ".[dev]"` pulls in:

| Package | Used for |
|---|---|
| `fastapi` + `uvicorn` | Dashboard and REST API |
| `grpcio`, `protobuf>=7.35.1` | Brain ⇄ Agent control plane |
| `httpx` | HTTP endpoint devices |
| `pydantic`, `PyYAML` | Request models and config files |
| `numpy` | Steering vectors and boundary tensors |
| `qrcode` | Device enrollment codes |
| `grpcio-tools`, `pytest` *(dev extra)* | Proto regeneration and tests |

**Only for optional paths:**

| Path | Needs |
|---|---|
| Android app | JDK 17, Android SDK 35, `adb` |
| Real Snapdragon NPU execution | QAIRT 2.48 SDK with Genie, plus the model bundle referenced by `configs/model-artifacts.yaml` — setup in [docs/HARDWARE_CONTRACT.md](docs/HARDWARE_CONTRACT.md) |
| X Elite local model tooling | `pip install -e ".[xelite]"` → `psutil`, `torch`, `transformers`, `sentencepiece` |
| Qualcomm AI Hub validation | A **separate** venv from `requirements-ai-hub.txt` (`qai-hub` pins protobuf 6; DragonNest needs protobuf 7) |

No Qualcomm SDK, no phone, and no special network hardware are needed for
setup, tests, or the demo — without them, Agents execute with a mock executor
behind the real control plane.

---

## Admin guide

The admin runs the Brain, decides which devices are trusted, and watches the
fabric. All admin commands run from the repository root.

### 1. Install

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

<details>
<summary>macOS / Linux equivalent</summary>

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Every command below works the same way — substitute `.venv/bin/python` for
`.venv\Scripts\python.exe`, and forward slashes for backslashes.
</details>

### 2. Verify the install (no hardware needed)

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\demo_scenarios.py
```

`demo_scenarios.py` boots an in-process Brain plus two simulated devices and
walks the full feature set — warm-model preference, behavior locality, thermal
reroute, memory rejection, steering fallback, disconnect recovery, and
provisioning — printing each route explanation and PASS/FAIL. It must end with
`All scenarios passed.` This is the fastest way to see what DragonNest does.

### 3. Start the Brain

```powershell
.venv\Scripts\python.exe scripts\run_brain.py
```

That single process serves everything:

| URL | Who | What |
|---|---|---|
| `http://<host>:8080/` | User | The ask-a-question app |
| `http://<host>:8080/admin` | Admin | Device registry, live requests, route traces |
| `http://<host>:8080/regimes` | Admin | What each device can actually achieve, and why |
| `<host>:50051` | Devices | gRPC control plane for Agents |

It binds `0.0.0.0` so phones on the same Wi‑Fi can reach it. Use
`--http-host 127.0.0.1` to restrict it to this machine. State (devices,
profiles, endpoints) persists in `local/dragonnest-state.sqlite3` and is
restored on restart.

Useful flags: `--http-port`, `--address` (gRPC), `--enrollment-token`,
`--default-task-timeout-ms`, `--state-db`. Run with `--help` for the full list.

### 4. Add devices

Pick whichever applies. A device becomes routable only after it registers and
reports a capability.

**a. A local software agent** — the quickest way to have something to route to:

```powershell
.venv\Scripts\python.exe scripts\run_agent.py --device-id pc-01
.venv\Scripts\python.exe scripts\run_agent.py --device-id phone-01
```

Run each in its own terminal. They connect to `127.0.0.1:50051` with the
default `dev-token` and execute with a mock executor — real control plane,
simulated compute.

**b. A phone running the DragonNest Android app** — see
[Connect your phone](#2-connect-your-phone) in the user guide. As admin you can
also drive the same flow from **Add device** on `/admin`.

**c. A trusted HTTP inference endpoint** — for a box that speaks HTTP instead
of gRPC:

```powershell
$env:DRAGONNEST_HTTP_ENDPOINT_ADMIN_TOKEN = "replace-me"
$env:EDGE_API_TOKEN = "endpoint-bearer-token"
.venv\Scripts\python.exe scripts\run_brain.py --http-endpoint-allow-host edge-box.local
```

Then register it under **Advanced → Add HTTP Endpoint Device** on `/admin`.
The credential field takes an *environment variable name* (`EDGE_API_TOKEN`),
never a secret value — secrets are never written to SQLite. Hostnames must be
listed with `--http-endpoint-allow-host`; literal IPs must fall inside a
`--http-endpoint-allow-cidr`. The endpoint must implement `GET /health`,
`GET /info`, `POST /execute`, `POST /execute_shard`,
`POST /execute_pipeline_stage`, and `POST /cancel`. `/info` metadata is
validated before it can become a routing capability.

### 5. Drive the dashboard

`/admin` is the fabric console. Top level shows the three things you watch
during operation:

- **Device Registry** — every device, live telemetry, and health state
  (healthy / degraded / stale / offline). The **Simulate** control forces
  offline, thermal, battery, load, memory, reachability, or RTT conditions on
  any device so you can prove reroute behavior on demand.
- **Live Requests** — every task from the user app and from admin, streaming.
- **Selected Request** — pick any request to see **Why Brain chose this route**
  as an ordered trace, plus the result and its timings.

Everything else is folded under **Advanced** so it stays out of the way:
manual task submission, behavior routing candidates and rejections, steering
vectors, provisioning jobs, and HTTP endpoints.

`/regimes` answers a different question — not "where did this go?" but "what is
this fabric capable of?" It maps the achievable tradeoffs per task class, per
model, and across cross-device pipelines.

### 6. Submit tasks from the CLI (optional)

```powershell
.venv\Scripts\python.exe scripts\submit_task.py "Compare both options and recommend one."
```

```powershell
# Keep it on the originating device — never leaves the phone
.venv\Scripts\python.exe scripts\submit_task.py `
  --preferred-mode private --origin-device-id phone-01 "Rewrite this private note."

# Race the same shard on two devices, take the first success
.venv\Scripts\python.exe scripts\submit_task.py `
  --execution-mode data_parallel --reducer first_success "Answer this quickly."

# Simulate a hot, loaded, high-latency phone
.venv\Scripts\python.exe scripts\run_agent.py --device-id phone-01 `
  --simulate-thermal 0.95 --simulate-load 0.90 --simulate-rtt 180
```

### 7. Production transport (optional)

Production mode requires mutual TLS. The Brain verifies that each registration
fingerprint matches the certificate gRPC authenticated:

```powershell
.venv\Scripts\python.exe scripts\run_brain.py --production `
  --tls-certificate certs\brain.crt --tls-key certs\brain.key `
  --tls-client-ca certs\device-ca.crt
.venv\Scripts\python.exe scripts\run_agent.py --device-id phone-01 `
  --tls-ca certs\device-ca.crt `
  --tls-certificate certs\phone-01.crt --tls-key certs\phone-01.key
```

QR enrollment is a **development-mode** onboarding path; production enrollment
must issue an mTLS client certificate rather than a token credential.

---

## User guide

As a user you never touch a config file. You install the app, connect your
phone once, say what you care about, and ask.

### 1. Install the Android app

With JDK 17 and Android SDK 35 installed:

```bash
scripts/build_android.sh
adb install -r android-agent/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.dragonnest.agent/.AgentSettingsActivity
```

One APK holds both the Compose care UI and the long-lived Device Agent
(foreground service, gRPC stream, telemetry, reconnect backoff). Point it at
`10.0.2.2:50051` from the emulator, or the Brain host's LAN address from a
physical phone. Details: [android-agent/README.md](android-agent/README.md).

### 2. Connect your phone

1. Open `http://<brain-host>:8080/` in a browser (your laptop, or the phone
   itself — it installs as a PWA via the **Install** button).
2. Select **Show QR code**. If the Brain is reachable on more than one network,
   choose the one your phone is on.
3. On the phone, open the DragonNest Android app and select
   **Scan enrollment QR**, then confirm the address.
4. The card flips to **Your DragonNest is ready**. You're paired.

The QR carries a bootstrap credential that expires in five minutes and binds to
the **first** device that registers with it. The Brain immediately swaps it for
a device-specific reconnect credential, which the app stores in Android
Keystore. Unused codes can be cancelled and expire on their own.

> No phone handy? Copy the credential from the QR payload and claim it with a
> software agent instead: `run_agent.py --device-id phone-01 --enrollment-token <credential>`.

### 3. Set your preferences

Under **Your preferences**, three choices shape every request:

| Field | Options | Effect |
|---|---|---|
| Name | free text | Names the durable personal profile |
| What matters most? | Balanced · Fast answers · Keep it on this device · Best quality | Routing preference — "keep it on this device" pins execution to your phone |
| Answer style | Balanced · Concise · Detailed | Behavior steering; the Brain picks a runtime vector, a baked profile, or a prompt profile depending on what the chosen device supports |

Saved preferences live in the Brain's profile store and apply to every task
that originates from your device — no need to restate them.

### 4. Ask

Type into **Ask DragonNest** and send. The **Response** card shows the state
and then the answer. Behind the scenes the Brain classifies the request, checks
which devices are healthy and warm, prefers your own device when it's eligible,
and falls back to a more capable one when it isn't. If a device drops
mid-request, the task is retried once on the next compatible device — you just
see a slightly slower answer.

Curious where an answer actually ran? Ask your admin to open the same request
in `/admin` → **Selected Request** → **Why Brain chose this route**.

---

## Reference

### Commands

| Command | Purpose |
|---|---|
| `pytest -q` | Unit + integration tests |
| `scripts/demo_scenarios.py` | Full 7-scenario fabric demo, no hardware |
| `scripts/demo_mock.py` | Minimal routing walkthrough |
| `scripts/demo_recovery.py` | Failure and retry behavior |
| `scripts/demo_grpc.py` | Forced disconnect, parallel fanout, layer pipeline |
| `scripts/run_brain.py` | Start the Brain + dashboard |
| `scripts/run_agent.py` | Start a Device Agent |
| `scripts/submit_task.py` | Submit a task from the CLI |
| `scripts/check_artifacts.py` | Validate artifact paths and checksums |
| `scripts/hash_artifact.py` | Compute an artifact hash for the manifest |
| `scripts/probe_hardware.py` | Physical execution proof on real hardware |
| `scripts/build_android.sh` | Build the Android APK |
| `scripts/generate_proto.sh` | Regenerate protobuf bindings after editing `proto/` |

### Configuration

| File | Role |
|---|---|
| `configs/brain.yaml` | Brain defaults |
| `configs/dev-fabric.yaml` | Mock device inventory for dev and tests |
| `configs/hardware-fabric.yaml` | Real Snapdragon device inventory |
| `configs/demo-fleet.yaml` | Deterministic X Elite + S25 demo fleet |
| `configs/model-artifacts.yaml` | Artifact paths and checksums |
| `configs/artifact-catalog.yaml` | Per-device deployment state |
| `configs/behavior-profiles.yaml` | Behavior profiles and steering fallback policy |
| `configs/steering-vectors.yaml` | Steering vector registry |

### Layout

```text
DragonNest/
├── proto/dragonnest.proto      # device ⇄ Brain contract
├── configs/                    # fabric, artifact, behavior, steering config
├── scripts/                    # entry points, demos, hardware tooling
├── src/dragon_nest/
│   ├── classifier · planner · router · dispatch   # the routing core
│   ├── registry · telemetry · regimes             # device state and capability
│   ├── behavior · steering · scheduler            # behavior-aware deployment
│   ├── executors · runtime/                       # mock + QNN/Genie execution
│   ├── transport/                                 # gRPC brain/agent, HTTP devices
│   ├── dashboard.py · web/                        # FastAPI + user/admin/regimes UI
│   └── enrollment · profiles · provisioning
├── android-agent/              # Android app: care UI + Device Agent, one APK
├── AskQuery/ · Image2Audio/    # on-device OCR / TTS / ASR pipelines
├── docs/                       # architecture, runbooks, hardware evidence
└── tests/
```

### Documentation map

| Doc | Read it for |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Component design and data flow |
| [docs/SPEC.md](docs/SPEC.md) | Complete behavioral specification |
| [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md) | Scripted physical two-device demo |
| [docs/SNAPROUTER.md](docs/SNAPROUTER.md) | Routing and scheduling internals |
| [docs/BEHAVIOR_SCHEDULER.md](docs/BEHAVIOR_SCHEDULER.md) | Behavior profiles, cost model, provisioning |
| [docs/HARDWARE_CONTRACT.md](docs/HARDWARE_CONTRACT.md) | What a device must implement and prove for real QNN/Genie execution |
| [docs/HARDWARE_AUDIT.md](docs/HARDWARE_AUDIT.md) · [HARDWARE_BENCHMARKS.md](docs/HARDWARE_BENCHMARKS.md) | Evidence-classified device audit and measurements |
| [docs/STEERING_VECTOR_PROVENANCE.md](docs/STEERING_VECTOR_PROVENANCE.md) | Where each steering vector came from |
| [android-agent/README.md](android-agent/README.md) | APK build, QNN/Genie runtime builds |

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: uvicorn` | You ran system `python`. Use `.venv\Scripts\python.exe`. |
| Dashboard loads, no devices listed | No Agent is running, or its `--brain` address / `--enrollment-token` doesn't match the Brain's. |
| Phone can't reach the Brain | Brain must bind `0.0.0.0` (the default), phone must be on the same network, and the host firewall must allow ports 8080 and 50051. |
| QR scan fails or expires | Sessions last five minutes and bind to the first device that registers. Cancel and generate a new one. |
| Task rejected, no eligible device | Check **Advanced → Behavior Routing** on `/admin` for the rejection reason (memory fit, capability, compatibility key). |
| Agent starts but advertises nothing | Its artifacts failed path/checksum validation. Run `scripts/check_artifacts.py`. |
| Stale devices after a restart | Delete `local/dragonnest-state.sqlite3` to reset the fabric to empty. |

# DragonNest

DragonNest combines the best parts of the two hackathon repos in this
workspace:

- `PersonaCare`: a concrete on-device Snapdragon demo pipeline for OCR,
  structured extraction, email drafting, speech, and local QAIRT/QNN execution.
- `PersonaCare-Steering-Research`: tested activation steering, ONNX/QNN
  steering validation, and layer split-compute experiments.

The goal is one unified Snapdragon AI fabric:

> A trusted set of Snapdragon devices can route tasks, run multimodal pipelines,
> steer model behavior, split work across devices, and reroute around thermal,
> battery, memory, and availability constraints.

## What This Repo Is

DragonNest is the umbrella product/repo for:

- **SnapRouter control plane**: device/model routing, health-aware scheduling,
  reroute, data parallelism, layer-pipeline planning, and route explainability.
- **PersonaCare demo app**: doctor-note/photo/audio workflows that prove the
  product value in a user-facing scenario.
- **Steering research path**: activation-steering metadata, policy checks, and
  QNN/Genie runtime integration.
- **Split-compute path**: pipeline-stage model execution with hidden-state
  boundary tensors.

The first version is intentionally lightweight: it includes a runnable control
core, mock execution, hardware runtime adapters, and extension points for gRPC
agents, a FastAPI dashboard, and live cross-device transport.

## Current Status

Implemented in this scaffold:

- Rule-based task classifier.
- Steering vector registry and compatibility checks.
- Execution planner for `single`, `data_parallel`, and `layer_pipeline`.
- Health-aware deterministic router.
- Mock executor for single, shard, and pipeline-stage execution.
- Artifact manifests with path and checksum validation.
- Ported PersonaCare QAIRT/QNN and Genie runners.
- Runtime-aware Genie, QNN graph, and QNN layer-pipeline executor adapters.
- Opt-in Qualcomm AI Hub device-lab adapter and live steering/boundary proof
  for remote Snapdragon 8 Elite and Snapdragon X Elite hardware.
- Registry lifecycle for healthy, degraded, stale, and offline devices.
- Stable task IDs, unique attempts, result fencing, cancellation state, and one
  offline fallback retry.
- Persistent bidirectional gRPC registration/control streams with health,
  task assignment, results, reconnect, and graceful shutdown.
- Synchronous gRPC single-device submission with disconnect fallback.
- Concurrent remote data-parallel shard dispatch, per-shard retry, and stable
  deterministic reduction.
- Origin-bound private routing and first-success replica races with loser
  cancellation and late-result fencing.
- Remote layer-pipeline stage dispatch with checksummed tensor boundaries and
  compatible-stage fallback.
- Platform telemetry abstraction with live memory/CPU collection, measured
  Brain RTT, active-task/model-warm reporting, and immediate network-change
  refresh.
- Persistent device simulation overlays for offline, thermal, battery, load,
  memory, reachability, and RTT conditions.
- Production-mode mTLS transport with verified client-certificate fingerprint
  tracking, revocation, and certificate rotation on reconnect.
- Buildable Android Agent APK with a foreground service, persistent generated
  gRPC client, mock task/shard/pipeline executor, cancellation, reconnect
  backoff, network callbacks, Keystore enrollment storage, live telemetry,
  graceful shutdown, settings UI, and simulation controls.
- FastAPI dashboard/API with health, simulations, task submission, steering,
  route traces, progress, results, and live events.
- Behavior-aware deployment scheduler: behavior profiles with explicit
  steering realizations (runtime vector / baked profile / prompt profile /
  none) and fallback policies, an artifact catalog with per-device deployment
  states, feasibility filtering with projected-memory fits, an explainable
  deterministic cost model, a provisioning state machine behind a mock AI Hub
  adapter, and dashboard panels for candidates, rejections, and provisioning.
  See [docs/BEHAVIOR_SCHEDULER.md](docs/BEHAVIOR_SCHEDULER.md).
- Native Windows memory/battery telemetry probes for Snapdragon X Elite
  laptops.
- Deterministic demo fleet (X Elite laptop + Galaxy S25 Ultra) and scenario
  runner covering warm preference, behavior locality, thermal reroute, memory
  rejection, steering fallback, disconnect recovery, and provisioning. See
  [docs/DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md).
- Unit and integration tests for routing, runtimes, disconnects, and recovery.

Not implemented yet:

- On-target smoke testing of the DragonNest runtime adapters.
- On-target validation of real QNN layer boundaries over the live transport.
- Actual on-device QNN layer-pipeline execution on a Snapdragon target. The
  Android Agent now has a direct QAIRT 2.48 Genie JNI bridge for verified S25
  bundles, plus artifact validation and dynamic capability registration. A
  matching licensed SDK, S25-targeted model bundle, and physical smoke test are
  still required before it is a release claim.

## Repository Layout

```text
DragonNest/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- proto/
|   `-- dragonnest.proto
|-- configs/
|   |-- brain.yaml
|   |-- dev-fabric.yaml
|   |-- model-artifacts.yaml
|   `-- steering-vectors.yaml
|-- android-agent/                     # Android foreground-agent package
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- MIGRATION_PLAN.md
|   |-- SNAPROUTER.md
|   `-- SPEC.md
|-- scripts/
|   |-- check_artifacts.py
|   |-- demo_mock.py
|   |-- demo_recovery.py
|   |-- demo_grpc.py
|   `-- hash_artifact.py
|-- src/
|   `-- dragon_nest/
|       |-- __init__.py
|       |-- artifacts.py
|       |-- classifier.py
|       |-- dispatch.py
|       |-- executors.py
|       |-- models.py
|       |-- planner.py
|       |-- registry.py
|       |-- router.py
|       |-- steering.py
|       |-- tasks.py
|       |-- runtime/
|       `-- transport/
`-- tests/
```

## Quick Start

```bash
cd DragonNest
.venv/bin/python -m pytest -q
.venv/bin/python scripts/demo_mock.py
.venv/bin/python scripts/demo_recovery.py
.venv/bin/python scripts/demo_grpc.py
.venv/bin/python scripts/demo_scenarios.py
.venv/bin/python scripts/check_artifacts.py
```

Create the environment and install for editable development first:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

On Windows PowerShell, use the venv executables under `.venv\Scripts`:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
```

## gRPC Brain And Agents

With the virtual environment active, start the Brain and two Agents in separate
terminals:

```bash
python scripts/run_brain.py
python scripts/run_agent.py --device-id phone-01
python scripts/run_agent.py --device-id pc-01
python scripts/submit_task.py "Compare both options and recommend one."
```

The dashboard is available at `http://127.0.0.1:8080` by default.

The Agent reconnects with exponential backoff. A stream loss fails active task,
shard, or pipeline-stage attempts, and the Brain retries each once on the next
eligible compatible device. Run `python scripts/demo_grpc.py` for a
self-contained forced-disconnect, parallel-fanout, and layer-pipeline demo.

Private routing requires the originating device explicitly:

```bash
python scripts/submit_task.py \
  --preferred-mode private \
  --origin-device-id phone-01 \
  "Rewrite this private note."
```

Run the same shard on two devices and accept the first successful result:

```bash
python scripts/submit_task.py \
  --execution-mode data_parallel \
  --reducer first_success \
  "Answer this low-latency request."
```

Agent-side simulation flags feed the same telemetry contract as real platform
measurements:

```bash
python scripts/run_agent.py --device-id phone-01 \
  --simulate-thermal 0.95 --simulate-load 0.90 --simulate-rtt 180
```

Production mode requires mutual TLS. The Brain verifies that each registration
fingerprint matches the certificate authenticated by gRPC:

```bash
python scripts/run_brain.py --production \
  --tls-certificate certs/brain.crt --tls-key certs/brain.key \
  --tls-client-ca certs/device-ca.crt
python scripts/run_agent.py --device-id phone-01 \
  --tls-ca certs/device-ca.crt \
  --tls-certificate certs/phone-01.crt --tls-key certs/phone-01.key
```

Regenerate checked-in protobuf bindings after editing the contract:

```bash
scripts/generate_proto.sh
```

## HTTP Endpoint Devices

The Brain can also call trusted inference endpoints over HTTP. This control
plane is disabled by default. Enable it with an admin token and explicit
network policy:

```bash
export DRAGONNEST_HTTP_ENDPOINT_ADMIN_TOKEN="replace-me"
export EDGE_API_TOKEN="endpoint-bearer-token"
python scripts/run_brain.py \
  --enable-http-endpoints \
  --http-endpoint-allow-host edge-box.local
```

Use the admin dashboard to register the endpoint. Its credential field is an
environment variable name such as `EDGE_API_TOKEN`; secret values are never
stored in SQLite. Literal IP endpoint URLs must fall within a configured
`--http-endpoint-allow-cidr`. DNS hostnames must be explicitly listed with
`--http-endpoint-allow-host`.

Endpoint configuration persists in the Brain state database and is restored
on restart. Profile context is denied by default and must be enabled for each
trusted endpoint. The endpoint implements this JSON API:

```text
GET  /health
GET  /info
POST /execute
POST /execute_shard
POST /execute_pipeline_stage
POST /cancel
```

The Brain sends the same task, steering, timeout, shard, and boundary fields
used by the gRPC transport. `/info` metadata is validated before it becomes a
routing capability. Registration, discovery, and removal require the endpoint
admin bearer token.

## Android APK

With JDK 17 and Android SDK 35 available, build and test the Android Agent:

```bash
scripts/build_android.sh
```

All build caches default to `/tmp/dragonnest-toolchain`. The APK is produced at
`android-agent/app/build/outputs/apk/debug/app-debug.apk`. Install it with:

```bash
adb install -r android-agent/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.dragonnest.agent/.AgentSettingsActivity
```

Use `10.0.2.2:50051` from the Android emulator or the Brain host's LAN address
from a physical device. See [android-agent/README.md](android-agent/README.md)
for configuration and verification details.

For QR enrollment, open the dashboard, select **Add device**, enter the Brain
host's LAN address and gRPC port, and generate the code. In the APK, select
**Scan enrollment QR** and confirm the address. The five-minute bootstrap
credential is bound to the first device ID that registers; the Brain returns a
device-specific reconnect credential which the APK replaces in Android
Keystore. Unused sessions can be cancelled and expire automatically. This is a
development-mode onboarding path; production QR enrollment must issue an mTLS
client certificate instead of a token credential.

The Add Device form also creates a durable personal profile containing the
person and device names, routing preference, and optional steering
vector/alpha/position defaults. The claimed device is associated with that
profile in `local/dragonnest-state.sqlite3`. Tasks submitted with that device as
their origin inherit the saved preference and steering policy unless profile
steering is disabled or an explicit steering request is supplied.

At registration, Android Agents also report automatic static inventory: device
manufacturer/model, Android/API version, SoC where Android exposes it, CPU
ABIs/core count, storage, and QNN/NPU probe status. Live memory, thermal, and
load remain heartbeat telemetry. For normal tasks, the Brain now prefers an
eligible origin device before routing to a more capable remote fallback.

## Product Threads

DragonNest keeps three product threads under one roof:

1. **Multimodal care demo**
   - OCR -> structured extraction -> drafted email -> TTS/audio.
   - Based on the practical `PersonaCare` pipeline.

2. **Trusted Snapdragon fabric**
   - Brain + Device Agents.
   - Health-aware routing and reroute.
   - Single-device, data-parallel, and layer-pipeline execution.

3. **Behavior control**
   - Vector steering metadata and policy.
   - Runtime steering where supported.
   - Compiled steering variants where runtime steering is unavailable.

## Source Repo Relationship

This repo should not blindly vendor the two source repos. Use them as follows:

- Keep the ported `PersonaCare` QNN and Genie runners aligned with fixes proven
  against real hardware.
- Port tested steering/split-compute modules from `PersonaCare-Steering-Research`
  behind DragonNest interfaces.
- Keep generated artifacts, downloaded model binaries, and hardware-specific
  cache files out of git.

## Hardware Runtime Configuration

The manifest at `configs/model-artifacts.yaml` describes the validated
PersonaCare Genie bundle and split-compute QNN artifacts. Set the referenced
paths and checksums before starting an Agent:

```powershell
$env:GENIE_DIR = "C:\path\to\qwen3_4b-genie-w4a16-qualcomm_snapdragon_x_elite"
$env:QWEN3_4B_GENIE_SHA256_TREE = "sha256-tree:<bundle-tree-sha256>"
$env:QWEN3_SPLIT_PART_A_QNN = "C:\path\to\part_a.bin"
$env:QWEN3_SPLIT_PART_A_SHA256 = "sha256:<file-sha256>"
$env:QWEN3_SPLIT_PART_B_QNN = "C:\path\to\part_b.bin"
$env:QWEN3_SPLIT_PART_B_SHA256 = "sha256:<file-sha256>"
python scripts/hash_artifact.py $env:GENIE_DIR
python scripts/hash_artifact.py $env:QWEN3_SPLIT_PART_A_QNN
python scripts/hash_artifact.py $env:QWEN3_SPLIT_PART_B_QNN
python scripts/check_artifacts.py
```

An Agent must advertise only artifacts reported as `READY`. The checked-in
manifest does not imply that the external model binaries are present.

## Qualcomm AI Hub Validation

AI Hub can validate QNN compilation, inference, profiling, steering numerics,
and sequential split-boundary compatibility on remote Snapdragon hardware. It
does not emulate an Android Agent, provide live device-to-device transport, or
replace APK JNI integration with the QNN SDK.

Use a dedicated environment because `qai-hub==0.53.0` requires protobuf 6 while
the DragonNest gRPC environment uses protobuf 7. Keep its home and cache outside
the repository:

```bash
python3 -m venv /tmp/dragonnest-aihub-venv
HOME=/tmp/dragonnest-aihub-home \
  PIP_CACHE_DIR=/tmp/dragonnest-aihub-cache \
  /tmp/dragonnest-aihub-venv/bin/pip install -r requirements-ai-hub.txt
HOME=/tmp/dragonnest-aihub-home \
  /tmp/dragonnest-aihub-venv/bin/qai-hub configure --api_token '<token>'
```

The validation command is dry-run by default. `--submit` explicitly consumes
AI Hub quota. It compiles a steering stage for a phone-class target, feeds its
real output boundary into a second stage on a PC-class target, profiles both,
and writes a secret-free proof record under `/tmp`:

```bash
HOME=/tmp/dragonnest-aihub-home \
  /tmp/dragonnest-aihub-venv/bin/python scripts/validate_ai_hub.py
HOME=/tmp/dragonnest-aihub-home \
  /tmp/dragonnest-aihub-venv/bin/python scripts/validate_ai_hub.py --submit
```

The first submitted run prints linked model IDs. Pass them back through
`--stage-0-model-id` and `--stage-1-model-id` to rerun inference/profile checks
without recompiling.

See [docs/MIGRATION_PLAN.md](docs/MIGRATION_PLAN.md) for the staged merge plan.
See [docs/SPEC.md](docs/SPEC.md) for the complete DragonNest coding-agent
specification.

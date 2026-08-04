# DragonNest Architecture

DragonNest has one control plane and several execution backends.

## Control Plane

The SnapRouter Brain control plane consists of:

- Device registry.
- Health-state cache.
- Task classifier.
- Execution planner.
- Deterministic router.
- Dispatch/retry manager.
- Result reducer.
- Steering vector registry.
- Dashboard API.

The registry, classifier, planner, router, dispatch/retry manager, task state,
artifact registry, and executor adapters are implemented as transport-neutral
Python modules. Persistent gRPC streams and the FastAPI dashboard expose these
modules rather than owning duplicate lifecycle state.

### Device And Task Authority

`DeviceRegistry` owns stream/heartbeat health transitions and routing
eligibility. `TaskStore` owns stable task IDs, per-assignment attempt IDs,
accepted results, and stale late-result history. `DispatchManager` performs at
most one retry after `DEVICE_OFFLINE` and fails with
`NO_ELIGIBLE_FALLBACK` when no candidate remains.

Agents execute best-effort attempts. Only `TaskStore.record_result()` can
accept a result, which fences responses from offline or superseded attempts.

### gRPC Transport

`BrainService.Connect` requires registration as the first stream message, then
accepts health, result, and shutdown messages while sending assignments and
heartbeat acknowledgements. `DeviceAgent` reconnects with exponential backoff
and advertises real models only after artifact validation.

Development mode accepts either the configured shared token or QR bootstrap.
The dashboard creates a short-lived bootstrap secret, the first registration
binds it to one device ID, and the Brain returns a device-specific credential
for reconnects. Android validates the QR and stores the replacement credential
with an Android Keystore key. Production mode requires mutual TLS, matches the
Agent-reported SHA-256 certificate fingerprint to the authenticated peer
certificate, tracks rotation on reconnect, and can revoke a fingerprint and
offline every associated session. Production QR onboarding must replace the
development credential exchange with a Keystore-backed CSR and signed client
certificate.

`ProfileStore` persists personal profiles and device associations in SQLite.
The profile owns the human-entered name, routing preference, and steering
defaults; the Agent remains authoritative for hardware inventory and live
telemetry. When a QR session is claimed, the Brain associates its profile with
the generated device ID and uses the saved device name in the registry. A task
originating from that device inherits profile steering unless the request
disables profile steering or supplies an explicit steering spec.

Agent heartbeats come from an injectable platform telemetry interface. The
default source reports host battery, thermal, memory, and CPU data when
available, uses explicit unknown values otherwise, measures gRPC heartbeat RTT,
and includes active task IDs and warm model IDs. Platform network callbacks can
force an immediate refresh instead of waiting for the normal heartbeat period.

`SubmitTask` supports remote single-device execution and concurrent shard
fanout. Each shard has an internal lifecycle record and retry fence while every
wire message retains the stable parent task ID. Layer pipelines execute stages
sequentially, validate SHA-256 boundary payloads, and retry a failed stage only
on a device advertising the same pipeline and layer range. The mock path is
fully runnable; real QNN boundary transport still requires target-hardware
validation.

Private requests filter the eligible set to their declared origin device before
routing. A `first_success` data-parallel request sends one shard to two replicas,
accepts the first valid result, cancels remaining attempts, and records any late
loser output as stale.

### HTTP Dashboard

The FastAPI application reads directly from `DeviceRegistry`, `TaskStore`, and
Brain route metadata. It provides task submission, device simulations, steering
profiles, parent-task progress, results, and a merged registry/task event log.
The browser polls once per second; no separate dashboard state store exists.

### Android Agent

`android-agent` generates Android-lite protobuf and gRPC bindings from the same
canonical contract as the Python services. Its foreground service maintains the
bidirectional stream, reports platform telemetry, dispatches task, shard, and
pipeline commands to `AndroidTaskExecutor`, returns normalized metrics/results,
and applies Brain cancellation by attempt ID.

The packaged MVP executor is deterministic mock inference. The executor
interface is the Android integration point for Qualcomm QNN/Genie libraries;
the Python runtime adapters and artifact manifests remain the validated host
reference until those vendor libraries and model binaries are available on the
target Android device.

## Execution Modes

### Single Device

One task goes to one device/model. This is the default path and the simplest
demo mode.

### Data Parallel

The planner splits a compound task into independent shards. Shards can run on
different devices, then a reducer combines outputs.

Initial reducers:

- `concat`
- `first_success`
- `mock_synthesis`

Future reducers can call a real LLM on the best eligible device.

### Layer Pipeline

A single model is split into layer ranges. Each device advertises model segment
capabilities. A valid pipeline must cover contiguous layers without gaps.

This is based on the local split-compute research pattern:

```text
Part A: input_ids -> hidden boundary
Part B: hidden boundary -> logits/output
```

The portable MVP sends mock boundary tensors over the live gRPC stream.
`QnnPipelineExecutor` and the Agent stage adapter can serialize real QNN output
arrays into the same boundary contract using validated manifests. The remaining
gate is running that path with compiled split artifacts on target devices.

### Vector Steering

Steering metadata is first-class in routing and execution. The policy checks:

- vector ID
- model family
- target layer
- alpha range
- positions mode
- local/remote vector sharing

The eventual runtime operation is:

```text
steered_hidden = hidden + alpha * mask * steering_vector
```

The current scaffold passes steering metadata through routing and execution.
QNN and Genie runners are ported; injecting a steering vector into a deployed
runtime graph remains model-specific work.

## Source Repo Inheritance

From `PersonaCare`:

- QAIRT/QNN runner pattern.
- EasyOCR, MeloTTS, Whisper, and Genie orchestration lessons.
- Doctor-note end-to-end demo.
- Chunking and local/offline execution practices.

From `PersonaCare-Steering-Research`:

- Tested steering injection.
- ONNX export and backend comparison.
- QNN/AI Hub proof workflow.
- Layer split-compute proof.
- Reproducibility and result artifact discipline.

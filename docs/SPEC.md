# SnapRouter Coding-Agent Spec

SnapRouter is a hackathon-ready prototype for routing AI work across a trusted
fabric of Snapdragon devices.

The Brain runs on a Snapdragon PC and owns the control plane. One or more
Snapdragon devices run Device Agents. Agents register with the Brain over a
persistent gRPC bidirectional stream, report capability and health state, and
receive inference assignments.

This version expands the original single-device router into a small execution
planner that supports:

- Single-device routing.
- Data parallelism for independent shards, batch fanout, hedged replicas, and
  map-reduce style compound tasks.
- Layer parallelism, also called pipeline split execution, for one model split
  across devices by layer ranges.
- Vector steering for runtime behavior control through activation steering
  metadata and steering tensors.

The core demo value proposition is:

> Adding trusted Snapdragon devices adds usable model capability, execution
> capacity, parallel execution patterns, and controlled model behavior, not only
> raw throughput.

## 0. Workspace References

The design should reuse patterns already present under this workspace:

- `PersonaCare/qnn_runner.py`
  is a working local QAIRT/QNN wrapper for `.dlc` graphs and QNN context
  binaries. It includes single and batched execution, CPU/HTP backend
  selection, timeouts, retry handling, profiling, and tensor I/O.
- `PersonaCare/genie_runner.py`
  is a working Snapdragon Genie wrapper for Qwen3-4B text generation through
  `genie-t2t-run`, including prompt-file handling, timeout/error propagation,
  and response parsing.
- `PersonaCare/README.md`
  documents the validated external artifacts and device setup for EasyOCR,
  MeloTTS, Whisper, and Qwen3-4B Genie execution. The large model artifacts are
  downloaded separately and must remain configurable rather than committed.
- `PersonaCare-Steering-Research/src/steering_poc/inject.py`
  demonstrates activation steering as:
  `steered_hidden = hidden + alpha * mask * steering_vector`.
- `PersonaCare-Steering-Research/src/steering_poc/export_onnx.py`
  exports the steering operation as NPU-friendly `Mul` and `Add` graphs with
  runtime inputs `hidden`, `steering`, `alpha`, and `mask`.
- `PersonaCare-Steering-Research/src/split_compute/split_model.py`
  demonstrates layer split execution where Part A emits a residual hidden-state
  tensor and Part B consumes that tensor.
- `PersonaCare-Steering-Research/src/split_compute/submit_split.py`
  compiles and runs Part A on phone-class Snapdragon silicon and Part B on
  laptop-class Snapdragon silicon through AI Hub. It documents the current
  limit: the downloaded/uploaded boundary substitutes for live transport.
- `PersonaCare-Steering-Research/docs/results/split/`
  records the split metadata, local parity result, and successful AI Hub job
  evidence for the two-device Qwen3-0.6B experiment.
- `PersonaCare-Steering-Research/docs/results/qwen3_1_7b_patch/README.md`
  records successful Qwen3-1.7B GenieX QAIRT export, compile, link, and NPU
  profiling of multi-part context binaries.
- `DragonNest/scripts/validate_ai_hub.py` and
  `DragonNest/docs/results/ai_hub_device_lab_proof.json` provide an opt-in,
  quota-gated DragonNest proof on hosted Snapdragon 8 Elite and Snapdragon X
  Elite hardware. The proof compiles and profiles both QNN stages, validates
  runtime steering, and feeds the first device's returned activation boundary
  into the second device with FP16-appropriate numerical checks.

These are validated source implementations and evidence, not runtime imports
from DragonNest. Port or adapt them behind DragonNest executor interfaces while
keeping the source repositories independently runnable. Mock executors remain
the portable MVP path. Hardware-specific SDK locations and downloaded model
artifacts must be supplied through configuration.

## 1. Objective

Build a prototype named `SnapRouter`.

The Brain:

- Receives a text task.
- Classifies it into a small routing-relevant task profile.
- Builds an execution plan.
- Selects the best eligible device, model, and execution mode.
- Dispatches the task.
- Displays the decision, health inputs, plan, execution trace, and result.
- Reroutes when a selected device becomes unhealthy, overloaded, unavailable,
  or simulated-constrained.

Device Agents:

- Register with the Brain.
- Advertise hardware, model, parallel execution, and vector steering
  capabilities.
- Send heartbeats and health updates.
- Execute assigned tasks or task shards.
- Return task results, partial results, boundary tensors, and execution events.

## 2. Scope

### 2.1 MVP Scope

The MVP must implement:

- Persistent bidirectional gRPC registration and control streams.
- Device health tracking with stale/offline transitions.
- Rule-based task classification.
- Deterministic routing with explainable scoring.
- Single-device execution through a mock executor.
- Data-parallel execution for compound or batch-style tasks through mock
  shard executors.
- Vector steering as first-class routing and execution metadata, with mock
  behavior and a QNN adapter based on the validated PersonaCare runner.
- Layer-parallel execution as a planned execution mode with a mock pipeline
  executor. Real live hidden-state transport can remain an extension point.
- Dashboard/API panels showing devices, health, execution mode, route trace,
  steering settings, shard/stage progress, results, and reroute events.
- Unit tests covering registration, health timeout, routing, data-parallel
  planning, steering eligibility, layer-pipeline planning, and reroute behavior.

### 2.2 Non-Goals

Do not implement these in the MVP:

- Tensor parallelism within a single layer.
- Distributed attention or distributed matrix multiplication.
- Shared KV cache across devices.
- Real peer-to-peer discovery.
- Device-to-device execution without the Brain as coordinator.
- Leader election or distributed control plane.
- Full production authentication or PKI provisioning.
- Model training or fine-tuning.
- A mandatory real NPU integration for the first runnable demo.

Layer parallelism is in scope as a planned and mockable execution mode. Tensor
parallelism remains out of scope.

## 3. Required Architecture

```text
+-------------------------------------------------------------+
| Snapdragon PC: Brain                                         |
|                                                             |
|  gRPC Server                                                |
|  +-- Device Registry                                        |
|  +-- Health-State Cache                                     |
|  +-- Task Classifier                                        |
|  +-- Execution Planner                                      |
|  |   +-- Single-Device Planner                              |
|  |   +-- Data-Parallel Planner                              |
|  |   +-- Layer-Pipeline Planner                             |
|  |   +-- Steering Planner                                   |
|  +-- Deterministic Router                                   |
|  +-- Dispatch / Retry / Reroute Manager                     |
|  +-- Result Reducer                                         |
|  +-- Steering Vector Registry                               |
|  +-- HTTP Dashboard API                                     |
|  +-- Dashboard Event Log                                    |
+-----------------------------+-------------------------------+
                              |
                    persistent gRPC stream
                              |
          +-------------------+-------------------+
          |                                       |
+---------v----------------+          +-----------v-------------+
| Device Agent: Phone      |          | Device Agent: PC / Dev   |
|                          |          |                          |
| - registration           |          | - registration           |
| - capability manifest    |          | - capability manifest    |
| - health heartbeat       |          | - health heartbeat       |
| - task executor          |          | - task executor          |
| - shard executor         |          | - shard executor         |
| - pipeline stage stub    |          | - pipeline stage stub    |
| - steering support       |          | - steering support       |
| - result reporting       |          | - result reporting       |
+--------------------------+          +-------------------------+
```

The split between semantic classification and deterministic routing is
intentional. The classifier interprets the request. The router uses live
telemetry, device policy, capability manifests, and explicit execution-mode
rules to make the final resource decision.

## 4. Execution Modes

### 4.1 Single-Device Mode

Single-device mode is the default path. The Brain chooses one device and one
model, dispatches one `ExecuteTask`, and receives one `TaskResult`.

Use this mode when:

- The task is short or not compound.
- Only one device has the required model.
- Privacy mode excludes remote devices.
- The request needs one coherent model context.

### 4.2 Data-Parallel Mode

Data-parallel mode splits a request into independent shards that can run on
multiple devices or multiple model replicas.

Data parallelism is supported as an MVP-plus execution mode: the Brain may
split a compound task into independent shards, assign shards to multiple
eligible devices, and then merge the results.

Supported MVP patterns:

- `batch_fanout`: each item in a batch is routed as an independent shard.
- `map_reduce`: a compound text task is decomposed into independent subtasks,
  then reduced into one final answer.
- `replica_race`: the same shard is sent to more than one device and the first
  valid result wins. This is useful for latency demos.
- `replica_vote`: the same shard is sent to multiple devices and a reducer
  chooses or merges the results. This is useful for quality demos.

The first implementation can use rule-based task splitting. Examples:

- "Summarize these three sections" becomes three summary shards plus one
  reducer.
- "Extract action items from each note" becomes one extraction shard per note.
- "Compare A and B, then recommend one" can become parallel analysis shards
  for A, B, and comparison, followed by a reducer.

Data-parallel mode must preserve privacy rules. A `device_only` or `private`
task cannot be split across remote devices unless the policy explicitly allows
the trusted fabric for that data.

Result reduction options:

- `concat`: join shard results in stable shard order.
- `first_success`: return the first successful replica result.
- `majority_vote`: choose the most common normalized answer.
- `synthesis`: run a final reducer task on the best eligible device.
- `mock_synthesis`: deterministic synthetic reducer for the MVP.

Reroute behavior:

- If one shard fails and the task has a fallback device, retry that shard once.
- If a replica race has one successful result, cancel remaining replicas.
- If the reducer fails, reroute only the reducer when shard outputs are still
  available.
- If privacy mode changes or a device becomes offline, do not dispatch new
  shards to that device.

### 4.3 Layer-Parallel Mode

Layer-parallel mode runs one model as a sequence of pipeline stages across
devices. This is pipeline parallelism by layer range, not tensor parallelism.

The guiding local example is the split-compute experiment in
`PersonaCare-Steering-Research/src/split_compute/`, where:

- Part A runs embeddings and layers `[0, split)`.
- Part A emits a residual hidden-state tensor.
- Part B consumes that hidden-state tensor and runs layers `[split, N)` plus
  norm and LM head.
- The boundary tensor is the only cross-device model tensor.

The existing experiment is more than a design sketch: `split_model.py` and
`export_split.py` produce the two model halves, `verify_local.py` checks chained
ONNX parity, and `submit_split.py` has executed the halves on real phone- and
laptop-class Snapdragon devices. The committed evidence is under
`PersonaCare-Steering-Research/docs/results/split/`.

SnapRouter should support a mock layer pipeline in the MVP:

```text
Task input
  -> Stage 0: phone, qwen3-0.6b layers 0..13
  -> Boundary tensor: hidden [1, T, D] fp16/fp32
  -> Stage 1: pc, qwen3-0.6b layers 14..27 + head
  -> Final result
```

Eligibility requirements:

- All stages must advertise the same `model_family`.
- All stages must advertise the same tokenizer and model version.
- All stages must advertise compatible precision.
- All stages must use a fixed, agreed split point.
- Adjacent stages must agree on `boundary_schema`.
- Adjacent stages must agree on the tensor boundary format.
- Stage order must cover a contiguous layer range without overlap or gaps.
- All devices must satisfy task privacy requirements.
- Estimated boundary payload and network RTT must fit the latency tier.
- A layer pipeline cannot cross into an offline or stale device.

Portable MVP implementation:

- Provide `MockPipelineExecutor` that simulates each stage and returns synthetic
  boundary metadata.
- Provide `LayerPipelinePlan` and route explanations.
- Do not require real hidden-state tensors to cross devices for the demo.
- Preserve the boundary contract required by the existing split-compute code.

DragonNest runtime integration:

- Port the validated split wrappers behind `QnnPipelineExecutor`.
- Use fixed-shape prompt graphs first.
- Add cached decode graphs later.
- Transfer boundary tensors through the Brain or a controlled trusted channel.
- Record boundary payload size, transport latency, and per-stage latency.
- Recompute deterministic rotary embeddings in later stages when possible
  instead of transporting them.

The AI Hub experiments prove functional and numerical viability on real
silicon. The DragonNest-specific proof is implemented by
`scripts/validate_ai_hub.py`, with a secret-free result record under
`docs/results/ai_hub_device_lab_proof.json`. AI Hub does not provide live
device-to-device transport, streaming token-by-token decode, KV-cache handoff,
or end-to-end network latency; those remain DragonNest implementation
requirements.

Reroute behavior:

- If a stage fails before producing a boundary, retry that stage or restart the
  full pipeline once.
- If a later stage fails after receiving a boundary, retry from the last
  available boundary checkpoint when compatible fallback stages exist.
- If no compatible fallback stage exists, return a controlled
  `NO_PIPELINE_FALLBACK` error and show the failed stage in the dashboard.

### 4.4 Vector Steering Mode

Vector steering is a runtime behavior-control feature. It lets the Brain select
or attach steering metadata to an execution plan so a compatible model can
shift style, tone, verbosity, safety posture, or persona without fine-tuning.

Vector steering is a runtime policy attached to an execution request. The Brain
may select a steering profile, but only devices and models that advertise
compatible steering support are eligible.

The local steering operation is:

```text
steered_hidden = hidden + alpha * mask * steering_vector
```

MVP steering support:

- The Brain stores steering metadata in `SteeringVectorRegistry`.
- Device capabilities advertise whether a model supports steering.
- The router checks model compatibility before assigning a steering request.
- Mock executors include steering settings in deterministic output.
- Dashboard displays vector ID, target layer, alpha, positions, and whether the
  selected model supports runtime steering or a compiled steering variant.

Steering vector metadata:

- `vector_id`: stable identifier, for example `concise-vs-verbose-layer-7`.
- `model_family`: model family the vector was derived from.
- `model_revision`: optional exact revision.
- `hidden_size`: vector dimension.
- `target_layers`: supported layer indices.
- `default_layer`: preferred layer.
- `alpha_min` and `alpha_max`: allowed dose range.
- `positions`: `all`, `last`, or `mask`.
- `storage_uri`: local path or opaque URI. The MVP can omit raw vector loading.
- `safety_label`: human-readable policy label.

Steering eligibility:

- The selected model hidden size must match the vector hidden size.
- The selected model family must match the vector model family.
- The requested layer must be listed by the device and vector metadata.
- Alpha must be clamped or rejected according to policy.
- Private steering vectors must not be sent to remote devices unless the policy
  allows trusted-fabric sharing.

Layer-pipeline interaction:

- Steering can be applied only by the stage that owns the target layer.
- If the target layer is before the split, Stage 0 applies steering.
- If the target layer is after the split, Stage 1 applies steering.
- If no stage owns the target layer, the pipeline is ineligible.

Data-parallel interaction:

- All shards can use the same steering settings when the request asks for one
  global behavior.
- Shards may use different steering settings only if the planner explicitly
  emits per-shard `SteeringSpec` entries.
- Reducer tasks can use their own steering settings, often with lower alpha to
  avoid over-amplifying style.

## 5. Suggested Technology Stack

| Area | Suggested choice |
| --- | --- |
| Language | Python for the MVP |
| gRPC | `grpcio`, `grpcio-tools`, Protocol Buffers |
| Brain API | FastAPI |
| Dashboard | Minimal React/Vite app or server-rendered FastAPI HTML |
| State | In-memory dictionaries first; optional SQLite persistence |
| Streaming | gRPC bidirectional stream |
| Tests | pytest |
| Configuration | YAML and environment variables |
| Logging | Structured JSON logs |
| Device runtime abstraction | Python interface with mock, QNN, and steering-aware implementations |
| Parallel execution | Brain-coordinated dispatch with deterministic reducers |

## 6. Repository Layout

```text
snaprouter/
+-- README.md
+-- requirements.txt
+-- docker-compose.yml                  # optional; Brain/dashboard only
+-- proto/
|   +-- snaprouter.proto
+-- brain/
|   +-- main.py                         # FastAPI + gRPC startup
|   +-- grpc_server.py
|   +-- registry.py                     # registered devices and TTL state
|   +-- health_store.py                 # latest health per device
|   +-- classifier.py                   # rule / embedding classifier interface
|   +-- planner.py                      # chooses execution mode and plan shape
|   +-- router.py                       # deterministic scoring and selection
|   +-- dispatcher.py                   # assign, retry, reroute
|   +-- parallel_dispatcher.py          # shard, replica, pipeline orchestration
|   +-- reducer.py                      # data-parallel result reduction
|   +-- steering.py                     # steering vector registry and policy checks
|   +-- policy.py                       # eligibility and privacy policy
|   +-- models.py                       # Pydantic DTOs
|   +-- dashboard_api.py
|   +-- events.py
|   +-- config.py
+-- agent/
|   +-- main.py
|   +-- grpc_client.py
|   +-- telemetry.py
|   +-- executor.py                     # executor protocols
|   +-- mock_executor.py
|   +-- mock_parallel_executor.py
|   +-- mock_pipeline_executor.py
|   +-- steering_executor.py            # mock/runtime steering adapter
|   +-- qnn_executor.py                 # stub only
|   +-- qnn_pipeline_executor.py        # stub only
|   +-- config.py
+-- dashboard/
|   +-- ...                             # optional React app
+-- config/
|   +-- brain.yaml
|   +-- phone-agent.yaml
|   +-- pc-agent.yaml
|   +-- steering-vectors.yaml
+-- tests/
|   +-- test_classifier.py
|   +-- test_router.py
|   +-- test_health.py
|   +-- test_registration.py
|   +-- test_reroute.py
|   +-- test_data_parallel.py
|   +-- test_layer_pipeline.py
|   +-- test_steering_policy.py
+-- scripts/
    +-- run_brain.sh
    +-- run_mock_phone.sh
    +-- run_mock_pc.sh
    +-- demo_reroute.sh
    +-- demo_data_parallel.sh
    +-- demo_layer_pipeline.sh
    +-- demo_steering.sh
```

## 7. Device Registration and Connection Lifecycle

### 7.1 Connection Behavior

Each Device Agent creates one persistent bidirectional gRPC stream:

```proto
rpc Connect(stream DeviceToBrain) returns (stream BrainToDevice);
```

The first agent message must be `RegisterDevice`.

The Brain must respond with either:

- `RegistrationAccepted`
- `RegistrationRejected`

After acceptance:

- Agent sends a heartbeat every 2 seconds.
- Agent sends a full health update every 5 seconds or when a meaningful state
  change occurs.
- Brain considers a device `STALE` after 10 seconds without a heartbeat.
- Brain considers a device `OFFLINE` after 20 seconds without a heartbeat.
- Offline devices must never receive new tasks.
- In-flight tasks assigned to an offline device must be retried on an eligible
  fallback.
- Data-parallel shard tasks assigned to an offline device must retry only the
  affected shard when possible.
- Layer-pipeline tasks assigned to an offline stage must retry from the latest
  compatible checkpoint when possible, otherwise restart once or fail
  gracefully.

### 7.2 Device Identity

For the MVP, use a generated UUID persisted in the agent config directory:

```json
{
  "device_id": "phone-4fc2a0",
  "display_name": "Snapdragon Phone",
  "device_type": "phone",
  "platform": "android",
  "agent_version": "0.1.0"
}
```

Do not use MAC addresses or hardware serial numbers as the primary identity.

### 7.3 MVP Authentication

Support two modes:

- `dev_mode=true`: shared enrollment token in configuration or an expiring QR
  bootstrap credential that binds to one device ID and is exchanged for a
  device-specific reconnect credential.
- `dev_mode=false`: placeholder interface for mTLS/client certificates.

The Brain must reject registration if the enrollment token is missing or
invalid. QR bootstrap sessions must be single-device, expire automatically,
support operator cancellation, and never expose the persistent reconnect
credential in dashboard JSON or logs.

The Add Device workflow must persist a personal profile independently of live
Agent state. At minimum it stores the person name, friendly device name,
routing preference, steering vector ID, steering alpha, steering positions,
remote-vector policy, and notes. The QR session references the profile ID and
the Brain associates it with the generated device ID after the first successful
claim. Hardware inventory remains Agent-reported and is not editable profile
data.

For a task with an associated origin device, the Brain applies the personal
profile's routing and steering defaults when no explicit override is supplied.
Explicit task steering takes precedence, and callers must be able to disable
profile steering. Profile and device associations must survive Brain restarts.

### 7.4 Network Departure and Mid-Job Recovery

A device that loses its persistent gRPC stream, misses heartbeats, or reports
`reachable=false` must transition through the registry health lifecycle:

- `HEALTHY` / `DEGRADED`: eligible for routing.
- `STALE`: temporarily excluded unless no better fallback exists.
- `OFFLINE`: never eligible for new tasks.

If the gRPC stream closes unexpectedly, the Brain may mark the device `STALE`
immediately and `OFFLINE` after the configured heartbeat timeout. If the Agent
explicitly sends a shutdown or simulated-offline event, the Brain may mark it
`OFFLINE` immediately.

Offline devices must not receive new assignments.

For in-flight tasks:

- The Brain records a stable `task_id` for the user request.
- Every device assignment receives a unique `attempt_id`.
- If the selected device becomes `OFFLINE`, the current attempt is marked
  `DEVICE_OFFLINE`.
- The Dispatch Manager resubmits the task once to the best eligible fallback
  device.
- Late results from an offline or superseded attempt are recorded as stale and
  must not overwrite the accepted result.
- If no fallback exists, the task fails with a controlled
  `NO_ELIGIBLE_FALLBACK` result.

Agents should treat task execution as best-effort. The Brain owns final task
state.

## 8. gRPC Contract

Create `proto/snaprouter.proto`.

The coding agent may simplify field names, but must preserve the message
categories, lifecycle, and execution-mode concepts.

```proto
syntax = "proto3";

package snaprouter.v1;

service BrainControl {
  rpc Connect(stream DeviceToBrain) returns (stream BrainToDevice);
  rpc RouteTask(RouteTaskRequest) returns (RouteTaskResponse);
}

message DeviceToBrain {
  oneof payload {
    RegisterDevice register_device = 1;
    HealthUpdate health_update = 2;
    TaskResult task_result = 3;
    CommandAck command_ack = 4;
    PartialTaskResult partial_task_result = 5;
    PipelineStageResult pipeline_stage_result = 6;
    ExecutionEvent execution_event = 7;
  }
}

message BrainToDevice {
  oneof payload {
    RegistrationAccepted registration_accepted = 1;
    RegistrationRejected registration_rejected = 2;
    ExecuteTask execute_task = 3;
    ExecuteShard execute_shard = 4;
    ExecutePipelineStage execute_pipeline_stage = 5;
    CancelTask cancel_task = 6;
    HeartbeatAck heartbeat_ack = 7;
  }
}

message RegisterDevice {
  string device_id = 1;
  string display_name = 2;
  string device_type = 3;       // phone | pc | xr | automotive | simulator
  string platform = 4;          // android | windows | linux
  string agent_version = 5;
  string enrollment_token = 6;
  DeviceCapabilities capabilities = 7;
  string certificate_fingerprint = 8; // required outside dev mode
}

message DeviceCapabilities {
  repeated Accelerator accelerators = 1;
  repeated ModelCapability models = 2;
  repeated ParallelCapability parallel = 3;
  uint64 total_memory_mb = 4;
  bool supports_network_execution = 5;
}

message Accelerator {
  string name = 1;              // htp | gpu | cpu | npu
  bool available = 2;
  repeated string precisions = 3;
}

message ModelCapability {
  string model_id = 1;
  string model_family = 2;      // qwen3 | llama | mock
  string model_revision = 3;
  string role = 4;              // classifier | small_chat | large_reasoning | vision
  repeated string task_classes = 5;
  uint32 max_context_tokens = 6;
  bool warm = 7;
  float quality_score = 8;
  repeated SteeringCapability steering = 9;
  repeated ModelSegmentCapability segments = 10;
}

message ParallelCapability {
  string mode = 1;              // data_parallel | replica | layer_pipeline
  uint32 max_concurrent_tasks = 2;
  uint32 max_batch_shards = 3;
  bool supports_cancellation = 4;
}

message SteeringCapability {
  string steering_mode = 1;     // none | runtime_input | compiled_variant | mock
  repeated string vector_ids = 2;
  repeated uint32 supported_layers = 3;
  uint32 hidden_size = 4;
  float alpha_min = 5;
  float alpha_max = 6;
  repeated string positions = 7; // all | last | mask
}

message ModelSegmentCapability {
  string segment_id = 1;
  string pipeline_id = 2;
  string model_id = 3;
  string model_family = 4;
  string model_revision = 5;
  uint32 start_layer = 6;       // inclusive
  uint32 end_layer = 7;         // exclusive
  uint32 total_layers = 8;
  BoundarySchema input_boundary = 9;
  BoundarySchema output_boundary = 10;
  bool includes_embedding = 11;
  bool includes_lm_head = 12;
}

message BoundarySchema {
  string name = 1;              // input_ids | hidden | logits
  repeated uint32 shape = 2;    // use fixed MVP shapes when possible
  string dtype = 3;             // fp16 | fp32 | int32
  uint64 bytes_per_token = 4;
}

message HealthUpdate {
  string device_id = 1;
  int64 timestamp_ms = 2;
  float battery_pct = 3;        // -1 when not available
  bool charging = 4;
  float thermal_level = 5;      // normalized 0.0 cool to 1.0 critical
  float cpu_utilization = 6;    // normalized 0.0 to 1.0
  float accelerator_utilization = 7;
  uint64 available_memory_mb = 8;
  float network_rtt_ms = 9;
  bool reachable = 10;
  repeated string active_task_ids = 11;
  bool simulated_constraint = 12;
}

message ExecuteTask {
  string task_id = 1;
  string request_text = 2;
  TaskProfile profile = 3;
  string model_id = 4;
  uint32 timeout_ms = 5;
  SteeringSpec steering = 6;
  string attempt_id = 7;
}

message ExecuteShard {
  string task_id = 1;
  string shard_id = 2;
  string request_text = 3;
  TaskProfile profile = 4;
  string model_id = 5;
  uint32 timeout_ms = 6;
  SteeringSpec steering = 7;
  ReduceSpec reduce = 8;
  string attempt_id = 9;
}

message ExecutePipelineStage {
  string task_id = 1;
  string stage_id = 2;
  string request_text = 3;
  TaskProfile profile = 4;
  string model_id = 5;
  string segment_id = 6;
  uint32 stage_index = 7;
  BoundaryTensor input_boundary = 8;
  BoundarySchema output_schema = 9;
  SteeringSpec steering = 10;
  uint32 timeout_ms = 11;
  string attempt_id = 12;
}

message TaskProfile {
  string task_class = 1;
  string complexity = 2;        // low | medium | high
  string privacy_tier = 3;      // device_only | trusted_fabric
  string latency_tier = 4;      // realtime | interactive | background
  uint32 estimated_input_tokens = 5;
  uint32 estimated_output_tokens = 6;
  float confidence = 7;
  bool is_compound = 8;
  bool data_parallelizable = 9;
  bool layer_parallel_candidate = 10;
  bool steering_requested = 11;
}

message SteeringSpec {
  bool enabled = 1;
  string vector_id = 2;
  string model_family = 3;
  uint32 target_layer = 4;
  float alpha = 5;
  string positions = 6;         // all | last | mask
  string mask_policy = 7;       // none | prompt_last | generated_only | custom
  bool allow_remote_vector = 8;
}

message TaskResult {
  string task_id = 1;
  string device_id = 2;
  bool success = 3;
  string output_text = 4;
  string error_code = 5;
  string error_message = 6;
  uint32 execution_latency_ms = 7;
  float observed_thermal_delta = 8;
  string attempt_id = 9;
  ExecutionMetrics metrics = 10;
}

message PartialTaskResult {
  string task_id = 1;
  string shard_id = 2;
  string device_id = 3;
  bool success = 4;
  string output_text = 5;
  string error_code = 6;
  string error_message = 7;
  uint32 execution_latency_ms = 8;
  string attempt_id = 9;
  ExecutionMetrics metrics = 10;
}

message PipelineStageResult {
  string task_id = 1;
  string stage_id = 2;
  string device_id = 3;
  bool success = 4;
  BoundaryTensor output_boundary = 5;
  string output_text = 6;
  string error_code = 7;
  string error_message = 8;
  uint32 execution_latency_ms = 9;
  string attempt_id = 10;
  ExecutionMetrics metrics = 11;
}

message ExecutionMetrics {
  string model_id = 1;
  string model_version = 2;
  string runtime_name = 3;
  string runtime_version = 4;
  string accelerator = 5;
  uint32 execution_latency_ms = 6;
  string error_code = 7;
  string error_message = 8;
  int64 observed_memory_delta_mb = 9;
  float observed_thermal_delta = 10;
}

message BoundaryTensor {
  string tensor_id = 1;
  BoundarySchema schema = 2;
  string storage_uri = 3;       // MVP may use opaque mock URI
  uint64 byte_size = 4;
  string checksum = 5;
}

message RouteTaskRequest {
  string request_text = 1;
  string preferred_mode = 2;    // auto | fast | private | quality | parallel
  string execution_mode = 3;    // auto | single | data_parallel | layer_pipeline
  SteeringSpec steering = 4;
}

message RouteTaskResponse {
  string task_id = 1;
  TaskProfile profile = 2;
  ExecutionPlan plan = 3;
  RouteDecision decision = 4;
}

message ExecutionPlan {
  string task_id = 1;
  string execution_mode = 2;    // single | data_parallel | layer_pipeline
  repeated PlannedTask tasks = 3;
  repeated PipelineStage stages = 4;
  ReduceSpec reduce = 5;
  SteeringSpec steering = 6;
}

message PlannedTask {
  string shard_id = 1;
  string request_text = 2;
  repeated string candidate_device_ids = 3;
  string selected_device_id = 4;
  string selected_model_id = 5;
  repeated string fallback_device_ids = 6;
  SteeringSpec steering = 7;
}

message PipelineStage {
  string stage_id = 1;
  uint32 stage_index = 2;
  string selected_device_id = 3;
  string selected_model_id = 4;
  string segment_id = 5;
  uint32 start_layer = 6;
  uint32 end_layer = 7;
  repeated string fallback_device_ids = 8;
  BoundarySchema input_boundary = 9;
  BoundarySchema output_boundary = 10;
  SteeringSpec steering = 11;
}

message ReduceSpec {
  string reducer = 1;           // concat | first_success | majority_vote | synthesis | mock_synthesis
  string reducer_device_id = 2;
  string reducer_model_id = 3;
}

message RouteDecision {
  string selected_device_id = 1;
  string selected_model_id = 2;
  repeated string fallback_device_ids = 3;
  repeated string reasons = 4;
  float route_score = 5;
  string execution_mode = 6;
  repeated string selected_shard_device_ids = 7;
  repeated string selected_stage_device_ids = 8;
}

message RegistrationAccepted {
  string brain_id = 1;
  uint32 heartbeat_interval_ms = 2;
  string device_credential = 3; // set when a bootstrap credential is exchanged
}

message RegistrationRejected {
  string reason = 1;
}

message CommandAck {
  string command_id = 1;
  bool accepted = 2;
  string reason = 3;
}

message CancelTask {
  string task_id = 1;
  string reason = 2;
  string attempt_id = 3;
}

message HeartbeatAck {
  int64 brain_timestamp_ms = 1;
}

message ExecutionEvent {
  string task_id = 1;
  string event_type = 2;
  string message = 3;
  int64 timestamp_ms = 4;
}
```

## 9. Task Classification

### 9.1 MVP Task Taxonomy

Start with:

- `chat_qa`
- `summarization`
- `document_extraction`
- `reasoning_analysis`
- `translation_rewrite`
- `vision_understanding`
- `code_assistance`
- `unknown`

### 9.2 Classifier Interface

```python
class TaskClassifier(Protocol):
    def classify(self, request_text: str, preferred_mode: str) -> TaskProfile:
        ...
```

Implement:

- `RuleBasedTaskClassifier`: required and default.
- `EmbeddingTaskClassifier`: optional extension point.

The classifier must identify:

- Task class.
- Complexity: `low`, `medium`, or `high`.
- Estimated input size.
- Estimated output size.
- Privacy tier.
- Latency tier.
- Classification confidence.
- Whether the request appears compound.
- Whether the request is data-parallelizable.
- Whether the request is a layer-pipeline candidate.
- Whether vector steering is requested or implied.

Example rules:

| Trigger | Classification |
| --- | --- |
| Contains `summarize`, `summary`, or `key points` | `summarization` |
| Contains `extract`, `action items`, `find all`, or `parse` | `document_extraction` |
| Contains `compare`, `recommend`, `trade-offs`, or `analyze` | `reasoning_analysis` |
| Contains `rewrite`, `shorten`, `translate`, or `tone` | `translation_rewrite` |
| Contains `code`, `function`, `stack trace`, or `bug` | `code_assistance` |
| Contains `concise`, `verbose`, `persona`, `tone`, or `style` | set `steering_requested=true` |
| Contains numbered sections, repeated delimiters, or `for each` | set `data_parallelizable=true` |
| Explicit mode `layer_pipeline` or large model with compatible segments | set `layer_parallel_candidate=true` |
| Otherwise | `chat_qa` or `unknown` |

Complexity should consider:

- Request length.
- Presence of comparison, recommendation, or reasoning words.
- Compound markers such as `then`, `and also`, or multi-part numbering.
- Explicit quality preference.
- Required output length.

## 10. Execution Planner

The planner sits between classification and routing.

```python
class ExecutionPlanner(Protocol):
    def plan(
        self,
        request_text: str,
        profile: TaskProfile,
        preferred_mode: str,
        requested_execution_mode: str,
        steering: SteeringSpec | None,
    ) -> ExecutionPlan:
        ...
```

Planner rules:

- Use `single` when the task is not parallelizable or privacy excludes remote
  execution.
- Use `data_parallel` when the task has independent shards, a batch shape, or
  the user requests `parallel`.
- Use `layer_pipeline` only when compatible model segments are registered and
  the task benefits from a model larger or different than any one eligible
  device can run alone.
- Use steering metadata when explicitly requested or inferred from task words
  such as `concise`, `verbose`, `friendly`, `clinical`, or `tone`.
- Keep the plan deterministic for repeated inputs and same device state.

The planner must emit route reasons for mode selection. Examples:

- `Selected data_parallel: request contains 4 independent sections.`
- `Selected layer_pipeline: no single device advertises full large model, but phone and pc advertise contiguous compatible segments.`
- `Selected steering vector concise-vs-verbose-layer-7 for requested concise style.`
- `Selected single: private mode excludes remote shard execution.`

## 11. Health Model

### 11.1 Device Health Status

Calculate an explicit health status:

- `HEALTHY`
- `DEGRADED`
- `UNHEALTHY`
- `STALE`
- `OFFLINE`

Suggested thresholds:

| Condition | Result |
| --- | --- |
| No heartbeat for more than 20 seconds | `OFFLINE` |
| No heartbeat for 10 to 20 seconds | `STALE` |
| `thermal_level >= 0.85` | `UNHEALTHY` |
| `battery_pct < 10` and not charging | `UNHEALTHY` |
| `available_memory_mb < 512` | `DEGRADED` |
| `accelerator_utilization > 0.85` | `DEGRADED` |
| Otherwise | `HEALTHY` |

Make all thresholds configurable in `brain.yaml`.

### 11.2 Simulated Constraints

The MVP must provide endpoints or CLI flags to simulate:

```text
--simulate-thermal=0.95
--simulate-battery=5
--simulate-offline
--simulate-load=0.95
--simulate-memory=256
--simulate-rtt=150
```

This is essential for a reliable hackathon demo. Do not depend on a real phone
overheating or physically disconnecting from Wi-Fi.

## 12. Routing Policy

Implement a deterministic, explainable scoring function.

### 12.1 Hard Eligibility Filters

A device is normally eligible only if all conditions are true:

- Device health is `HEALTHY` or `DEGRADED`.
- Device is connected and reachable.
- Device advertises a compatible model for the task class.
- Device meets the task privacy requirement.
- Device has sufficient available memory.
- Device is not excluded by user mode:
  - `private`: originating or local device only.
  - `fast`: lowest predicted latency among eligible plans.
  - `quality`: highest model quality among eligible plans.
  - `parallel`: prefer data-parallel plan when safe.
- For steering, the device/model advertises a compatible steering mode, vector
  ID, hidden size, layer, alpha range, and positions mode.
- For layer pipelines, all selected stages form a compatible contiguous
  pipeline.

A `STALE` device is excluded from normal routing. It may be used only as a
last-resort fallback when no `HEALTHY` or `DEGRADED` device satisfies the other
hard filters. An `OFFLINE` device is never eligible for a new assignment.

### 12.2 Score

Use a normalized score:

```text
score = w_q Q + w_l L + w_h H + w_m M + w_p P + w_e E
```

Where:

- `Q`: model quality fit.
- `L`: inverse predicted latency.
- `H`: device health score.
- `M`: memory/runtime readiness score.
- `P`: policy/preference score.
- `E`: execution-mode fit score.

Suggested default weights:

```yaml
routing:
  quality_weight: 0.28
  latency_weight: 0.22
  health_weight: 0.22
  readiness_weight: 0.10
  preference_weight: 0.10
  execution_mode_weight: 0.08
```

For data-parallel plans, score the plan as:

```text
plan_score = min(shard_scores) * 0.35
           + average(shard_scores) * 0.35
           + reducer_score * 0.20
           + parallelism_fit * 0.10
```

For layer-pipeline plans, score the plan as:

```text
pipeline_score = min(stage_scores) * 0.30
               + average(stage_scores) * 0.25
               + boundary_transport_score * 0.20
               + model_contiguity_score * 0.15
               + steering_stage_fit * 0.10
```

### 12.3 Explainability

Every route decision must return reasons such as:

```json
[
  "Selected pc-01: model large-reasoning-v1 supports reasoning_analysis.",
  "Selected pc-01: thermal level 0.18 is lower than phone-01 at 0.76.",
  "Selected pc-01: 6144 MB memory available.",
  "Fallback phone-01 retained: supports small-chat-v1.",
  "Selected data_parallel: request split into 3 independent summary shards.",
  "Selected layer_pipeline: phone owns layers 0..13 and pc owns layers 14..27.",
  "Applied steering vector concise-vs-verbose-layer-7 at alpha=-2.0."
]
```

## 13. Executor Abstractions

### 13.1 Single Task Executor

```python
class TaskExecutor(Protocol):
    async def execute(
        self,
        task_id: str,
        attempt_id: str,
        request_text: str,
        model_id: str,
        profile: TaskProfile,
        timeout_ms: int,
        steering: SteeringSpec | None = None,
    ) -> TaskResult:
        ...
```

Required executor: `MockExecutor`.

The mock executor must:

- Sleep for a configurable latency based on model tier.
- Return deterministic synthetic output.
- Include steering settings in the output when steering is enabled.
- Optionally fail when a simulated condition is active.
- Emit execution latency.

Example behavior:

```text
small-chat-v1         -> about 250 ms
large-reasoning-v1    -> about 900 ms
steered-small-chat-v1 -> about 300 ms
```

Example output:

```text
[Mock result from Snapdragon PC using large-reasoning-v1]
Task class: reasoning_analysis
Execution mode: single
Steering: concise-vs-verbose-layer-7 alpha=-2.0 positions=last
Response: Selected execution completed successfully.
```

### 13.2 Data-Parallel Executor

```python
class ShardExecutor(Protocol):
    async def execute_shard(
        self,
        task_id: str,
        attempt_id: str,
        shard_id: str,
        request_text: str,
        model_id: str,
        profile: TaskProfile,
        timeout_ms: int,
        steering: SteeringSpec | None = None,
    ) -> PartialTaskResult:
        ...
```

The Brain owns orchestration. Agents only execute assigned shards.

Required MVP components:

- `MockShardExecutor`
- `ParallelDispatcher`
- `ResultReducer`

### 13.3 Layer-Pipeline Executor

```python
class PipelineStageExecutor(Protocol):
    async def execute_stage(
        self,
        task_id: str,
        attempt_id: str,
        stage_id: str,
        request_text: str,
        segment_id: str,
        input_boundary: BoundaryTensor | None,
        output_schema: BoundarySchema,
        profile: TaskProfile,
        timeout_ms: int,
        steering: SteeringSpec | None = None,
    ) -> PipelineStageResult:
        ...
```

Required MVP component: `MockPipelineExecutor`.

The mock pipeline executor must:

- Validate the stage owns the requested steering layer when steering is set.
- Return deterministic synthetic boundary metadata for intermediate stages.
- Return deterministic output text for the final stage.
- Emit stage latency.

Runtime integration target: `QnnPipelineExecutor`.

```python
class QnnPipelineExecutor(PipelineStageExecutor):
    """
    DragonNest adapter for the validated split-compute implementation in
    PersonaCare-Steering-Research/src/split_compute/.

    Responsibilities:
    - resolve segment artifacts
    - initialize/load QNN context binaries
    - run prompt/decode stage graphs
    - read/write boundary tensors
    - apply steering at owned layer when supported
    - collect per-stage latency and accelerator telemetry
    - map runtime failures to PipelineStageResult error codes
    """
```

### 13.4 Runtime Executor Integration

The executor layer must be pluggable and provide these implementations:

- `MockExecutor`: required for the MVP.
- `GenieExecutor`: adapt `PersonaCare/genie_runner.py` for Snapdragon LLM
  execution.
- `QnnExecutor`: adapt `PersonaCare/qnn_runner.py` for local `.dlc` and QNN
  context-binary execution.

Every executor reports the model ID and version, runtime name and version,
accelerator used, execution latency, error code and message, and observed
memory and thermal deltas when available.

```python
class GenieExecutor(TaskExecutor):
    """DragonNest adapter around the validated PersonaCare Genie runner."""
```

```python
class QnnExecutor(TaskExecutor):
    """
    DragonNest adapter around the validated PersonaCare QAIRT/QNN runner.

    Responsibilities:
    - resolve model artifact
    - initialize/load runtime
    - submit inference
    - collect latency and accelerator telemetry
    - map runtime failures to TaskResult error codes
    """
```

The adapters should preserve the existing runners' timeout, retry, subprocess
error, backend-selection, and profiling behavior. DragonNest adds manifest
resolution, normalized metrics/errors, cancellation, and Agent lifecycle
integration around those runners; it should not duplicate the low-level tensor
I/O and runtime invocation without a demonstrated need.

## 14. Steering Vector Registry

Create `config/steering-vectors.yaml`.

Example:

```yaml
vectors:
  - vector_id: concise-vs-verbose-layer-7
    model_family: qwen3
    model_revision: qwen3-0.6b-demo
    hidden_size: 1024
    target_layers: [7]
    default_layer: 7
    alpha_min: -4.0
    alpha_max: 4.0
    default_alpha: -2.0
    positions: ["all", "last"]
    default_positions: "last"
    storage_uri: "artifacts/vector_layer_7.pt"
    safety_label: "style_verbosity"
    allow_remote_vector: false
```

The MVP does not need to load `.pt` vector files. It must load metadata,
perform policy and compatibility checks, and pass `SteeringSpec` through the
route, dispatch, execution, result, and dashboard layers.

## 15. Dashboard / Demo UI

Build a minimal dashboard with six panels.

### A. Registered Device Cards

For each device show:

- Device name/type.
- Connection state.
- Health status.
- Battery.
- Thermal level.
- Memory available.
- Accelerator utilization.
- Installed model roles.
- Parallel capabilities.
- Steering capabilities.
- Layer segment capabilities.
- Active tasks.

### B. Task Submission

Show:

- Text area.
- Mode selector: `Auto`, `Fast`, `Private`, `Quality`, `Parallel`.
- Execution selector: `Auto`, `Single`, `Data Parallel`, `Layer Pipeline`.
- Optional steering selector:
  - Vector ID.
  - Alpha.
  - Positions.
  - Apply steering checkbox.

### C. Routing Trace

Show:

```text
Request
  -> Task class: reasoning_analysis (0.88 confidence)
  -> Execution mode: data_parallel
  -> Eligible devices: phone-01, pc-01
  -> Shard 1 selected: phone-01 / small-chat-v1
  -> Shard 2 selected: pc-01 / large-reasoning-v1
  -> Reducer: pc-01 / large-reasoning-v1
  -> Steering: concise-vs-verbose-layer-7 alpha=-2.0 positions=last
  -> Reason: compound request split into independent analysis shards
```

### D. Parallel Progress

Show:

- Shard or stage ID.
- Selected device.
- Selected model or segment.
- Status: queued, running, succeeded, failed, cancelled, rerouted.
- Latency.
- Retry count.
- Boundary tensor metadata for layer-pipeline mode.

### E. Result

Show:

- Final output.
- Partial outputs when data-parallel mode is used.
- Reducer output and reducer policy.
- Final stage output when layer-pipeline mode is used.

### F. Live Event Log

Show registration, health transitions, dispatch, shard start, stage start,
completion, failure, cancellation, reroute, and reducer events.

WebSocket/SSE is optional. Polling every second is acceptable for the MVP.

## 16. Required Demo Scenarios

### Scenario 1: Normal Single-Device Route

1. Start Brain.
2. Start phone agent and PC agent.
3. Both register and become `HEALTHY`.
4. Submit: `Compare these two project approaches and recommend one.`
5. Classifier returns `reasoning_analysis`.
6. Router chooses PC with `large-reasoning-v1`.
7. Dashboard displays the reason and result.

### Scenario 2: Thermal Reroute

1. Simulate PC thermal state of `0.95`.
2. Submit the same task.
3. Router excludes or heavily penalizes PC.
4. Router chooses phone with `small-chat-v1`, or reports a controlled quality
   downgrade.
5. Dashboard visibly explains the reroute.

### Scenario 3: Private Mode

1. Submit a task in private mode.
2. Router selects only the originating/local device.
3. Dashboard states that remote trusted devices were excluded by policy.

### Scenario 4: Device Disconnect

1. Dispatch a task to the PC.
2. Stop the PC agent or simulate offline state before completion.
3. Brain marks it offline.
4. Brain retries once on the fallback device.
5. Dashboard shows the original failure and successful reroute.

### Scenario 5: Data-Parallel Fanout

1. Start phone and PC agents.
2. Submit:
   `Summarize section 1, section 2, and section 3, then give one final set of key points.`
3. Planner creates three summary shards and one reducer.
4. Router assigns shards across phone and PC.
5. Reducer combines shard outputs.
6. Dashboard shows shard progress, reducer choice, and final output.

### Scenario 6: Data-Parallel Replica Race

1. Submit a low-latency request with execution mode `data_parallel` and
   reducer `first_success`.
2. Brain sends the same shard to phone and PC.
3. The first successful result wins.
4. Brain cancels the slower replica if cancellation is supported.
5. Dashboard shows the winning device and cancelled replica.

### Scenario 7: Layer-Pipeline Mock Split

1. Phone advertises `qwen3-0.6b` segment layers `0..14`.
2. PC advertises `qwen3-0.6b` segment layers `14..28`.
3. Submit a high-quality task with execution mode `layer_pipeline`.
4. Planner builds a two-stage pipeline.
5. Stage 0 runs on phone and emits mock hidden boundary metadata.
6. Stage 1 runs on PC and emits final mock output.
7. Dashboard shows stage order, boundary schema, and per-stage latency.

### Scenario 8: Vector Steering

1. Register a steering vector such as `concise-vs-verbose-layer-7`.
2. Start an agent that advertises compatible steering support.
3. Submit:
   `Answer concisely: explain why local AI routing matters.`
4. Classifier or UI sets steering enabled with negative alpha.
5. Router selects a steering-compatible model.
6. Mock executor includes steering settings in output.
7. Dashboard shows vector ID, alpha, target layer, positions, and policy reason.

### Scenario 9: Steering With Layer Pipeline

1. Phone owns layers `0..14`.
2. PC owns layers `14..28`.
3. Steering vector targets layer `7`.
4. Planner attaches steering to the phone stage only.
5. Dashboard explains that steering is applied at Stage 0 because Stage 0 owns
   layer 7.

## 17. Acceptance Criteria

The implementation is complete when all are true:

- Brain starts gRPC server and HTTP dashboard/API.
- At least two independently started agents can register.
- Brain maintains current capabilities and health for every agent.
- Agents use a persistent bidirectional gRPC stream.
- A text task is classified into a structured `TaskProfile`.
- The planner emits an `ExecutionPlan`.
- The router makes a deterministic model/device/plan decision.
- The route includes human-readable reasons and fallback devices.
- A selected device receives and executes a task.
- Agent returns a task result through the stream.
- An unhealthy/offline device is excluded from new routing.
- An interrupted task can reroute once to an eligible fallback.
- Data-parallel execution can dispatch at least two shards to different agents.
- Data-parallel execution can reduce partial results into one final response.
- Layer-pipeline execution can create and execute at least a two-stage mock
  pipeline.
- Layer-pipeline routing validates compatible contiguous model segments.
- Vector steering metadata can be loaded from config.
- Steering policy rejects incompatible vector/model/layer/alpha combinations.
- Steering metadata is passed through route, dispatch, execution, and dashboard.
- Dashboard shows devices, health, route trace, execution mode, steering state,
  shard/stage progress, and result.
- Unit tests cover registration, health timeout, routing policy, data-parallel
  planning/reduction, layer-pipeline planning, steering policy, and reroute
  behavior.
- Integration tests cover stream disconnect, heartbeat expiry, mid-job retry,
  stale late results, reconnection, offline routing exclusion, and simulated
  network and resource constraints.
- `README.md` provides copy/paste commands for the full demo.

## 18. Required README Commands

The coding agent should ensure these kinds of commands work:

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Generate gRPC bindings
python -m grpc_tools.protoc \
  -I proto \
  --python_out=. \
  --grpc_python_out=. \
  proto/snaprouter.proto

# Start the Brain
python -m brain.main --config config/brain.yaml

# In separate terminals: start two agents
python -m agent.main --config config/phone-agent.yaml
python -m agent.main --config config/pc-agent.yaml

# Run tests
pytest -q

# Demo scripts
bash scripts/demo_reroute.sh
bash scripts/demo_data_parallel.sh
bash scripts/demo_layer_pipeline.sh
bash scripts/demo_steering.sh
```

## 19. Implementation Order

1. Define protobuf messages and generate bindings.
2. Implement Brain registry and bidirectional `Connect` stream.
3. Implement agent registration plus heartbeat loop.
4. Implement health state transitions and TTL expiry.
5. Implement mock capabilities and mock single-task executor.
6. Implement rule-based classifier.
7. Implement steering vector metadata registry and policy checks.
8. Implement execution planner for `single`, `data_parallel`, and
   `layer_pipeline`.
9. Implement deterministic routing and explanation output.
10. Implement task dispatch/result handling for single-device execution.
11. Implement data-parallel shard dispatch and result reduction.
12. Implement mock layer-pipeline stage dispatch and boundary metadata.
13. Implement retry/reroute behavior for single tasks and shards.
14. Implement basic pipeline failure handling.
15. Port `PersonaCare/qnn_runner.py` and `PersonaCare/genie_runner.py` behind
    DragonNest executor adapters.
16. Create artifact manifests for the first validated QNN and Genie models.
17. Port the validated split-compute wrappers behind `QnnPipelineExecutor`.
18. Add dashboard/API.
19. Add Android-agent packaging interfaces and platform telemetry abstraction.
20. Add simulation controls and disconnect/recovery integration tests.
21. Document demo flow.

## 20. Configuration Sketch

`config/brain.yaml`:

```yaml
brain:
  brain_id: snaprouter-brain-dev
  dev_mode: true
  enrollment_token: dev-token

health:
  heartbeat_interval_ms: 2000
  stale_after_ms: 10000
  offline_after_ms: 20000
  unhealthy_thermal: 0.85
  unhealthy_battery_pct: 10
  degraded_memory_mb: 512
  degraded_accelerator_utilization: 0.85

routing:
  quality_weight: 0.28
  latency_weight: 0.22
  health_weight: 0.22
  readiness_weight: 0.10
  preference_weight: 0.10
  execution_mode_weight: 0.08
  min_available_memory_mb: 512
  max_rtt_interactive_ms: 80
  max_rtt_layer_pipeline_ms: 25

parallel:
  max_shards_per_task: 8
  max_retries_per_shard: 1
  default_reducer: mock_synthesis
  enable_replica_race: true
  enable_layer_pipeline: true

steering:
  enabled: true
  vector_config: config/steering-vectors.yaml
  reject_alpha_out_of_range: true
  allow_remote_vectors_by_default: false
```

`config/phone-agent.yaml`:

```yaml
device:
  device_id: phone-01
  display_name: Snapdragon Phone
  device_type: phone
  platform: android
  enrollment_token: dev-token

capabilities:
  total_memory_mb: 8192
  accelerators:
    - name: htp
      available: true
      precisions: [fp16, int8]
  models:
    - model_id: small-chat-v1
      model_family: mock
      role: small_chat
      task_classes: [chat_qa, summarization, translation_rewrite, reasoning_analysis]
      max_context_tokens: 4096
      warm: true
      quality_score: 0.65
      steering:
        - steering_mode: mock
          vector_ids: [concise-vs-verbose-layer-7]
          supported_layers: [7]
          hidden_size: 1024
          alpha_min: -4.0
          alpha_max: 4.0
          positions: [all, last]
      segments:
        - segment_id: qwen3-0.6b-layers-0-14
          pipeline_id: qwen3-0.6b-split-14
          model_id: qwen3-0.6b-part-a
          model_family: qwen3
          model_revision: demo
          start_layer: 0
          end_layer: 14
          total_layers: 28
          includes_embedding: true
          includes_lm_head: false
  parallel:
    - mode: data_parallel
      max_concurrent_tasks: 2
      max_batch_shards: 4
      supports_cancellation: true
    - mode: layer_pipeline
      max_concurrent_tasks: 1
      max_batch_shards: 1
      supports_cancellation: true
```

`config/pc-agent.yaml` should mirror the phone config but advertise
`large-reasoning-v1`, higher memory, and the complementary pipeline segment
`qwen3-0.6b-layers-14-28`.

## 21. Required Runtime Extensions

### 21.1 Android Agent Packaging

DragonNest must support a future Android Agent package that can run as a
long-lived device agent. Its design must include:

- A foreground service or equivalent long-running execution mode.
- A reconnect loop with exponential backoff.
- A network-change callback that triggers an immediate heartbeat and health
  refresh.
- An enrollment credential stored in Android Keystore.
- A graceful shutdown message when possible.
- Support for simulated offline, thermal, battery, and load states.

### 21.2 QNN / Genie Executor Integration

The Agent executor layer must support the pluggable runtimes defined in
Section 13.4: `MockExecutor` for the MVP, a `GenieExecutor` adapter based on
`PersonaCare/genie_runner.py`, and a `QnnExecutor` adapter based on
`PersonaCare/qnn_runner.py`.

The initial integration should reuse the models already validated by
PersonaCare where they fit the requested task: Qwen3-4B for Genie text
generation and the existing `.dlc`/QNN context-binary pipelines for QNN-backed
execution. `QnnPipelineExecutor` should reuse the validated split-compute code
and evidence from `PersonaCare-Steering-Research`.

All executors must populate `ExecutionMetrics`, including model ID and version,
runtime name and version, accelerator used, execution latency, error code and
message, and observed memory and thermal deltas when available.

### 21.3 Model Artifact Management

Model capabilities must be backed by a model artifact manifest containing:

- `model_id`
- `model_version`
- `runtime`
- `artifact_path`
- `checksum`
- `tokenizer_id`
- `precision`
- `supported_accelerators`
- `min_memory_mb`
- `max_context_tokens`
- `supports_steering`
- `supports_data_parallel`
- `supports_layer_pipeline`
- Optional split-layer boundary metadata.

The Agent must validate the manifest and artifact checksum before advertising
the model as available. The Brain uses the advertised manifest fields for
runtime, memory, steering, and pipeline compatibility checks.

Artifact manifests should initially describe the externally downloaded models
documented in `PersonaCare/README.md` and the split/GenieX artifacts documented
under `PersonaCare-Steering-Research/docs/results/`. A manifest may reference
an absolute path, configured artifact root, or managed cache; the large binary
artifacts do not need to be committed to DragonNest.

### 21.4 Secure Enrollment

The MVP may use a shared enrollment token when `dev_mode=true`.

The Android development workflow should also support QR onboarding. The QR may
contain only a short-lived bootstrap credential, Brain address, TLS mode,
schema version, session ID, and expiry. The Agent must validate these fields,
bind the first successful claim to its generated device ID, and replace the
bootstrap credential in Android Keystore with the device credential returned by
the Brain. A claimed bootstrap must not enroll a different device.

Personal registration data used for steering must be persisted in durable Brain
state. It must not be stored only in the QR session or Android process.

The production design must include:

- mTLS/client certificate enrollment.
- Device certificate fingerprint tracking.
- Trust revocation.
- Certificate rotation.
- No dependency on a MAC address or hardware serial number as primary identity.

### 21.5 Task Lifecycle, Cancellation, and Retry

The Brain must manage task state explicitly:

```text
QUEUED -> DISPATCHED -> RUNNING -> SUCCEEDED
QUEUED -> DISPATCHED -> RUNNING -> RETRYING -> SUCCEEDED
QUEUED -> DISPATCHED -> RUNNING -> FAILED
RUNNING -> CANCELLING -> CANCELLED
```

Retries must preserve the same `task_id` and create a new `attempt_id`. The
Brain is the authority for accepted results and final state; Agent execution is
best-effort. Results for cancelled, offline, or superseded attempts remain in
the event history but cannot change the final task result.

### 21.6 Real Telemetry

Agents should expose a platform telemetry abstraction for:

- Battery percentage and charging state.
- Thermal level.
- Available memory.
- CPU utilization.
- Accelerator utilization when available.
- Network RTT to the Brain.
- Active task IDs.
- Runtime/model warm state.

Unavailable platform metrics must use an explicit unknown value rather than a
healthy default.

### 21.7 Network and Disconnect Tests

Integration tests must cover:

- Agent stream disconnect.
- Missed heartbeat timeout.
- Mid-job disconnect and fallback retry.
- Late result ignored after retry.
- Reconnect after offline.
- No routing to offline devices.
- Simulated high RTT, load, and thermal conditions.

## 22. Definition of Done

The final product should let a judge see, in under two minutes:

> Two Snapdragon devices joined a trusted fabric. The Brain understood the
> incoming task, selected a device/model/execution mode using live resource
> health, executed it, applied steering when requested, split work when useful,
> and rerouted when the first option became constrained.

# Hardware Contract for Desktop Codex

What must be implemented for real QAIRT/QNN/Genie execution and real
artifact provisioning on the Snapdragon X Elite laptop and the Galaxy S25
Ultra. The control plane (scheduler, behavior fallback, provisioning state
machine, dashboard) is complete and tested against mock adapters; the items
below are the only hardware-facing obligations, each behind an existing
interface.

## 1. Agent advertisement contract (both devices)

The Brain-side artifact catalog (`configs/artifact-catalog.yaml`) is keyed by
`artifact_id`. An agent advertises a deployment by reporting a
`ModelCapability` whose `model_id` **equals the catalog `artifact_id`**, and
must:

- advertise an artifact only after `ArtifactRegistry` path + checksum
  validation succeeds (existing behavior in `transport/agent.py` /
  `AndroidArtifactRegistry`);
- report `warm` truthfully: `warm=true` (and membership in heartbeat
  `warm_model_ids`) only while the runtime actually holds the model loaded —
  the scheduler's cold-load cost and eviction penalty depend on it;
- set `supports_steering=true` on a capability **only if** the runtime can
  inject `steered_hidden = hidden + alpha * mask * vector` at the layers in
  `supported_steering_layers` for the vectors in `steering_vector_ids`.
  Prompt-prefixing must never be advertised as steering support;
- report SoC identity in `HardwareInventory.soc_model` (drives
  compatibility-class mapping: `X Elite`/`X1E` → `snapdragon-x-elite`,
  `8 Elite`/`SM8750` → `snapdragon-8-elite`);
- report `npu_status` from a real probe (`available | unavailable |
  not_probed`), never inferred from the brand name.

No proto changes are required for any of this.

## 2. X Elite laptop: Genie execution

Behind `GenieExecutor` (`src/dragon_nest/runtime/executors.py`, wrapping the
validated `runtime/genie_runner.py`):

- resolve `qwen3-4b-genie` via the artifact manifest env vars
  (`GENIE_DIR`, `QWEN3_4B_GENIE_SHA256_TREE`);
- execute `ExecuteTask` prompts through `genie-t2t-run`, honoring
  `timeout_ms`, and populate `ExecutionMetrics` (runtime name/version,
  accelerator `htp`, latency, memory/thermal deltas when measurable);
- measure and report real load time and prefill/decode throughput so the
  catalog's `measured_*` fields can replace estimates;
- runtime steering: Genie has no runtime-vector injection today — the agent
  must NOT advertise `supports_steering` for `qwen3-4b-genie`. The concise
  behavior on this artifact is realized by the baked variant
  (`qwen3-4b-genie-concise-baked`) once compiled (see §4).

## 3. S25 Ultra: QNN/Genie execution in the Android Agent

Behind `AndroidTaskExecutor` (the packaged `GenieAndroidTaskExecutor` /
`QnnAndroidTaskExecutor` and the QAIRT 2.48 Genie JNI bridge):

- load the verified S25 bundle (`qwen3-4b-qnn-s25` target), validate
  checksums via `AndroidArtifactRegistry`, and only then advertise it;
- execute tasks/shards on the Hexagon NPU, mapping failures to the existing
  error codes; keep the mock executor as the fallback path;
- report warm state transitions to the heartbeat `warm_model_ids`;
- pipeline stages: implement real boundary-tensor I/O for
  `qwen3-0.6b-part-a` using the existing checksummed `BoundaryTensor`
  contract (the Python `QnnPipelineExecutor` is the host reference);
- runtime steering on QNN requires the steering-input graphs from
  `PersonaCare-Steering-Research/src/steering_poc/export_onnx.py`; until a
  compiled steering-enabled graph is validated on-device, do not advertise
  `supports_steering` for QNN artifacts. Update
  `configs/steering-vectors.yaml` `validated_runtimes`/`validated_quantizations`
  only after on-target numerical validation (the scheduler refuses
  unvalidated combinations by design).

## 4. Real provisioning adapter (AI Hub)

Implement a `ProvisioningAdapter` (see `src/dragon_nest/provisioning.py`)
that replaces `MockAiHubAdapter`:

```python
class AiHubProvisioningAdapter:
    name = "qualcomm-ai-hub"
    def advance(self, job: ProvisioningJob) -> str: ...
```

- `build_queued/compiling`: submit and poll a real AI Hub compile job for the
  target compatibility class (pattern: `scripts/validate_ai_hub.py`, which
  already compiles/profiles on hosted Snapdragon 8 Elite and X Elite);
- `validating`: run the numerics checks (steering delta / boundary parity)
  before reporting success; a failed check must move the job to `failed`,
  never to `ready_remote`;
- `downloading/installed/warm`: deliver the artifact to the device agent,
  verify its checksum on-device, and only then let the agent advertise it —
  the Brain's `mark_ready` + deployment override flow is already wired to
  `on_deployed`;
- detail strings must reflect reality (include AI Hub job IDs); the `[mock]`
  prefix is reserved for the mock adapter. The dashboard renders whatever the
  adapter reports — honest detail strings are the contract;
- baked steering variants (`qwen3-4b-genie-concise-baked`,
  `family-assistant-v0-baked` equivalents) are produced here: bake = apply
  the validated steering vector at export time, recompile, and record
  provenance + checksum in the catalog entry.

## 5. Telemetry obligations

- Windows agent: native memory/battery probes exist (`telemetry.py`);
  Desktop Codex should add CPU utilization (`GetSystemTimes`) and, where
  exposed, thermal-zone data; unknown values stay explicit (-1/0).
- Android agent: continue reporting real memory/thermal/battery/load in
  heartbeats; RTT is measured against the Brain automatically.
- Measured prefill/decode throughput per (device, artifact) should be fed
  back into `DeploymentState.measured_*` (the scheduler already prefers
  measured values over catalog estimates).

## 6. Acceptance checks

1. `pytest -q` stays green with the real adapters importable but inactive.
2. `scripts/demo_scenarios.py` still passes (mock path must keep working).
3. On hardware: a `concise` request routes runtime-vector on a device that
   advertises validated steering, and baked on one that does not, with the
   dashboard explanation matching what actually executed.
4. A provisioning run driven by the real adapter shows real job IDs and
   produces an artifact whose on-device checksum matches the catalog.

# Behavior-Aware Deployment Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve DragonNest into a runnable heterogeneous Snapdragon demo where the Brain routes *executable deployments* — (device, artifact, behavior realization) triples — using live telemetry, explicit feasibility filtering, an explainable cost model, explicit behavior fallback, and a provisioning state machine, all visible in the dashboard.

**Architecture:** New Brain-side modules (`behavior.py`, `deployments.py`, `scheduler.py`, `provisioning.py`) layered on top of the existing registry/tasks/dispatch machinery. No proto changes: devices keep advertising `ModelCapability` (model_id, warm, supports_steering); the Brain-side artifact catalog supplies everything else keyed by artifact_id, and a `DeploymentIndex` derives per-(device, artifact) states from advertisements + heartbeat warm lists + simulation overlays. The dashboard gets a behavior-routing panel, candidate/rejection tables, provisioning progress, and extended simulation controls. Existing single/data_parallel/layer_pipeline paths, gRPC behavior, and recovery semantics are preserved untouched.

**Tech Stack:** Python 3.13, dataclasses + StrEnum, FastAPI/pydantic dashboard, pytest, vanilla-JS dashboard, YAML configs. No new infrastructure.

## Global Constraints

- Do not rewrite the project; preserve existing abstractions, tests, gRPC behavior, recovery semantics, and the single/data_parallel/layer_pipeline modes.
- The scheduler routes executable deployments, not abstract models and not raw free memory.
- Never describe prompt-only conditioning as activation steering.
- Never silently switch to a different behavior profile; fallback must follow an explicit policy: exact_only | allow_baked_equivalent | allow_runtime_equivalent | allow_prompt_fallback | allow_unsteered | reject.
- No learned scheduler — deterministic, explainable, configurable scoring only.
- Unknown estimates use conservative defaults and are visibly flagged as estimates.
- Keep runnable without Qualcomm SDKs (mock adapters); real hardware behind interfaces.
- No proto changes needed (verified: catalog-keyed design). If that changes, regenerate bindings and update clients.
- No Kubernetes/Redis/Celery; no arbitrary repartitioning or KV migration; fixed validated pipeline templates only.
- Persist only what must survive Brain restarts (nothing new qualifies; provisioning jobs are demo-scoped, in-memory).
- No secrets in SQLite/config/logs/browser payloads; no artifact binaries in git.

---

### Task 1: Windows-native telemetry (fixes 16 pre-existing failures on Windows)

**Files:**
- Modify: `src/dragon_nest/telemetry.py`
- Test: `tests/test_telemetry.py` (extend)

`SystemTelemetry` returns memory 0 / cpu -1 / battery -1 on Windows, so agents on the Snapdragon X Elite laptop heartbeat "no memory" and are rejected by the router. Add ctypes-based Windows probes (no new deps):
- `_available_memory_mb()`: on win32, `GlobalMemoryStatusEx` → `ullAvailPhys // 1MB`.
- `_battery_state()`: on win32, `GetSystemPowerStatus` → (BatteryLifePercent if != 255 else -1, ACLineStatus == 1).
- `_cpu_utilization()`: keep `os.getloadavg` fallback (`AttributeError` on Windows → -1 explicit unknown; acceptable).

Steps: write failing test asserting `SystemTelemetry(...).sample().health.available_memory_mb > 0` on win32 (skip elsewhere); implement; run `pytest tests/test_telemetry.py tests/test_grpc_transport.py -q` → all pass; commit.

**Interfaces produced:** unchanged signatures; heartbeats now carry real memory/battery on Windows.

### Task 2: BehaviorProfile + SteeringRealization (`behavior.py`)

**Files:**
- Create: `src/dragon_nest/behavior.py`, `configs/behavior-profiles.yaml`
- Test: `tests/test_behavior.py`

Enums:
```python
class SteeringRealizationMode(StrEnum):
    RUNTIME_VECTOR = "runtime_vector"; BAKED_PROFILE = "baked_profile"
    PROMPT_PROFILE = "prompt_profile"; NONE = "none"

class BehaviorFallbackPolicy(StrEnum):
    EXACT_ONLY = "exact_only"; ALLOW_BAKED_EQUIVALENT = "allow_baked_equivalent"
    ALLOW_RUNTIME_EQUIVALENT = "allow_runtime_equivalent"
    ALLOW_PROMPT_FALLBACK = "allow_prompt_fallback"
    ALLOW_UNSTEERED = "allow_unsteered"; REJECT = "reject"
```

Dataclasses (frozen):
- `SteeringRealization(mode, vector_id="", alpha=0.0, alpha_min=0.0, alpha_max=0.0, injection_layer=-1, positions="last", baked_artifact_id="", prompt_template="", compatible_model_families=(), compatible_runtimes=(), compatible_quantizations=(), verification_status="unverified")`
- `BehaviorProfile(profile_id, display_name, description, base_model_family, version, policy_tags=(), fallback_policy=BehaviorFallbackPolicy.ALLOW_UNSTEERED, provenance="", evaluation_status="draft", realizations=tuple[SteeringRealization, ...])` — realizations are declared in preference order.
- `BehaviorProfileRegistry.from_yaml(path)`; `.get(profile_id)`; `.all()`; `.allowed_modes(profile) -> tuple[SteeringRealizationMode, ...]` implementing the fallback ladder: exact_only → (first realization's mode only); allow_baked_equivalent → {preferred, baked}; allow_runtime_equivalent → {preferred, baked, runtime}; allow_prompt_fallback → + prompt; allow_unsteered → + none; reject behaves as exact_only but the scheduler reports `reject` semantics.

`configs/behavior-profiles.yaml`: concise (runtime_vector concise-vs-verbose-layer-7 alpha −2.0 + baked `*-concise-baked` + prompt fallback; allow_baked_equivalent), friendly (runtime vector `friendly-warmth-layer-7` + prompt; allow_prompt_fallback), medical-safe (baked only, exact_only, evaluation_status validated), creative (runtime + prompt), formal (prompt only), family-assistant (missing realizations for demo scenario G → provisioning; fallback reject).

Tests: registry loads all six; fallback ladders produce expected mode sets; prompt realization is never labeled activation steering (assert `SteeringRealization.describe()` says "prompt profile (not activation steering)" for prompt mode).

### Task 3: Steering-vector lifecycle (`steering.py` + config)

**Files:**
- Modify: `src/dragon_nest/models.py` (`SteeringVector` fields), `src/dragon_nest/steering.py`, `configs/steering-vectors.yaml`
- Test: `tests/test_steering.py` (extend)

Extend `SteeringVector` with optional lifecycle fields (backward-compatible defaults): `model_revision=""`, `base_model_fingerprint=""`, `tokenizer_fingerprint=""`, `source_layer=-1`, `extraction_method=""`, `positive_dataset_hash=""`, `negative_dataset_hash=""`, `normalization=""`, `dtype=""`, `checksum=""`, `created_at=""`, `creator=""`, `evaluation_metrics: tuple[tuple[str, float], ...] = ()`, `evaluation_dataset_version=""`, `validated_runtimes: tuple[str, ...] = ()`, `validated_quantizations: tuple[str, ...] = ()`, `status="draft"` (draft|calibrated|validated|deprecated|rejected).

`SteeringRegistry.from_yaml` parses the new fields with defaults. Add `SteeringRegistry.runtime_compatible(vector_id, model_family, model_revision, runtime, quantization, injection_layer) -> tuple[bool, str]`: refuses cross-family, cross-revision (when vector pins one), unvalidated runtime, unvalidated quantization, and unlisted injection layers; refuses `status` in {draft, deprecated, rejected}. Existing `validate()` unchanged except vectors with status deprecated/rejected fail.

Update `configs/steering-vectors.yaml`: enrich `concise-vs-verbose-layer-7` (status: validated, validated_runtimes [mock, genie], validated_quantizations [none, w4a16], dtype fp32, extraction_method mean_difference, etc.) and add `friendly-warmth-layer-7` (status: calibrated, validated_runtimes [mock]).

Tests: yaml round-trip of new fields; runtime_compatible rejects wrong quantization / unvalidated runtime / draft status; existing tests stay green.

### Task 4: ArtifactSpec + DeploymentState (`deployments.py`)

**Files:**
- Create: `src/dragon_nest/deployments.py`, `configs/artifact-catalog.yaml`
- Test: `tests/test_deployments.py`

```python
class ArtifactState(StrEnum):
    ABSENT="absent"; AVAILABLE_REMOTE="available_remote"; DOWNLOADING="downloading"
    INSTALLED="installed"; LOADING="loading"; WARM="warm"; FAILED="failed"; QUARANTINED="quarantined"

@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str; base_model_id: str; base_model_family: str; model_version: str
    behavior_profile_id: str = ""        # non-empty => baked artifact
    steering_realization: str = "none"   # how behavior is realized IN the artifact
    compatibility_classes: tuple[str, ...] = ("mock",)
    runtime: str = "mock"; runtime_version: str = ""
    quantization: str = "none"; max_context_tokens: int = 4096
    topology: str = "full_model"         # full_model | pipeline_stage
    start_layer: int = -1; end_layer: int = -1
    boundary_schema: str = ""            # e.g. "hidden[1,32,1024]fp32"
    checksum: str = ""; artifact_size_mb: int = 0
    estimated_memory_mb: int = 0; measured_memory_mb: int = 0
    measured_load_time_ms: int = 0
    prefill_tokens_per_s: float = 0.0; decode_tokens_per_s: float = 0.0
    kv_cache_bytes_per_token: int = 0
    build_provenance: str = ""; readiness: str = "ready"   # ready | unvalidated
    def memory_mb(self) -> tuple[int, bool]:  # (value, is_estimate)
    def supports_context(self, tokens: int) -> bool

@dataclass(frozen=True)
class DeploymentState:
    device_id: str; artifact_id: str; state: ArtifactState
    resident_bytes: int = 0
    measured_prefill_tokens_per_s: float = 0.0
    measured_decode_tokens_per_s: float = 0.0
```

- `ArtifactCatalog.from_yaml(path)`, `.get(artifact_id)`, `.all()`, `.baked_for(profile_id, base_model_family)`, `.base_for(base_model_family, compatibility_classes)`.
- `device_compatibility_classes(device) -> tuple[str, ...]`: maps `hardware.soc_model` ("Snapdragon X Elite"/"X1E" → `snapdragon-x-elite`; "SM8750"/"8 Elite"/"8750" → `snapdragon-8-elite`), plus `mock` always (mock artifacts run anywhere for the portable demo).
- `DeploymentIndex.build(records, catalog, overrides) -> dict[(device_id, artifact_id), DeploymentState]`: advertised model_id matching a catalog artifact → WARM if in `record.warm_model_ids` (or model.warm) else INSTALLED; catalog artifacts not advertised → ABSENT; `overrides[(device_id, artifact_id)] = ArtifactState` wins (simulation), including forcing ABSENT/INSTALLED/WARM.

`configs/artifact-catalog.yaml` entries (all mock-runnable; realistic numbers as *estimates*):
- `small-chat-v1` (mock class, base qwen3-mini family "mock", 900 MB est, warm-capable, kv 96 KB/token, load 1500 ms, prefill 350/s decode 45/s)
- `large-reasoning-v1` (mock class, 3800 MB est, kv 192 KB/token, load 9000 ms, prefill 240/s decode 18/s)
- `small-chat-v1-concise-baked` (baked_profile concise on small-chat-v1)
- `large-reasoning-v1-medical-safe-baked` (baked medical-safe)
- `qwen3-4b-genie` (class snapdragon-x-elite, runtime genie QAIRT-2.48, w4a16, ctx 4096, est 4096 MB, provenance "PersonaCare validated bundle")
- `qwen3-4b-genie-concise-baked` (baked concise, class snapdragon-x-elite, readiness unvalidated)
- `qwen3-4b-qnn-s25` (class snapdragon-8-elite, runtime qnn, ctx 4096)
- `qwen3-0.6b-part-a` / `qwen3-0.6b-part-b` (pipeline stages, boundary `hidden[1,32,1024]fp32`, layers 0-14 / 14-28)

Tests: catalog loads; deployment index derives WARM/INSTALLED/ABSENT; overrides win; compatibility class mapping for X Elite / 8 Elite / mock.

### Task 5: DeploymentScheduler (`scheduler.py`)

**Files:**
- Create: `src/dragon_nest/scheduler.py`
- Test: `tests/test_scheduler.py`

```python
@dataclass(frozen=True)
class RequestSpec:
    request_text: str = ""
    base_model_family: str = "mock"
    behavior_profile_id: str = ""
    estimated_input_tokens: int = 256
    estimated_output_tokens: int = 128
    privacy: str = "trusted_fabric"     # trusted_fabric | private
    latency_preference: str = "interactive"  # realtime | interactive | background
    origin_device_id: str = ""
    fallback_policy_override: str = ""  # optional override of profile policy

@dataclass(frozen=True)
class MemoryProjection:
    fixed_runtime_mb: int; artifact_mb: int; kv_cache_mb: int
    boundary_mb: int; safety_margin_mb: int; total_mb: int
    estimated_fields: tuple[str, ...]   # names of components that are estimates

@dataclass(frozen=True)
class CostBreakdown:
    queue_delay_ms: float; cold_load_ms: float; prefill_ms: float; decode_ms: float
    network_ms: float; thermal_battery_penalty_ms: float
    eviction_penalty_ms: float; failure_risk_ms: float; total_ms: float

@dataclass(frozen=True)
class ExecutionCandidate:
    device_id: str; artifact: ArtifactSpec
    realization_mode: str; realization: SteeringRealization | None
    deployment: DeploymentState
    feasible: bool; rejection_reasons: tuple[str, ...]
    memory: MemoryProjection | None; cost: CostBreakdown | None

@dataclass(frozen=True)
class RoutePlan:
    request: RequestSpec
    profile: BehaviorProfile | None
    fallback_policy: str
    candidates: tuple[ExecutionCandidate, ...]   # feasible + rejected
    chosen: ExecutionCandidate | None
    steering: SteeringSpec                       # what to send when runtime_vector
    prompt_prefix: str                           # when prompt_profile
    explanation: tuple[str, ...]                 # ordinary-language route story
    error_code: str = ""                         # BEHAVIOR_UNAVAILABLE | NO_FEASIBLE_DEPLOYMENT | ""
    provisioning_hint: str = ""                  # profile_id to provision when applicable

@dataclass(frozen=True)
class SchedulerConfig:      # explicit deterministic knobs, all overridable
    fixed_runtime_mb: int = 512; safety_margin_mb: int = 384
    boundary_mb: int = 8; default_kv_bytes_per_token: int = 131072
    default_load_time_ms: int = 8000
    default_prefill_tps: float = 120.0; default_decode_tps: float = 12.0
    queue_delay_per_task_ms: float = 1500.0
    thermal_penalty_ms: float = 6000.0          # x thermal_level when > 0.55
    thermal_soft_threshold: float = 0.55
    low_battery_pct: float = 25.0; battery_penalty_ms: float = 4000.0
    eviction_penalty_ms: float = 3000.0
    failure_risk_stale_ms: float = 5000.0; failure_risk_degraded_ms: float = 1500.0
```

`DeploymentScheduler(catalog, behaviors, steering_registry, config)`:
- `plan(request, records: tuple[DeviceRecord, ...], deployments) -> RoutePlan`.
- Candidate generation: for each connected/reachable, non-OFFLINE device × each catalog artifact whose `base_model_family` matches × each allowed realization mode (from the profile fallback ladder, in preference order; `none` mode with empty profile_id yields the plain base-artifact candidate). Baked mode swaps in the baked artifact; other modes use base artifacts.
- Hard constraints (accumulate ALL rejection reasons, don't stop at first): health tier (UNHEALTHY/OFFLINE/unreachable), compatibility class, runtime match (implicit but checked vs advertisement), context fit, deployment readiness (state ∈ {WARM, INSTALLED}; ABSENT/AVAILABLE_REMOTE/FAILED/QUARANTINED rejected with reason), runtime_vector needs: device advertises `supports_steering` for that artifact's model + runtime-steering not disabled by simulation + `steering_registry.runtime_compatible(...)` OK; privacy=private → device==origin; memory projection ≤ available_memory_mb (memory unknown/0 → reject with "telemetry reports unknown memory"); pipeline_stage artifacts excluded from single-deployment scheduling (topology != full_model → rejected "pipeline stages are scheduled by the layer-pipeline planner").
- Scoring feasible candidates with `CostBreakdown`; deterministic tie-break `(total_ms, device_id, artifact_id, mode)`.
- Behavior resolution: candidates are grouped by realization preference order; choose the globally cheapest feasible candidate, but the explanation must state which realization was used and why higher-preference realizations were rejected. If profile requested and NO feasible candidate in any allowed mode → `error_code=BEHAVIOR_UNAVAILABLE`, `provisioning_hint=profile_id`. If no profile and nothing feasible → `NO_FEASIBLE_DEPLOYMENT`.
- `steering` field: populated only for runtime_vector (never for prompt/baked); `prompt_prefix` only for prompt_profile; explanation for prompt says "prompt profile (not activation steering)".

Tests (drive scenarios A–E at unit level):
- warm beats cold when otherwise equal (A)
- runtime-on-phone vs baked-on-laptop chooses per cost and explains the trade-off (B)
- thermal 0.95 → UNHEALTHY rejection; thermal 0.7 → penalty flips winner (C)
- long context: kv projection exceeds phone memory → phone rejected pre-dispatch with explicit memory numbers in the reason; laptop wins (D)
- runtime steering unavailable on device → baked equivalent chosen under allow_baked_equivalent; rejected with exact_only (E)
- medical-safe exact_only with no baked deployment → BEHAVIOR_UNAVAILABLE + provisioning_hint (G precondition)
- never-silent-switch: rejected profile never yields a candidate realizing a *different* profile
- determinism: same inputs → identical plan.

### Task 6: Provisioning state machine (`provisioning.py`)

**Files:**
- Create: `src/dragon_nest/provisioning.py`
- Test: `tests/test_provisioning.py`

```python
class ProvisioningState(StrEnum):
    MISSING="missing"; BUILD_QUEUED="build_queued"; COMPILING="compiling"
    VALIDATING="validating"; READY_REMOTE="ready_remote"; DOWNLOADING="downloading"
    INSTALLED="installed"; WARM="warm"; FAILED="failed"

_TRANSITIONS = {missing→build_queued→compiling→validating→ready_remote→downloading→installed→warm; any_active→failed}

@dataclass
class ProvisioningJob:
    job_id: str; profile_id: str; target_device_id: str; artifact_id: str
    state: ProvisioningState; history: list[tuple[str, float]]; detail: str
    adapter_name: str      # "mock-aihub" — never claims real compilation

class ProvisioningAdapter(Protocol):
    def advance(self, job) -> tuple[ProvisioningState, str]: ...

class MockAiHubAdapter:  # deterministic: one legal transition per advance(); detail
                          # strings always prefixed "[mock]" so the UI cannot claim
                          # a real AI Hub compile happened.
class ProvisioningManager:
    def start(self, profile_id, target_device_id, artifact_id) -> ProvisioningJob
    def advance(self, job_id) -> ProvisioningJob      # one step
    def tick_all(self) -> None                        # advance every active job one step
    def jobs(self) -> tuple[ProvisioningJob, ...]
    def get(self, job_id) -> ProvisioningJob
```
When a job reaches INSTALLED/WARM the manager invokes an `on_deployed(device_id, artifact_id, state)` callback (Brain wires this to a deployment override so the artifact becomes routable).

Tests: legal chain walks missing→warm; illegal jumps raise; failure allowed from active states; mock adapter details are `[mock]`-prefixed; on_deployed fires with INSTALLED and WARM.

### Task 7: BrainService + dashboard API integration

**Files:**
- Modify: `src/dragon_nest/transport/brain.py`, `src/dragon_nest/dashboard.py`, `scripts/run_brain.py`
- Test: `tests/test_dashboard_behavior.py`

BrainService additions (constructor accepts `artifact_catalog`, `behavior_registry`; run_brain wires yaml paths with new CLI flags defaulting to the new configs):
- `self.deployment_overrides: dict[tuple[str, str], ArtifactState]`; `self.runtime_steering_disabled: set[str]` (device_ids); `self.scheduler`, `self.provisioning = ProvisioningManager(MockAiHubAdapter(), on_deployed=...)`; `self.route_plans: dict[str, RoutePlan]`.
- `def build_route_plan(self, spec: RequestSpec) -> RoutePlan` (pure preview).
- `async def submit_behavior_task(self, spec, timeout_ms) -> tuple[RoutePlan, pb.SubmitTaskResponse]`: plan; on error return failed response (BEHAVIOR_UNAVAILABLE / NO_FEASIBLE_DEPLOYMENT) without dispatch; else dispatch single-device via existing `DispatchManager.submit` using candidate order (chosen + remaining feasible device order) — runtime_vector sends SteeringSpec; prompt_profile prefixes the prompt template; baked sends the baked artifact_id as model_id. Store plan under the task_id; reuse `_route_reasons` with `plan.explanation` so the existing Routing Trace panel shows it. Scenario F falls out: mid-flight disconnect → DispatchManager retries next feasible candidate; late results fenced by TaskStore.

Dashboard endpoints:
- `GET /api/behavior-profiles`, `GET /api/artifact-catalog`
- `GET /api/deployments` → per device: artifact states (from DeploymentIndex incl. overrides)
- `POST /api/route-plan` (RequestSpec payload) → full plan JSON (candidates, costs, memory projections with `estimated_fields`, explanation)
- `POST /api/behavior-tasks` → executes; returns task info + plan
- `GET /api/tasks/{id}` gains `route_plan` when present
- `POST /api/provisioning` {profile_id, device_id, artifact_id}; `GET /api/provisioning`; `POST /api/provisioning/{job_id}/advance`
- `POST /api/devices/{id}/simulate` extended: `battery_pct`, `available_memory_mb` already exist; add `artifact_states: dict[str, str]` (artifact_id → absent|installed|warm) and `runtime_steering_enabled: bool | None`.

Tests: route-plan preview returns rejected candidates with reasons; artifact absent simulation flips the decision; runtime steering disable triggers baked fallback; behavior task executes end-to-end on mock agents (httpx ASGI + registered dev-fabric devices); provisioning flow reaches warm and the artifact becomes routable; BEHAVIOR_UNAVAILABLE for family-assistant with provisioning hint; no silent profile switch.

### Task 8: Dashboard UI

**Files:**
- Modify: `src/dragon_nest/web/admin/index.html`, `src/dragon_nest/web/admin/app.js`, `src/dragon_nest/web/admin/admin.css`

- New band "Behavior Routing" (between Task Submission and Routing Trace): request text, base model family select, behavior profile select (from API, "None" option), input/output token numbers, privacy + latency selects, origin device select, buttons **Preview route** and **Route & execute**.
- Candidate table: device / artifact / realization / state / total cost / feasible-or-reason; chosen row highlighted; rejected rows show reasons. Memory projection line with "(est)" badges from `estimated_fields`. Plain-language explanation list. Provisioning hint button "Provision profile…" when `provisioning_hint` set.
- New band "Provisioning": job list with state chips (missing → … → warm), Advance button per job.
- Device cards: chips for steering realization modes (`runtime_vector` when any model supports_steering & not disabled, `baked_profile` when a baked artifact is deployed, `prompt_profile` always) and per-artifact deployment state chips (artifact_id · warm/installed/absent).
- Simulate dialog: add battery slider, memory number, runtime-steering checkbox, and per-artifact state selects for that device.
- Keep existing panels untouched.

Verify: existing `test_dashboard.py` panel assertions still pass; add assertions for "Behavior Routing" and "Provisioning" panels.

### Task 9: Demo fleet fixture + scenario script

**Files:**
- Create: `configs/demo-fleet.yaml`, `scripts/demo_scenarios.py`

`demo-fleet.yaml`: `x-elite-01` (Snapdragon X Elite laptop; hardware.soc_model "Snapdragon X Elite X1E-80-100", platform windows; models: large-reasoning-v1 (warm), small-chat-v1-concise-baked, qwen3-0.6b-part-b, plus small-chat-v1) and `s25-ultra-01` (Samsung Galaxy S25 Ultra; soc_model "Qualcomm SM8750 Snapdragon 8 Elite", platform android, npu_status available; models: small-chat-v1 (warm, supports_steering), qwen3-0.6b-part-a). Both advertise `mock` runtime so the demo runs without SDKs; realistic identity strings drive the compatibility mapping + dashboard.

`scripts/demo_scenarios.py`: self-contained (in-process Brain + two DeviceAgents from demo-fleet.yaml, pattern of `demo_grpc.py`); runs scenarios A–G sequentially, printing the route plan explanation and PASS/FAIL assertion per scenario:
- A warm preference; B behavior locality (runtime on s25 vs baked on x-elite); C thermal reroute via simulation overlay; D memory rejection with a 3000-token request; E runtime-steering disabled → baked/reject per policy; F disconnect mid-task → fenced retry on the other device (uses `simulate_disconnect_on_next_task`); G family-assistant → BEHAVIOR_UNAVAILABLE → provisioning job → advance to warm → reroute succeeds.

### Task 10: Docs + final report

**Files:**
- Create: `docs/BEHAVIOR_SCHEDULER.md` (concise architecture note: object model + request flow diagram), `docs/DEMO_RUNBOOK.md`, `docs/HARDWARE_CONTRACT.md`
- Modify: `README.md` (Current Status + Quick Start additions)

DEMO_RUNBOOK: exact startup commands (Windows venv paths), registration flow expectations, dashboard walkthrough for each scenario incl. expected routing decisions, `scripts/demo_scenarios.py` as the no-hardware fallback, troubleshooting.
HARDWARE_CONTRACT: what Desktop Codex must implement for real QAIRT/QNN/Genie execution — agent advertisement contract (model_id ↔ artifact_id, warm reporting, supports_steering semantics), runtime adapters (GenieExecutor/QnnExecutor interfaces already present in `runtime/executors.py`), baked-artifact build + validation obligations, steering-vector runtime validation obligations, provisioning adapter interface for real AI Hub, and the S25 Genie JNI bridge gap.

Final report in the closing message: files changed, tests run + results, real vs simulated, remaining hardware dependencies, single next action.

## Self-Review notes

- Spec coverage: telemetry (T1), BehaviorProfile/SteeringRealization (T2), vector lifecycle (T3), ArtifactSpec/DeploymentState/DeviceCapability separation (T4 — static capability = advertisement + hardware inventory; dynamic = registry heartbeats; already separate in the codebase), ExecutionCandidate/RouteDecision + hard constraints + cost model + memory projection (T5), fallback policies (T2/T5), provisioning (T6), Brain/dashboard integration + simulations (T7/T8), scenarios A–G (T5 unit + T9 end-to-end), docs (T10). Retry/fencing (F) reuses existing DispatchManager/TaskStore — preserved, not rebuilt.
- No proto changes → Android agent unaffected; gRPC contract untouched.
- Type consistency: scheduler consumes `DeviceRecord` from registry.records() (not just Device) because warm_model_ids/active_task_ids live on the record.

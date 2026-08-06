# Behavior-Aware Deployment Scheduler

The scheduler routes **executable deployments** — a (device, artifact,
behavior realization) triple — not abstract models and not raw amounts of
free memory. This note describes the object model and the request flow.

## Object Model

```text
BehaviorProfile ("concise", "medical-safe", ...)      configs/behavior-profiles.yaml
  |  fallback_policy: exact_only | allow_baked_equivalent | allow_runtime_equivalent
  |                   | allow_prompt_fallback | allow_unsteered | reject
  +-- SteeringRealization (preference-ordered)
        runtime_vector  -> SteeringVector record       configs/steering-vectors.yaml
        baked_profile   -> baked ArtifactSpec          configs/artifact-catalog.yaml
        prompt_profile  -> prompt template (never described as activation steering)
        none

ArtifactSpec (immutable executable artifact / pipeline stage)
  base model + version, baked behavior profile (if any), compatibility classes,
  runtime + version, quantization, context profile, topology + layer range,
  boundary schema, checksum/size, estimated vs measured memory, load time,
  prefill/decode throughput, KV bytes/token, provenance, readiness

DeploymentState (one device x one artifact)
  absent | available_remote | downloading | installed | loading | warm
  | failed | quarantined, plus resident bytes and measured throughput

DeviceCapability (static)          DeviceTelemetry (live)
  SoC identity -> compatibility     free memory, thermal, battery, CPU/load,
  classes, runtime versions,        RTT, active tasks, warm artifacts
  supported steering realization    (DeviceRegistry heartbeats)
  modes (advertisement-derived)
```

- **Modules:** `behavior.py`, `deployments.py`, `scheduler.py`,
  `provisioning.py`, integrated in `transport/brain.py` and `dashboard.py`.
- **`DeploymentIndex`** derives per-(device, artifact) states from device
  advertisements (`ModelCapability.model_id` ↔ `ArtifactSpec.artifact_id`),
  heartbeat warm lists, and Brain-side simulation overrides. No proto changes
  were needed; the gRPC contract is untouched.
- **`SteeringVector`** records carry full lifecycle metadata (extraction
  method, dataset hashes, dtype, evaluation metrics, validated runtimes and
  quantizations, status). A vector is never assumed to transfer across model
  versions, quantizations, runtimes, or layers.

## Request Flow

```text
RequestSpec (model family, behavior profile, token estimates, privacy, latency)
   |
   v
1. Resolve BehaviorProfile -> fallback ladder of admissible realization modes
2. Generate candidates: eligible device x matching artifact x admissible mode
   (baked mode swaps in the baked artifact; other modes use base artifacts;
   baked artifacts of *other* profiles are never candidates)
3. Hard constraints (ALL failures recorded per candidate):
     health tier, connectivity, compatibility class, artifact readiness,
     deployment state (installed/warm only), context fit, privacy/origin,
     runtime-vector validation (device support + vector lifecycle boundaries),
     projected memory <= available:
        fixed runtime + newly-resident artifact + KV bytes/token x total
        tokens + boundary buffers + safety margin
     (unknown values use conservative defaults and are flagged as estimates)
4. Score feasible candidates (deterministic ms-equivalent cost):
     queue delay + cold-load + prefill + decode + network(RTT)
     + thermal/battery penalty + eviction penalty + failure risk
5. Choose the cheapest candidate; emit RoutePlan with every candidate,
   rejection reasons, cost breakdowns, memory projections, and a
   plain-language explanation. Behavior fallback is explicit — the profile is
   never silently switched; if nothing can realize it, the plan returns
   BEHAVIOR_UNAVAILABLE with a provisioning hint.
   |
   v
6. Dispatch (BrainService.submit_behavior_task): existing DispatchManager
   executes the chosen device first, then remaining feasible devices on
   DEVICE_OFFLINE — each with its own per-device realization (runtime steering
   spec, prompt prefix, or baked artifact id). TaskStore fences late results.
   |
   v
7. Provisioning (missing profiles): missing -> build_queued -> compiling ->
   validating -> ready_remote -> downloading -> installed -> warm | failed,
   behind a ProvisioningAdapter. The bundled MockAiHubAdapter labels every
   step "[mock]" so the UI can never claim a real AI Hub compile happened.
```

## Interfaces to the rest of DragonNest

- The classic classifier/planner/router path (`single`, `data_parallel`,
  `layer_pipeline`) is unchanged and still serves `SubmitTask` over gRPC.
- The behavior path is exposed via the dashboard API:
  `POST /api/route-plan` (preview), `POST /api/behavior-tasks` (execute),
  `GET /api/behavior-profiles`, `GET /api/artifact-catalog`,
  `GET /api/deployments`, `POST/GET /api/provisioning...`, and extended
  `POST /api/devices/{id}/simulate` (memory, thermal, battery, load, RTT,
  offline, per-artifact deployment state, runtime-steering enable/disable).
- Real QAIRT/QNN/Genie execution plugs in behind the executor adapters and
  the provisioning adapter; see `docs/HARDWARE_CONTRACT.md`.

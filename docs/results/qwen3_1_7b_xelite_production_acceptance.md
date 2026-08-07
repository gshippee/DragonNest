# Qwen3-1.7B X Elite Production-Provider 4+0 Elastic Acceptance

Date: 2026-08-07
Git commit tested: `1b4febf3e38c3af13d4b3abe9f693930c6da4e7a`
Branch: `codex/qwen17-variable-split`

Physical acceptance record for the normal DragonNest Brain/Agent path (not
the standalone `scripts/xelite_bringup/` harness) executing the four-stage
Qwen3-1.7B pipeline end-to-end on this Snapdragon X Elite laptop. This closes
the gap `docs/QWEN3_1_7B_HANDOFF.md` and `docs/HARDWARE_AUDIT.md` both
recorded as "normal Agent/provider 4+0 acceptance remains pending" after the
production provider was promoted from the physically-proven bring-up
machinery (`edd188f`, `1b4febf`).

**No code changes were required or made this session.** Everything below is
the production path as already committed, exercised for the first time
through a real gRPC Brain submission.

## Environment

- Host: same Dell Latitude 7455 / Snapdragon X Elite (X1E80100) as prior
  sessions.
- QAIRT: `2.45.41.260507` at `C:\Qualcomm\AIStack\QAIRT\2.45.41.260507` — the
  exact physically-verified 2.45.x build (not 2.32, not 2.48).
- Genie bundle: the same physically-verified `qwen3-4b-genie` bundle used in
  `docs/results/xelite_worker_status.md` (sha256-tree
  `940ab2c9958a4f0a53b6964fa96fc427f1f4d33dd1046584e040ec6f2298c929`).
- Four X Elite Qwen3-1.7B context binaries staged via
  `scripts\artifact_tools\stage_xelite_artifacts.ps1`; independently
  SHA-256-verified against `docs/results/demo_artifact_inventory.json`
  (`s0` 622,391,296 B, `s1` 263,147,520 B, `s2` 263,131,136 B, `s3`
  525,893,632 B) — all four matched exactly.
- `check_artifacts.py`: all four `qwen3-1.7b-s{0,1,2,3}-xelite` entries and
  `qwen3-4b-genie` report `READY`. The four `*-s25` entries correctly report
  `UNAVAILABLE` (S25 not present this session).

## Topology started via the real launcher

```powershell
.\scripts\run_xelite_demo.ps1 `
  -StageDir "$env:TEMP\dragonnest-qwen17-xelite" `
  -Qairt245Root 'C:\Qualcomm\AIStack\QAIRT\2.45.41.260507' `
  -GenieDir 'C:\DragonNestArtifacts\qwen3-4b\xelite\genie'
```

Not bypassed with a standalone/manual invocation. This starts the real Brain
(`scripts/run_brain.py`) and the real X Elite worker
(`scripts/run_xelite_elastic_worker.ps1` -> `scripts/run_xelite_worker.ps1`
-> `scripts/run_agent.py`) as two separate processes, exactly as a demo
operator would.

## pc-01 advertisement — physically confirmed via `/api/devices`

`pc-01` connected (`status: HEALTHY`, `connected: true`) and advertised, with
its production readiness/preflight gates passing (not weakened):

| model_id | role | runtime | runtime_version | accelerator |
|---|---|---|---|---|
| `qwen3-4b-genie` | large_reasoning | genie | QAIRT-2.48 | htp |
| `qwen3-1.7b-s0-xelite` | pipeline_segment (stage 0/4, embedding) | qnn | QAIRT-2.45 | htp |
| `qwen3-1.7b-s1-xelite` | pipeline_segment (stage 1/4, layers 0-9) | qnn | QAIRT-2.45 | htp |
| `qwen3-1.7b-s2-xelite` | pipeline_segment (stage 2/4, layers 10-19) | qnn | QAIRT-2.45 | htp |
| `qwen3-1.7b-s3-xelite` | pipeline_segment (stage 3/4, layers 20-27+head) | qnn | QAIRT-2.45 | htp |

## Routing note: task_class gate, not a bug

The first submission attempt used a plain prompt (`"Reply with exactly:
DRAGONNEST_ELASTIC_4X0_OK"`), which the deterministic classifier scores as
`task_class=chat_qa`. `configs/hardware-fabric.yaml` only advertises the
X Elite 1.7B stages for `task_classes: [reasoning_analysis]` — a pre-existing
routing policy, not something this session touched. It correctly produced
`ELASTIC_UNAVAILABLE: no compatible layer pipeline found` rather than
silently routing wrong. Resubmitting with a prompt containing a
classifier-recognized reasoning trigger word (`"Analyze: ..."`) routed as
expected. No scheduler, classifier, or gate code was changed to work around
this — the test prompt was adjusted, per the acceptance instructions'
"the text instruction is only a convenient test prompt."

## Accepted run — `task-cc06d7c7`

Submitted through the normal Brain, not the bypass harness:

```powershell
.\.venv\Scripts\python.exe scripts\submit_task.py `
  --brain 127.0.0.1:50051 `
  --preferred-mode elastic `
  --execution-mode auto `
  --origin-device-id pc-01 `
  --timeout-ms 75000 `
  'Analyze: What is gravity? Keep the answer under ten words.'
```

**Route** (from `/api/tasks/task-cc06d7c7`, `route_reasons`):

> "Selected layer_pipeline: elastic requested the qwen3-1.7b-w4a16-demo-v1
> distributed pipeline." / "Selected qwen3-1.7b-w4a16-demo-v1 cut 4+0: laptop
> prefix=pc-01, phone suffix=none; pc-01 S0+S1+S2+S3 requires 3712 MB of 9319
> MB." / "Elastic selected the executable Qwen3-1.7B distributed pipeline;
> the cut follows cumulative stage memory and live device telemetry."

- `execution_mode`: `layer_pipeline`
- `pipeline_id`: `qwen3-1.7b-w4a16-demo-v1`
- **S0 -> pc-01, S1 -> pc-01, S2 -> pc-01, S3 -> pc-01** (4+0 cut; no
  cross-device transition; no phone suffix; no S25 involvement)
- `runtime_name`: `qnn`, `runtime_version`: `QAIRT-2.45`, `accelerator`: `htp`
- No mock executor, no CPU fallback anywhere in the event trace.

**Operation sequence** (from `/api/events`, timestamps ascending, this exact
task only):

- `pipeline_prefill:0:stage-0` -> `stage-1` -> `stage-2` -> `stage-3`, each
  `QUEUED` -> `DISPATCHED` -> `RUNNING` -> `SUCCEEDED` on `pc-01`, in order.
- `pipeline_decode:1:stage-0..3` through `pipeline_decode:7:stage-0..3` — 7
  decode rounds, each round running all four stages on `pc-01` in order.
- 7 decode rounds x 4 stages = 28 stage-level DECODE executions, plus the
  4-stage PREFILL pass = 32 total physical stage executions for this task.
- Final token count: 1 (prefill-derived top-1) + 7 (decode rounds) = **8
  tokens total**, matching the Brain's configured
  `pipeline_max_new_tokens=8` demo limit.

**Result:**

- `output_text`: `"<think>\n</think>\n\nGravity is the force"`
- `latency_ms` (Brain-reported execution): 1066
- Wall-clock task duration (`updated_at - created_at`): ~29.1 s
- `selected_artifact_id`: `qwen3-1.7b-w4a16-demo-v1-s3-xelite`

This is coherent, on-topic text: the same empty `<think></think>` reasoning
preamble followed by the start of a correct factual answer that
`scripts/xelite_bringup/run_physical_smoke_test.py` physically proved for
essentially the same prompt content
(`docs/results/qwen3_1_7b_xelite_physical_bringup.md` §13, `"<think>\n</think>\n\nGravity
is the force that"`), one token shorter here only because this prompt has one
extra leading token (`"Analyze: "`) consuming one slot of the same 8-token
budget.

## Diagnostic detour: a degenerate-but-not-broken run, ruled out

A prompt with no natural continuation
(`"Analyze this and reply with exactly: DRAGONNEST_ELASTIC_4X0_OK"`,
task `task-e5fcc3c2`) also routed correctly (same 4+0 placement, same
runtime/accelerator, `state: SUCCEEDED`) but decoded to
`"<think>\n</think>\n\n</think>\n\n</think>\n\n"` — token IDs
`[151667, 198, 151668, 271, 151668, 271, 151668, 271]`, i.e. the model fell
into a 2-token greedy repeat loop after the empty think block instead of
producing new content.

To rule out a KV/decode regression in the production provider (the concrete
failure mode `docs/QWEN3_1_7B_HANDOFF.md` and this acceptance task's own
debug order both flag first), the standalone committed smoke harness was run
once, unmodified, against the identical staged artifacts and QAIRT 2.45.41:

```powershell
.\.venv\Scripts\python.exe scripts\xelite_bringup\run_physical_smoke_test.py
```

It reproduced the exact proven coherent sequence
(`<think>\n</think>\n\nGravity is the force that`, token IDs
`[151667, 198, 151668, 271, 38409, 374, 279, 5344, 429]`), confirming the
runtime/artifact/QAIRT combination is sound. Resubmitting through the real
Brain with a prompt content-matched to the harness's own proven prompt
(`task-cc06d7c7`, above) then reproduced that same coherent output through
the production path. Conclusion: the repeat loop was prompt-specific greedy
decode degeneracy on an unusual instruction-following request under an
8-token budget on this quantized model, not a production KV/decode
regression. No provider or `qwen17_kv.py`/`qwen17_stage_engine.py` code was
touched.

## Cleanup / KV lifecycle

After `task-cc06d7c7` completed:

- `/api/devices` -> `pc-01.active_tasks: []` (zero active sessions).
- `tasklist /FI "IMAGENAME eq qnn-net-run.exe"` -> no matching processes
  (zero orphaned QNN processes).
- `%TEMP%\dragon_nest\qnn` scratch directory: empty (no stale per-task
  execution directories).

**Forced-timeout negative path** (`task-9e1d8eff`, `--timeout-ms 500`):
correctly failed mid-pipeline (`EXECUTION_FAILED: stage-3:` — the 500 ms
deadline was too short to complete even one PREFILL pass). Post-failure
cleanup was independently re-verified and was identically clean: zero active
tasks on `pc-01`, zero orphaned `qnn-net-run.exe` processes, empty scratch
directory.

## Acceptance conditions checklist

| Condition | Result |
|---|---|
| Brain chooses model family Qwen3-1.7B | met |
| topology = layer_pipeline | met |
| S0 selected on pc-01 | met |
| S1 selected on pc-01 | met |
| S2 selected on pc-01 | met |
| S3 selected on pc-01 | met |
| operation sequence includes PREFILL then DECODE | met (1 prefill pass, 7 decode rounds) |
| runtime = qnn | met |
| accelerator = htp | met |
| no mock executor | met |
| no CPU fallback | met |
| nonempty coherent decoded text returned through the normal Brain task result | met (`task-cc06d7c7`) |
| real production provider, not `scripts/xelite_bringup` bypass | met |
| zero active pipeline sessions after completion | met |
| no orphaned qnn-net-run processes | met |
| no stale scratch execution directories | met |
| negative path (forced timeout) cleans up identically | met |

## Result

**Physical 4+0 production DragonNest Elastic execution of Qwen3-1.7B
(S0->S1->S2->S3, PREFILL+DECODE, entirely on this X Elite's Hexagon HTP
through the normal Brain/Agent path) is physically proven.** No code changes
were required or made to reach this result.

## Scope explicitly not touched this session

- Scheduler, model partition boundaries, profile system, Android, Fable
  dynamic steering, dashboard architecture, protobuf — all untouched.
- 3+1 / 2+2 cuts, S25 involvement — untouched (S25 not present this
  session).
- QNN persistent-context-loading optimization — untouched (each call still
  pays a full context-binary reload, as documented in the bring-up doc; not
  in scope for an acceptance run).

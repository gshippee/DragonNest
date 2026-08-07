# X Elite Physical Worker Status

Date: 2026-08-07
Git commit tested: `1c46a7b5dc51ff7bbd19619d50f63d7162d86a5e` (origin/main)
Branch: `claude/xelite-real-worker`

This is the sanitized status record for turning a real Snapdragon X Elite
laptop into a verified DragonNest gRPC compute worker. It intentionally
excludes API tokens, licensed SDK files, model binaries, and full local
paths (bundle location is described by relative name only).

## Host identity

- Manufacturer/model: Dell Latitude 7455
- SoC: Snapdragon X Elite (X1E80100), Qualcomm Oryon CPU
- OS: Windows 11 Pro, build 10.0.26100, ARM64-based PC
- Accelerator: "Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Hexagon(TM) NPU"
  present in `Get-PnpDevice`, Status OK

## Runtime actually exercised

- QAIRT/Genie: **2.48.40.260702** (`genie-t2t-run.exe`, `Genie.dll`,
  `QnnHtp.dll` + `QnnHtpv73Stub.dll`/`libQnnHtpv73Skel.so`), sourced from the
  public `qualcomm/qai-appbuilder` GitHub release `v2.48.40`
  (`GenieService-win-arm64.zip`), copied alongside the model bundle so it is
  self-contained.
- The machine's pre-installed QAIRT 2.32.6.250402 SDK's `genie-t2t-run.exe`
  was tried first and reproducibly failed (`use-mmap not supported on
  target`, then context-binary load error `30001`) — a genuine graph-compiler
  version mismatch against context binaries built for QAIRT 2.45+, not a
  bundle or environment problem. This is why 2.48.40 was sourced instead.

## Model / artifact

- model_id: `qwen3-4b-genie`
- artifact_id: `qwen3-4b-w4a16-xelite-v73-qairt248`
- Source: Qualcomm AI Hub Models (`qai-hub-models fetch qwen3_4b -r genie -p
  w4a16 -v 0.55.0 -c qualcomm-snapdragon-x-elite`), classic `genie` runtime
  (not `geniex_qairt`), 2.53GB compressed / ~3.34GB on disk.
- target_compatibility_class: `windows-arm64-x1e-v73-qairt-2.48`
- Tree checksum (sha256-tree, `scripts/hash_artifact.py`):
  `940ab2c9958a4f0a53b6964fa96fc427f1f4d33dd1046584e040ec6f2298c929`

## Results, most-severe-verified-first

| Check | Result | Evidence |
|---|---|---|
| Direct AskQuery-equivalent Genie/HTP smoke | **PHYSICAL VERIFIED** | `genie-t2t-run.exe -c genie_config.json --prompt_file ...`, exit 0, 31019 ms, output `XELITE_NPU_OK`, output-file sha256 `a28f029b662a954d1b9be7e53c1e4b9eb147e90354284ed959ee370b863fc07a` |
| `scripts/probe_hardware.py --execute` | **PHYSICAL VERIFIED** | `artifact_validation.passed=true`, `execution.success=true`, `metrics.runtime_name="genie"`, `metrics.runtime_version="QAIRT-2.48"`, `accelerator="htp"`, output `XELITE_NPU_OK`, latency 21560 ms, proof sha256 `8689e2537ba2fdd290a8fc4c3acf734feb659c704329efe9e619a9d54f09a492` |
| Brain -> local Agent (loopback) -> Genie -> Brain, via normal `/api/tasks` submission | **PHYSICAL VERIFIED** + **CONTROL-PLANE VERIFIED** | task `task-831f9d4c`, routed by the real scheduler (not a manual `ExecutionPlan`), `metrics.runtime_name="genie"`, output `XELITE_BRAIN_LOOPBACK_OK`, latency 13920 ms |
| Brain -> Agent over real LAN interface (not `127.0.0.1`) -> Genie -> Brain | **PHYSICAL VERIFIED** + **CONTROL-PLANE VERIFIED** | task `task-ee113a3b`, agent connected via this machine's LAN IP, output `XELITE_REMOTE_WORKER_OK`, execution latency 20070 ms, client-observed E2E 20249 ms (network/control-plane overhead ~179 ms) |
| Telemetry during execution | **PHYSICAL VERIFIED** | `active_tasks` populated mid-call and cleared after; `available_memory_mb` dropped ~9GB during context load and recovered to ~12.4GB after; no orphaned `genie`/`qnn` processes afterward |
| One-command launcher (`scripts/run_xelite_worker.ps1`) | **PHYSICAL VERIFIED** | Auto-discovered the single bundle candidate, validated checksum, started the agent, which registered `connected=true, status=HEALTHY` |
| `runtime_vector` steering on this artifact | **not attempted / correctly disabled** | `qwen3-4b-genie` advertises `steering_mode: none`; `supports_steering` is not set. No fork/rebuild of Genie attempted, per scope. |

Important honesty note on "remote": both the Brain and the Agent ran on
**this same physical machine** in this session — there was no second machine
available to test a true cross-host deployment. The "remote" proof used this
machine's real LAN network interface (not `127.0.0.1`) so the gRPC transport
path, not just process-local loopback, was exercised. A genuinely separate
Brain host is expected to work identically (the agent only needs a reachable
`host:port`), but that specific claim is **not yet physically verified**.

## Tests

- `pytest -q`: 147 passed, 1 failed — `test_live_brain_sweeper_expires_missed_agent_heartbeat`.
  Reproduced independently 3/3 runs, same assertion, ~0.7s each. This is a
  gRPC-transport heartbeat-timing test unrelated to Genie/hardware code paths;
  it failed identically before any change made in this session and is treated
  as a pre-existing environment-timing issue on this machine, not weakened or
  worked around.
- `scripts/demo_scenarios.py`: all scenarios pass (mock path unaffected).

## Code changed

- `scripts/run_xelite_worker.ps1` (new): one-command launcher. Auto-discovers
  a Genie bundle under `$HOME` if `-GenieDir` is not given, refuses ambiguous
  candidates, validates checksum via the existing artifact contract, prints
  sanitized device/runtime/artifact info, then starts `scripts/run_agent.py`.
  No hardcoded usernames, tokens, proprietary bundle contents, or absolute
  local paths.
- No changes to `src/dragon_nest/**`, `configs/**`, or the OpenAI-compatible
  HTTP endpoint adapter (`run_openai_adapter.py` and friends were not
  touched, per the task's architecture boundary).

## Known limitations / remaining blockers

- Genuinely separate-machine Brain<->Agent has not been physically exercised
  (see honesty note above) — only same-machine-via-LAN-IP.
- `runtime_vector` steering remains correctly unsupported/undeclared for this
  artifact; no attempt was made to change that (out of scope, per instructions).
- The QAIRT 2.48.40 Genie/HTP binaries used here came from the public
  `qai-appbuilder` GitHub release rather than Qualcomm Package Manager; they
  are copied into the (gitignored) bundle directory only, not committed.
- `docs/HARDWARE_AUDIT.md`'s prior "Current blockers" section still describes
  the X Elite laptop as unattached; that document was written from a
  different (non-X-Elite) machine and predates this session's physical run.

## AskQuery secondary-capability follow-up (documentation only, no implementation)

AskQuery proves this X Elite machine can later advertise task-specific local
capabilities such as ASR/OCR/TTS (Whisper, EasyOCR, MeloTTS) in addition to
`qwen3-4b-genie`.

Those capabilities should eventually be represented explicitly in DragonNest's
device/artifact model — e.g. distinct capability/model entries per stage
(`whisper-base-onnx`, `easyocr-detector-recognizer`, `melotts-en`) — rather
than pretending they are LLM model IDs or squeezing them into the existing
`runtime: genie` / `runtime: qnn` shape. No implementation of this was done
in this milestone; this is the follow-up note requested for the next X Elite
task.

## Exact command for demo day

```powershell
.\scripts\run_xelite_worker.ps1 -Brain <brain-host>:50051
```

Optional: `-GenieDir <path>` to skip auto-discovery, `-EnrollmentToken
<token>` (or set `$env:DRAGONNEST_ENROLLMENT_TOKEN`) instead of the insecure
dev default.

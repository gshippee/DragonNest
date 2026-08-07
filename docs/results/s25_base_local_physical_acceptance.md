# Galaxy S25 Ultra Base Compute=Local Physical Acceptance

Date: 2026-08-07
Branch: `codex/qwen17-variable-split`

Physical acceptance record for `Compute = Local, Profile = Balanced` on the
real Galaxy S25 Ultra, through the normal DragonNest Brain path (not a
GenieX bypass smoke test). This is the first physical proof that
`android-mock-v1` has been removed from the real phone path: the hardware
APK never compiles the mock executor in, and the device advertised exactly
one real, checksum-verified, GenieX-admitted model.

## Base bundle recovery (prerequisite)

The local artifact cache (`C:\DragonNestArtifacts\qwen3-0.6b\s25\base`) was
empty at the start of this session. The complete Base bundle was restored
from two independent sources and verified against
`docs/results/s25_geniex_artifacts.json` using the committed verifier
(`scripts/artifact_tools/stage_s25_geniex_artifacts.py`, `digest_tree`), not
a second checksum algorithm:

- `part1_of_2.bin` / `part2_of_2.bin`: identified by hash (not by job name —
  the plainly-named AI Hub link jobs `qwen3_0_6b_w4a16_qwen3_0_6b_w4a16_part{1,2}_of_2`
  turned out to be a *different*, non-matching version) as the linked outputs
  of AI Hub jobs `jgll1k78g` (`personacare-base-part1-token-prompt-repair`)
  and `j56wd1v0g` (`personacare-base-part2-token-prompt-repair`). Job
  metadata confirmed QAIRT `2.45.0.260326154327`, HTP `v79`, graphs
  `prompt_ar128_cl512_1_of_2` + `token_ar1_cl512_1_of_2`.
- Seven sidecar files (`config.json`, `genie_config.json`,
  `htp_backend_ext_config.json`, `metadata.json`, `sample_prompt.txt`,
  `tokenizer.json`, `tokenizer_config.json`): transferred from Desktop Codex
  via the encrypted archive committed at `coordination/xelite/s25-base-sidecars-aes256.zip`
  (SHA-256 `b2267698a26747ad8bbff95fdb2295dc97ddbf033678df77c7d7e3788f6ae12a`,
  verified before extraction).

Full tree verification:

```
digest_tree(C:\DragonNestArtifacts\qwen3-0.6b\s25\base)
  = efc5728ba3ac7ee4a5bc2ee7fc8aaad8e875d66625234d71180dcec74a695827
```

Matches `docs/results/s25_geniex_artifacts.json`'s
`qwen3-0.6b-s25-base.sha256_tree` exactly, as do all nine individual file
hashes. Concise and Detailed bundles were not recovered this session and
remain absent from the cache — this acceptance run does not depend on them
(see "Tooling improvement" below).

## Tooling improvement: explicit model_id subset

`scripts/artifact_tools/stage_s25_geniex_artifacts.py` previously required
the inventory to resolve to exactly Base+Concise+Detailed, so a cache with
only Base present couldn't be verified or provisioned at all. Added
`--model-id` (repeatable; default unchanged: all three), extracted into a
testable `select_records()` helper, plus `deploy_s25_local_artifacts.ps1
-Profiles` (default `Base,Concise,Detailed`). See
`tests/test_s25_geniex_artifacts.py` for regression coverage.

## Hardware APK

Built this session via `scripts/build_s25_local_demo.ps1`
(`-PincludeS25GenieXRuntime=true`): GenieX 0.3.5 native runtime closure
packaged (`libgeniex.so`, `libQnnHtp*.so`, `libggml-htp-v79.so`, etc.),
`DRAGONNEST_ENABLE_MOCK_RUNTIME=false` confirmed directly from the generated
`BuildConfig.java`. `:app:testDebugUnitTest` and `:app:assembleDebug` both
`BUILD SUCCESSFUL`.

- APK SHA-256: `a255e595aede386ebc2b1862e673429d2245117e5b441e42a1b9e7d770f96fdb`
- Size: 107,190,761 bytes
- applicationId: `com.dragonnest.agent`

Installed on the physical device with `adb install -r` (the prior install
was a differently-signed build from earlier work; it was uninstalled once,
explicitly authorized, to resolve the signature mismatch — not part of the
committed acceptance script's default behavior, which never uninstalls
first).

## Provisioning — Base only

```
.\scripts\deploy_s25_local_artifacts.ps1 `
    -CacheRoot C:\DragonNestArtifacts\qwen3-0.6b\s25 `
    -Serial R3CXC0805HW `
    -Profiles Base
```

Source hashes verified before phone mutation; app-private phone hashes
re-verified after copy (both via the committed `stage_s25_geniex_artifacts.py`
machinery, unmodified). Manifest written to the phone contains only
`qwen3-0.6b-s25-base` — the device advertises what it actually has, not a
fabricated full catalog.

## Device advertisement — physically confirmed via `/api/devices`

| field | value |
|---|---|
| `device_id` | `android-13cda486-3208-4125-8966-9b6f3e3f83a0` |
| `status` / `connected` | `HEALTHY` / `true` |
| `models` | exactly `["qwen3-0.6b-s25-base"]` |
| `qwen3-0.6b-s25-base.runtime` | `genie` |
| `qwen3-0.6b-s25-base.runtime_version` | `GenieX-0.3.5 / QAIRT-2.45` |
| `qwen3-0.6b-s25-base.accelerators` | `["htp"]` |
| `qwen3-0.6b-s25-base.steering_modes` / `behavior_profile_ids` | `["none"]` / `[]` |
| `hardware.soc_model` / `npu_status` / `npu_name` | `SM8750` / `available` / `Qualcomm HTP` |
| `hardware.compatibility_key` | `android-arm64-v8a-sm8750-v79-qairt-2.45-geniex-0.3.5` |
| `deployments[qwen3-0.6b-s25-concise].state` | `absent` (not fake/unavailable-as-installed) |
| `deployments[qwen3-0.6b-s25-detailed].state` | `absent` |

`android-mock-v1` never appears — the hardware build never compiles the mock
executor in (`DRAGONNEST_ENABLE_MOCK_RUNTIME=false`), so there is nothing to
exclude at runtime; it simply cannot exist in this build.

## Enrollment note

The phone's previously-stored enrollment credential (from before this
session's `adb uninstall`/reinstall) was rejected by the currently-running
Brain process (`invalid enrollment token or expired credential` — a stale
credential against this Brain instance's live in-memory enrollment state,
not a code defect). Re-enrollment requires the Compose UI or a QR scan
because `EnrollmentStore`/`UserProfileStore` are Android-Keystore-encrypted
on device with no adb-writable equivalent — resolved via a fresh session
created through the Brain's existing `POST /api/enrollment-sessions` API
(the same one the dashboard's own enrollment UI calls) and completed by the
device operator scanning the resulting QR code. No second enrollment
protocol was invented.

## Accepted run — `task-1eeca72f`

Submitted through the normal Brain gRPC path, not a GenieX bypass:

```
python scripts\submit_task.py "What is the capital of Japan?" \
    --brain 127.0.0.1:50051 \
    --preferred-mode local \
    --origin-device-id android-13cda486-3208-4125-8966-9b6f3e3f83a0 \
    --persona-id balanced \
    --timeout-ms 60000
```

**Route** (`/api/tasks/task-1eeca72f`, `route_reasons`):

> "Selected single: request does not require parallel execution." /
> "qwen3-0.6b-s25-base supports chat_qa; thermal=0.00, memory=5193 MB;
> steering disabled" / "Origin preference selected
> android-13cda486-3208-4125-8966-9b6f3e3f83a0: compatible local capacity is
> available." / "Local selected qwen3-0.6b-s25-base on origin
> android-13cda486-3208-4125-8966-9b6f3e3f83a0; remote fallback is
> prohibited." / "Local mode restricted routing to origin
> android-13cda486-3208-4125-8966-9b6f3e3f83a0; remote devices were excluded
> by policy." / "Excluded android-a42dae60-87d1-4ff3-abbb-51ffbe428b53:
> device is OFFLINE."

- `preferred_mode`: `local`, `execution_mode`: `single`
- `behavior_profile_id`: `balanced`, `profile_realization`: `none`
- `selected_artifact_id`: `qwen3-0.6b-w4a16-sm8750-base-qairt245`
- `steering.enabled`: `false`, `steering.behavior_profile_id`: `""`
- No remote escape to `pc-01` (X Elite): request stayed on the origin device
  for its entire routing decision.

**Result:**

- `output_text`: `"Japan's capital is Tokyo."` — nonempty, coherent, correct.
- `metrics.runtime_name`: `genie`, `runtime_version`: `GenieX-0.3.5 / QAIRT-2.45`,
  `accelerator`: `htp`
- `latency_ms`: 1163 (single-attempt execution latency; consistent with
  wall-clock `updated_at - created_at` = 1.36 s for this task)
- `error_code`: `""`, `state`: `SUCCEEDED`

## Cleanup

- `/api/devices` after completion: `active_tasks: []`, `status: HEALTHY`.
- On-device diagnostic log (`client-debug.xml`) shows a clean lifecycle:
  `Connecting to Brain` -> `Brain returned RegistrationAccepted` -> (task
  execution) -> no error events after registration.
- `DragonNestRuntime` / `DragonNestGenieX` logcat: no warnings/errors for
  this run (those tags only log rejections; silence for an admitted,
  successful model is the expected positive signal — see
  `AndroidRuntimeCatalog.java`/`GenieXRuntimeBridge.kt`).

## Acceptance conditions checklist

| Condition | Result |
|---|---|
| preferred compute = Local | met |
| selected device = S25 (real physical device) | met |
| selected model = `qwen3-0.6b-s25-base` | met |
| profile = Balanced | met |
| realization = base/none, not steering | met |
| runtime = GenieX / genie | met |
| accelerator = HTP | met |
| nonempty coherent output | met (`"Japan's capital is Tokyo."`) |
| no mock | met (mock not compiled into this build) |
| no remote escape to X Elite | met |
| no `PROFILE_UNAVAILABLE` | met |
| task cleans up normally | met |

## Result

**Balanced Local is now physically proven end-to-end** on the real Galaxy
S25 Ultra through the normal Brain -> S25 -> `qwen3-0.6b-s25-base` -> GenieX
0.3.5 / QAIRT 2.45 -> HTP path, driven by `scripts/submit_task.py` (the
normal Brain gRPC surface, not a bypass). PersonaCare UI cross-check was not
performed by the assistant (no screen-automation tooling available this
session; the task explicitly disallows brittle screen-coordinate
automation) — recommended as a quick manual follow-up by the device
operator.

## Scope explicitly not touched this session

- Concise/Detailed bundle recovery — deferred; Balanced Local acceptance did
  not block on them, per instructions.
- Fable dynamic steering, S25 1.7B direct-QNN, 3+1/2+2 cuts, model split
  boundaries, profile semantics, `main` — all untouched.
- `com.personacare.steeringdemo` / `com.dragonnest.geniexsteeringlab` — not
  touched.

# Qwen3-1.7B four-stage demo handoff

## Evidence boundary

Pipeline `qwen3-1.7b-w4a16-demo-v1` has eight downloaded, checksummed QNN
context binaries: S0-S3 for S25/v79 and X Elite/v73. Qualcomm AI Hub
interface comparison passed for every prompt/decode graph on both targets.
The reproducible `scripts/xelite_bringup/` harness has physically proven the
X Elite v73 S0-S3 prefill, eight autoregressive decode steps, stage-local KV,
HTP-only execution, and coherent output with QAIRT 2.45. That proves the
runtime algorithm, not the new normal Agent/provider path. The latter remains
pending the 4+0 acceptance below. S25 S0-S3 remains unexecuted.

Artifact names say `w4a16`; recovered compile options say
`--quantize_full_type w8a16 --quantize_io`. The mismatch is unresolved and is
preserved verbatim in the inventory and runtime manifest.

## Fixed graph contract

| Stage | Transformer ownership | Input → output | Local state |
|---|---|---|---|
| S0 | none; embeddings only | `input_ids` → `embedding` | none |
| S1 | layers 0-9 | `embedding` → `add_21844` | KV 0-9 |
| S2 | layers 10-19 | `add_21844` → `add_42314` | KV 10-19 |
| S3 | layers 20-27 + norm/head | `add_42314` → `logits` | KV 20-27 |

Prompt sequence is 128, decode sequence is 1, context is 512, and hidden size
is 2048. S0 owns tokenization once for prefill. S3 owns deterministic top-1
sampling and returns one explicit `next_token_id`; the Brain sends that ID to
S0 for the next decode pass. KV tensors remain in Agent-local sessions keyed by
`task_id/pipeline_id/stage_index`; only the named activation crosses gRPC.

## Calibratable memory estimates

These are conservative routing inputs, not measurements.

| Stage | S25 bytes | X Elite bytes | `min_memory_mb` |
|---|---:|---:|---:|
| S0 | 622,391,296 | 622,391,296 | 1024 |
| S1 | 263,028,736 | 263,147,520 | 768 |
| S2 | 263,024,640 | 263,131,136 | 768 |
| S3 | 525,799,424 | 525,893,632 | 1152 |

The router sums all stages assigned to a device. Its legal first-demo cuts are:

- phone ≥1920 MB: laptop S0+S1 | phone S2+S3;
- phone ≥1152 MB but <1920 MB: laptop S0+S1+S2 | phone S3;
- phone <1152 MB: laptop S0+S1+S2+S3.

Every route has at most one laptop→phone transition. A per-stage fallback is
not allowed because it could invalidate cumulative memory or create pc→phone→pc.

## Desktop proof (no Snapdragon execution)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\demo_scenarios.py

$env:JAVA_HOME = 'C:\path\to\jdk-17'
$env:ANDROID_HOME = 'C:\path\to\android-sdk'
Push-Location android-agent
.\gradlew.bat testDebugUnitTest assembleDebug --no-daemon
Pop-Location
```

The public gRPC regression submits a normal `layer_pipeline` task and proves
PREFILL→S0/S1/S2/S3, DECODE→S0/S1/S2/S3, explicit token sampling, and RESET
cleanup. It uses mock executors and is control-plane evidence only.

## X Elite production-provider acceptance

No model download or compilation is required. On the laptop:

```powershell
git switch codex/qwen17-variable-split
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -e ".[dev,xelite]"
.\scripts\artifact_tools\stage_xelite_artifacts.ps1 `
  -CacheRoot C:\DragonNestArtifacts `
  -StageDir "$env:TEMP\dragonnest-qwen17-xelite"
```

Start Brain plus the production provider. `StageDir` is the directory created
above and `Qairt245Root` is the physically proven QAIRT 2.45 installation:

```powershell
$env:DRAGONNEST_ENROLLMENT_TOKEN = '<random-shared-token-at-least-24-chars>'
.\scripts\run_xelite_demo.ps1 `
  -StageDir "$env:TEMP\dragonnest-qwen17-xelite" `
  -Qairt245Root 'C:\Qualcomm\AIStack\QAIRT\2.45.0.260326' `
  -GenieDir 'C:\path\to\physically-verified-qwen3-4b-bundle'
```

The launcher preflights the public Qwen3-1.7B tokenizer/config before it
advertises the stages. The first run may fetch those public files from Hugging
Face, matching the physically proven harness. For an offline demo, download
them once and add `-Qwen17Tokenizer 'C:\path\to\local\Qwen3-1.7B'`; the same
source is used for tokenization and RoPE configuration.

Then exercise DragonNest's normal Brain submission path, not the standalone
smoke harness:

```powershell
.\.venv\Scripts\python.exe scripts\submit_task.py `
  --brain 127.0.0.1:50051 `
  --preferred-mode elastic `
  --execution-mode auto `
  --origin-device-id pc-01 `
  --timeout-ms 75000 `
  'Reply with exactly: DRAGONNEST_ELASTIC_4X0_OK'
```

Accept only if the dashboard/task proof shows `layer_pipeline`, pipeline
`qwen3-1.7b-w4a16-demo-v1`, S0/S1/S2/S3 all on `pc-01`, runtime `qnn`,
accelerator `htp`, nonempty real decoded text, no mock executor, and no
remaining provider sessions after success. Repeat RESET, CANCEL, timeout, and
disconnect cleanup checks before marking the production provider physically
verified. The Qwen3-4B Genie capability remains separately advertised and its
working execution semantics are unchanged.

## S25 physical handoff

Build and install the thin APK without model assets, then provision the four
contexts separately:

```powershell
$env:JAVA_HOME = 'C:\path\to\jdk-17'
$env:ANDROID_HOME = 'C:\path\to\android-sdk'
Push-Location android-agent
.\gradlew.bat assembleDebug --no-daemon
adb install -r app\build\outputs\apk\debug\app-debug.apk
Pop-Location

.\scripts\deploy_s25_demo_artifacts.ps1 `
  -CacheRoot C:\DragonNestArtifacts `
  -AdbPath "$env:ANDROID_HOME\platform-tools\adb.exe"
```

The wrapper requires exactly one expected S25 and a debuggable
`com.dragonnest.agent`, verifies all four source and installed SHA-256 values,
writes `files/dragonnest-models/manifest.json`, and force-stops the app. The
checked-in bridge deliberately reports `RUNTIME NOT YET EXECUTABLE` and does
not advertise those stages until its native execution-ready gate is flipped
after physical validation. Package the licensed arm64 QAIRT 2.45 QNN libraries
with the explicit hardware-build flag, relaunch PersonaCare, and only then
confirm all four models advertise installed/cold. Run 2+2, 3+1, and 4+0 requests and verify
nonempty decoded text, HTP metrics, no second device transition, and zero
stage sessions after success, failure, cancellation, and disconnect.

Ordinary UI APK rebuilds never repackage the 1.67 GB contexts. Production
artifact delivery remains a future Brain-directed download path.

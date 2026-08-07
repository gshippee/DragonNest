# Qwen3-1.7B four-stage demo handoff

## Evidence boundary

Pipeline `qwen3-1.7b-w4a16-demo-v1` has eight downloaded, checksummed QNN
context binaries: S0-S3 for S25/v79 and X Elite/v73. Qualcomm AI Hub
interface comparison passed for every prompt/decode graph on both targets.
Neither four-stage target has physically executed this exact recovered pipeline
through DragonNest. Do not report physical latency, memory, determinism, or
QAIRT 2.45-context/2.48-runtime compatibility until the commands below pass.

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

## X Elite physical handoff

No model download or compilation is required. On the laptop:

```powershell
git pull
.\scripts\artifact_tools\stage_xelite_artifacts.ps1 `
  -CacheRoot C:\DragonNestArtifacts `
  -StageDir "$env:TEMP\dragonnest-qwen17-xelite"

# Copy the four environment-variable commands printed by the staging helper.
.\.venv\Scripts\python.exe scripts\check_artifacts.py
```

Then complete the physical-only gate in this order:

1. Bind `qnn_jni.cpp`/the Python QNN stage adapter to QAIRT provider, backend,
   device, context-deserialization, graph selection, named I/O, and persistent
   per-stage KV buffers. The checked-in JNI methods deliberately fail rather
   than return fake execution.
2. Prove one S0 prompt and decode invocation against QAIRT 2.48. If the 2.45
   context does not load, use a matching 2.45 runtime; do not relabel it.
3. Repeat S1-S3, verify the declared boundary names/dtypes/shapes, then run the
   whole local four-stage loop for two tokens.
4. Only after that proof, start the combined Agent with both compatibility
   classes explicitly enabled:

```powershell
.\.venv\Scripts\python.exe scripts\run_agent.py `
  --device-id pc-01 `
  --brain 127.0.0.1:50051 `
  --enrollment-token "$env:DRAGONNEST_ENROLLMENT_TOKEN" `
  --fabric configs\hardware-fabric.yaml `
  --artifact-manifest configs\model-artifacts.yaml `
  --compatibility-key windows-arm64-x1e-v73-qairt-2.48 `
  --compatible-target-class windows-arm64-x1e-v73-qairt-2.45 `
  --runtime-name qnn+genie `
  --runtime-version QAIRT-2.48 `
  --accelerator-available
```

The existing `scripts/run_xelite_worker.ps1` and verified Qwen3-4B Genie path
remain unchanged.

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

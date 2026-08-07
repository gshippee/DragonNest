# PersonaCare Android Client / DragonNest Agent

This module is the integrated PersonaCare Android client. It builds one APK
containing the PersonaCare Compose UI and the long-lived DragonNest Device
Agent/runtime layer. It includes:

- a `START_STICKY` foreground service and boot receiver;
- a persistent bidirectional gRPC connection generated from
  `../proto/dragonnest.proto`;
- registration, heartbeat/RTT, single-task, shard, pipeline-stage, result, and
  cancellation handling;
- a pluggable `AndroidTaskExecutor` with an MVP `MockAndroidTaskExecutor`;
- an exponential reconnect loop;
- a `ConnectivityManager.NetworkCallback` that requests an immediate heartbeat;
- QR enrollment with schema, address, and expiry validation;
- bootstrap and device credential encryption using an Android Keystore AES-GCM
  key;
- best-effort graceful shutdown;
- Android battery, charging, thermal, and memory telemetry with explicit unknown
  values for unsupported CPU and accelerator metrics;
- automatic static hardware registration for model/SoC, Android version, ABIs,
  CPU cores, storage, and QNN/NPU probe status;
- settings for Brain address, enrollment token, TLS, and simulated offline,
  battery, thermal, CPU, accelerator, and RTT state.

The normal thin debug build includes `android-mock-v1` for control-plane tests.
The explicit S25 GenieX hardware build disables that capability, so a physical
runtime error can never become a mock response. Real capabilities are
registered only when the installed manifest is valid, every artifact checksum
passes, the target matches, and the corresponding vendor bridge successfully
loads the artifact. The Brain remains authoritative for task and attempt state.

The recovered Base, Concise, and Detailed cache entries each contain both
`prompt_ar128` and `token_ar1` graphs. They are complete autoregressive
bundles; no AI Hub compilation is pending. Prompt-only release copies are
preserved as invalid evidence outside the active cache and are rejected by the
provisioning inventory.

## QNN and Genie Runtime Builds

`scripts/build_android.sh` with no extra environment variables builds the thin,
open-source APK: no vendor runtime `.so` libraries and no model assets, even if
`android-agent/vendor/` happens to be populated. Including the model and
runtime is an explicit opt-in for hardware builds
(`DRAGONNEST_ANDROID_INCLUDE_MODEL_ARTIFACTS=true`, see below).

The base repository cannot include Qualcomm SDK libraries or model bundles. A
real S25 build stages the matching SDK libraries and model bundle under
`android-agent/vendor/` before Gradle runs:

```text
vendor/jniLibs/arm64-v8a/*.so
vendor/model-assets/models/manifest.json
vendor/model-assets/models/<model artifacts>
```

For QAIRT 2.48 Genie bundles, the app retains a direct native
`com.dragonnest.agent.vendor.GenieRuntimeBridge`. It creates a Genie dialog
from the verified bundle, probes it during registration, and runs prompt text
through the Genie callback API on HTP. The runtime is omitted from a normal
open-source build and never advertised until the bundled libraries and model
checksum both pass.

For the physically validated Qwen3-0.6B S25 bundles, use the separate explicit
GenieX 0.3.5 / QAIRT 2.45 build. Qualcomm libraries are pulled from the licensed
local Maven artifact only for that build; they and the model bytes are never
committed:

```powershell
$env:JAVA_HOME = 'C:\path\to\jdk-17'
$env:ANDROID_HOME = 'C:\path\to\android-sdk'
Push-Location android-agent
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug `
  -PincludeS25GenieXRuntime=true --no-daemon
Pop-Location
adb install -r android-agent\app\build\outputs\apk\debug\app-debug.apk
.\scripts\deploy_s25_local_artifacts.ps1 `
  -CacheRoot C:\DragonNestArtifacts\qwen3-0.6b\s25 `
  -AdbPath "$env:ANDROID_HOME\platform-tools\adb.exe"
```

The deployment helper requires exactly one debuggable SM8750 S25, verifies
the committed per-file and tree SHA-256 inventory before its first phone
mutation, verifies the temporary and installed copies, and writes only under
the scoped app-private external-files store for `com.dragonnest.agent`. QAIRT
requires this real filesystem path for mmap/HTP loading. Ordinary APK rebuilds
remain about the size of code plus the licensed runtime closure and never
contain the roughly 2 GB of Base/Concise/Detailed model bytes.

Stage the output of the S25 AI Hub export with the matching SDK root:

```bash
scripts/prepare_android_genie_runtime.sh \
  --qairt-sdk /path/to/qairt/2.48.0.260626 \
  --bundle /path/to/qwen3_1_7b-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite_for_galaxy
DRAGONNEST_QAIRT_SDK_ROOT=/path/to/qairt/2.48.0.260626 \
  DRAGONNEST_ANDROID_INCLUDE_MODEL_ARTIFACTS=true \
  scripts/build_android.sh
```

The staging script generates a tree-hashed Android manifest and copies only
the required arm64 Genie/QNN libraries. It refuses to overwrite an existing
staged bundle. The model is several gigabytes, so it is deliberately excluded
from Git and should be delivered through a release asset or trusted artifact
distribution rather than source control. Context `.bin` files are packaged
uncompressed because Genie memory-maps them on device.

The optional AAR route remains available for `QnnRuntimeBridge`. It must
implement `AndroidRuntimeBridge`; no model is routed until the bridge probe
returns true.

The bundled model manifest is JSON, not the host YAML manifest. It must contain
the artifact contract plus routing metadata. Paths are relative to `models/` and
may not escape it:

```json
{
  "models": [{
    "model_id": "qwen3-s25-qnn",
    "model_version": "s25-validated-build",
    "runtime": "qnn",
    "artifact_path": "qwen3/model.bin",
    "checksum": "sha256:<64 lowercase hex characters>",
    "tokenizer_id": "Qwen/Qwen3-0.6B",
    "precision": "fp16",
    "supported_accelerators": ["htp"],
    "min_memory_mb": 1024,
    "max_context_tokens": 128,
    "supports_steering": false,
    "supports_data_parallel": true,
    "supports_layer_pipeline": false,
    "model_family": "qwen3",
    "role": "small_chat",
    "task_classes": ["chat_qa", "summarization"],
    "quality_score": 0.8,
    "runtime_version": "<QNN or Genie runtime version>"
  }]
}
```

For a layer-pipeline QNN artifact, set `supports_layer_pipeline` to `true` and
add `split_boundary` with `pipeline_id`, `stage_index`, `stage_count`, optional
`transformer_start_layer`/`transformer_end_layer`, `input_tensor`,
`output_tensor`, `includes_embedding`, `includes_lm_head`, and
`boundary_format`. The transformer interval is intentionally absent for an
embedding-only stage. Every stage must be compiled for the S25 target, use the
same tokenizer/model version/precision contract, and share exact adjacent
tensor interfaces. Existing Snapdragon X Elite artifacts must not be treated
as S25 artifacts.

The four Qwen3-1.7B contexts are not APK assets. For the debug demo, install the
thin APK first and run `scripts/deploy_s25_demo_artifacts.ps1 -CacheRoot
<external-cache>` from the repo root. It verifies the external and app-private
hashes via `run-as`,
installs the manifest under the scoped app-private `dragonnest-models` store, and forces a clean
catalog reload. See `docs/QWEN3_1_7B_HANDOFF.md`.

At first launch the Agent copies optional packaged model assets into scoped private storage,
then validates them. The Agent reports `npu_status=available` only after a
verified HTP/NPU model and runtime bridge are usable. Otherwise it reports
`unavailable`; a thin build may advertise the mock model, while the explicit
hardware build advertises no model rather than falling back to mock.

Build prerequisites are JDK 17 and Android SDK 35:

```bash
scripts/build_android.sh
```

The build script keeps Gradle and Android user state under
`/tmp/dragonnest-toolchain` by default. The output is:

```text
android-agent/app/build/outputs/apk/debug/app-debug.apk
```

Install and launch on a connected device:

```bash
adb install -r android-agent/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.dragonnest.agent/.AgentSettingsActivity
```

Start the Brain with a network-visible gRPC listener:

```bash
.venv/bin/python scripts/run_brain.py \
  --address 0.0.0.0:50051 --http-host 0.0.0.0
```

Open `http://<brain-host>:8080`, select **Add device**, enter the Brain host's
LAN address, and generate the QR. Select **Scan enrollment QR** in the APK and
confirm the address. The QR expires after five minutes and can enroll only one
device ID. After registration, the Brain returns a device-specific credential
and the APK replaces the bootstrap secret in Keystore for reconnects.

The dashboard registration fields are persisted by the Brain as a personal
profile and associated with the device after the QR is claimed. Profile
steering is a Brain policy; the QR contains only the profile ID, not the
steering vector itself.

In the APK, use the host machine's LAN address for a physical device. The
Android emulator uses `10.0.2.2`. Dev mode defaults to port `50051`, enrollment
token `dev-token`, and plaintext transport. Enable TLS only when the Brain
certificate is trusted by Android.

QR enrollment currently operates only in Brain development mode. It removes
the need to type or distribute the shared token, but it is not a substitute for
production client-certificate provisioning.

The APK does not claim HTP/NPU availability from device branding alone. A
hardware build reports the admitted GenieX/QAIRT bridge version only after an
installed artifact passes checksum, target, runtime-load, and execution-ready
gates.

The implementation was verified on an API 35 x86_64 emulator against the Python
Brain: registration and live telemetry succeeded, a remote single task returned
an Android result, and three data-parallel shards were executed over the same
persistent stream.

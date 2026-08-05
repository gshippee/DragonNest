# DragonNest Android Agent

This module builds a runnable Android Device Agent APK. It includes:

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

The debug build always includes `android-mock-v1`. It reports normalized
execution metrics and is useful for testing the complete gRPC lifecycle. Real
QNN and Genie capabilities are registered only when the APK contains a valid
artifact manifest, every artifact checksum passes, and the corresponding vendor
runtime bridge can load the artifact on the target device. The Brain remains
authoritative for task and attempt state.

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

For QAIRT 2.48 Genie bundles, the app has a direct native
`com.dragonnest.agent.vendor.GenieRuntimeBridge`. It creates a Genie dialog
from the verified bundle, probes it during registration, and runs prompt text
through the Genie callback API on HTP. The runtime is omitted from a normal
open-source build and never advertised until the bundled libraries and model
checksum both pass.

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
add `split_boundary` with `pipeline_id`, `start_layer`, `end_layer`,
`total_layers`, `includes_embedding`, `includes_lm_head`, and `boundary_format`.
Both stages must be compiled for the S25 target, use the same tokenizer/model
version/precision, and share an exact boundary format. Existing Snapdragon X
Elite artifacts must not be treated as S25 artifacts.

At first launch the Agent copies packaged model assets into private storage,
then validates them. The Agent reports `npu_status=available` only after a
verified HTP/NPU model and runtime bridge are usable; otherwise it reports
`unavailable` and advertises only the mock model.

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

The APK reports `npu_status=not_probed` until a QNN or Genie Android runtime is
integrated. It does not claim HTP/NPU availability from device branding alone.

The implementation was verified on an API 35 x86_64 emulator against the Python
Brain: registration and live telemetry succeeded, a remote single task returned
an Android result, and three data-parallel shards were executed over the same
persistent stream.

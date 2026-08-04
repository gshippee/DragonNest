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
- settings for Brain address, enrollment token, TLS, and simulated offline,
  battery, thermal, CPU, accelerator, and RTT state.

The mock executor advertises `android-mock-v1` and reports normalized execution
metrics. Replace or compose it behind `AndroidTaskExecutor` when integrating a
Qualcomm Android QNN/Genie runtime. The Brain remains authoritative for task and
attempt state.

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

The implementation was verified on an API 35 x86_64 emulator against the Python
Brain: registration and live telemetry succeeded, a remote single task returned
an Android result, and three data-parallel shards were executed over the same
persistent stream.

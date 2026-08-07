# DragonNest Hardware Capability Audit

Audit date: 2026-08-05; X Elite evidence updated 2026-08-07 UTC

This audit uses five evidence labels exactly: **verified on physical hardware**,
**verified through Qualcomm AI Hub**, **verified locally without hardware**,
**inferred from official API/header/source inspection**, and **unverified**.
An AI Hub CRD/QRD result is never counted as an APK or laptop-runtime result.

## Audit environment and evidence boundary

The machine running this audit is a CyberPowerPC x64 desktop with an Intel
Core i7-14700KF and Windows 11 build 26200. It is not the Snapdragon X Elite
laptop. No Android device was attached, `adb` was not on `PATH`, and no QAIRT,
QNN, Genie, or AI Hub environment variable was configured in this process.
Consequently that initial desktop session could not create a physical-device
claim. A later session on the actual X Elite produced the sanitized evidence
in `docs/results/xelite_worker_status.md`; the table below incorporates it.

Adjacent `qcom_hackathon` evidence was inspected. It contains a physical S25
QNN transcript, checksummed contexts, AI Hub job records, vector metadata, a
baked-profile release, QAIRT 2.45.41 Android tooling, and current GenieX source.
Proprietary SDK/model files remain outside this repository.

The thin PersonaCare Android client/Agent APK was rebuilt locally after the
immediate simulation-heartbeat and Act 3 protocol changes. Ten Android unit
tests and `assembleDebug` passed. The 19,545,564-byte debug APK has SHA-256
`27b92b7f4189f9ebd1081d37d1a94a635508e84e791488f72f5d6d7b53df34af`.
It intentionally contains no vendor runtime or model, so this is **verified
locally without hardware**, not an NPU execution claim.

## Device audit

| Capability | Snapdragon X Elite laptop | Galaxy S25 Ultra |
|---|---|---|
| Exact identity | Dell Latitude 7455, Snapdragon X Elite `X1E80100`, **verified on physical hardware** | Samsung `SM-S938U1`, Snapdragon 8 Elite (`SM8750` class), **verified on physical hardware** by the saved ADB/QNN transcript |
| OS / ABI | Windows 11 Pro build 26100 / ARM64, **verified on physical hardware** | Android 15 / ARM64, **verified on physical hardware** for OS and inferred from the executed `aarch64-android` runtime for ABI |
| Accelerator | Hexagon NPU visible and real Genie execution reported HTP, **verified on physical hardware** | Hexagon v79 HTP execution is **verified on physical hardware** |
| Available runtime | Qwen3-4B W4A16 bundle plus `genie-t2t-run.exe`, **verified on physical hardware** through DragonNest | `qnn-net-run` build `2.45.41.260507231357` executed physically. GenieX Android 0.3.5 + QAIRT 2.45 bundles are built/compiled but DragonNest JNI execution is **unverified** |
| QAIRT/QNN/Genie/GenieX | Genie/QAIRT `2.48.40.260702`, **verified on physical hardware**; the older installed 2.32 runtime was correctly rejected as incompatible | QNN 2.45.41 physical; AI Hub compiler 2.45.0.260326154327; GenieX 0.3.5 package built. DragonNest's QAIRT 2.48 Genie JNI bundle remains **unverified** |
| Executable formats | Qwen3-4B Genie bundle/context binaries, **verified on physical hardware** | QNN context `.bin` is **verified on physical hardware**; GenieX/QAIRT zip/bundle is **verified through Qualcomm AI Hub**, not through DragonNest APK |
| Stock arbitrary named inputs | No public named-tensor input in Genie/GenieX LLM calls, **inferred from official header/source inspection** | Same GenieX boundary. Direct `qnn-net-run` input lists accept named graph inputs, **verified on physical hardware** for normal model tensors; steering inputs were **verified through AI Hub** only |
| Custom/fork path | Current GenieX QAIRT source exposes `Graph::write(name, data)` and `InputProvider`; a fork is feasible, **inferred from official source inspection** | Same, but rebuilding/package/license practicality on the phone is **unverified** |
| `runtime_vector` | Fixed-shape QNN graph inputs (`steering_vector`, `alpha`) are **verified through Qualcomm AI Hub** on X Elite/8 Elite classes. Runtime-vector injection in the physically verified laptop Genie generation path is **unsupported**, so capability is disabled | Full-model stock GenieX is disabled. Direct-QNN Qwen Part B dynamic input was **verified through Qualcomm AI Hub**, not in the APK or a complete generation loop |
| `baked_profile` | Build/export is feasible by graph rewrite, **verified locally without hardware**; X Elite compiled artifact is **unverified** | Qwen3-0.6B concise layer-7 artifact compiled and bundled, **verified through Qualcomm AI Hub**. Final base/steered physical APK comparison remains **unverified** |
| Fixed split stage | Qwen3-1.7B v73 contexts are downloaded/checksummed and their tensor interfaces are **verified through Qualcomm AI Hub**; physical laptop stage execution is **unverified** | Qwen3-0.6B two-stage execution is **verified on physical hardware**. The exact recovered Qwen3-1.7B prompt+decode+KV contexts are downloaded/checksummed and **verified through Qualcomm AI Hub**; physical execution is **unverified** |
| Install/load | External bundle discovery, tree checksum, manifest validation, installed/cold advertisement, execution, and cleanup are **verified on physical hardware**; persistent warm load remains unsupported | APK asset installer + app-private checksum registry are **verified locally without hardware**. Physical QNN proof used ADB `/data/local/tmp`; DragonNest APK install/load is **unverified** |
| Compatibility sharing | Exact X1E/v73/QAIRT-2.48 bundle compatibility is **verified on this laptop**. Do not share its contexts with SM8750 v79; other X Elite hosts remain **unverified** | Existing artifacts target `sm8750-ac`/v79. Same-family sharing is intended but only SM-S938U1 was physically exercised; wider sharing is **unverified** |
| Licenses/files | Required Genie/HTP 2.48 runtime files and model bundle were present and executed, **verified on physical hardware**; redistribution remains subject to their licenses | Matching arm64 QNN/Genie libraries, v79 skel/stub, model contexts, Android SDK/NDK/JDK are required. QAIRT 2.45 files used physically; redistribution rights remain subject to their licenses |

## Real measurements recovered

### X Elite, physical DragonNest worker

- Direct Qwen3-4B Genie/HTP generation succeeded in 31,019 ms.
- `probe_hardware.py --execute` traversed `HardwareRuntimeAdapter` and succeeded
  in 21,560 ms with runtime `genie`, accelerator `htp`, and a nonempty output.
- Brain -> local gRPC Agent -> Genie -> Brain succeeded through the normal
  scheduler/API path in 13,920 ms.
- Brain -> Agent through the laptop's real LAN interface -> Genie -> Brain
  succeeded in 20,070 ms server execution / 20,249 ms client-observed E2E.
- The worker advertised `qwen3-4b-genie` installed but cold and did not claim
  runtime steering. Full details and hashes are in
  `docs/results/xelite_worker_status.md`.
- A roughly 9 GB available-memory drop was observed during execution and
  recovered afterward. It is not yet a calibrated artifact-memory estimate.

The stage demo does not require a separate desktop Brain. The complete Act 1
to Act 2 topology is now **verified on physical hardware**: PersonaCare on the
S25 executed Request A on `android-mock-v1`; its immediate 64 MB simulated-RAM
heartbeat caused Request B to reject phone placement and route over LAN to the
same-host X Elite Brain/`pc-01` Agent; real Qwen3-4B then executed through
Genie/HTP and PersonaCare displayed `Ran on Snapdragon X Elite Laptop`.

### S25 Ultra, physical direct-QNN execution

Qwen3-0.6B fixed 32-token prompt graph:

| Stage | Context load | NPU execute | Process wall | Deinit |
|---|---:|---:|---:|---:|
| layers 0-13 | 206.7 ms | 37.0 ms | 460.7 ms | 19.5 ms |
| layers 14-27 + head | 226.7 ms | 95.0 ms | 913.6 ms | 13.8 ms |

The `[1,32,1024]` boundary is 128 KiB. Top-1 agreement was 100% against the
AI Hub and local FP32 references. Held-open Part A reported 8.4 MiB PSS and
12.4 MiB RSS while its approximately 718 MiB context was memory-mapped.

Qwen3-1.7B W4A16 four-context claim — **correction, 2026-08-07**:

- prompt partition walls: 0.40, 0.53, 0.51, and 1.15 seconds;
- warm decode partition sum: 1.73-1.77 seconds/token (about 0.56-0.58 tok/s);
- contexts total 1,674,248,192 bytes; largest single context 622,391,296 bytes;
- two runs produced the same four token IDs.

The byte totals are genuine and re-confirmed by the recovered 2026-08-05 AI
Hub artifacts. The prompt-wall timings, decode sum, and determinism statement
above have no corroborating script, log, physical transcript, or AI Hub profile
record and are therefore **unsourced and not physically verified**. Do not use
them as demo evidence. The exact prompt/decode graph interfaces did match across
S25 and X Elite at the AI Hub schema level for all four stages; that does not
prove physical execution, numerical parity, or KV ABI compatibility.

### Qualcomm AI Hub

- Runtime steering and alpha inputs remained live QNN inputs; the full Qwen3
  Part-B output changed across vector/alpha values. **Verified through Qualcomm
  AI Hub**, not through stock Genie.
- DragonNest's small steering/boundary proof passed on Snapdragon 8 Elite QRD
  (v79) and X Elite CRD (v73), all operations on NPU. Job/model IDs are in
  `docs/results/ai_hub_device_lab_proof.json`.

## Exact runtime boundary

```text
Application
  -> Genie/GenieX public LLM API (prompt/token IDs; no named tensor map)
  -> QAIRT LLM input-provider orchestration (stock providers only)
  -> low-level QAIRT Graph::write(name, data)
  -> QNN context graph inputs
  -> HTP
```

Concrete answers:

- Public Genie/GenieX cannot pass arbitrary named tensors.
- A QNN graph can expose fixed-shape steering-vector and scalar-alpha inputs.
- A custom GenieX QAIRT `InputProvider` can write them, but requires a runtime
  rebuild/new API and physical packaging verification.
- Prompt-only, decode-only, or both-token steering depends on which prefill and
  decode graphs contain the Add and receive the provider. It is not selectable
  in stock Genie.
- AI Hub compilation retained the Add/Mul and did not fold the inputs.
- HTP lowering introduced expected low-precision error; behavior-level
  degradation for a quantized full generation loop is still unverified.
- The hackathon-safe path is `baked_profile` first. `runtime_vector` stays off
  for full-model laptop/phone Genie deployments.

## Shortest paths

1. **X Elite QNN stages:** verify the four cached v73 hashes with
   `scripts/artifact_tools/stage_xelite_artifacts.ps1`, then bind and exercise
   the existing QNN runner against the prompt/decode graph names. Keep the
   physically verified Qwen3-4B Genie capability unchanged.
2. **S25 QNN stages:** build/install the thin debug APK, provision the four
   contexts afterward with `scripts/deploy_s25_demo_artifacts.ps1`, package matching
   licensed QAIRT 2.45 arm64 libraries, then complete the native QNN context/KV
   binding and run the acceptance sequence in `docs/QWEN3_1_7B_HANDOFF.md`.

## Current blockers

- The DragonNest Android QAIRT/Genie bridge has not loaded either S25 bundle on
  the physical phone. The recovered static bundle was built against QAIRT 2.45
  / GenieX 0.3.5 while the current JNI staging guide targets QAIRT 2.48; the
  config/API match must be tested, not assumed.
- The new Android QNN JNI boundary and app-private provisioning path are present,
  but context deserialization, named tensor binding, tokenizer integration, and
  stage-local KV updates still require the licensed QAIRT headers/libraries and
  physical S25 validation.
- X Elite QNN contexts were compiled for QAIRT 2.45 while the verified laptop
  Genie environment is QAIRT 2.48. Compatibility must be executed and measured;
  it is not inferred from the matching v73 tensor schema.
- The S25 is not attached now, so APK registration, gRPC execution, thermal
  telemetry, installed/warm artifact state, and baked-profile routing are not
  physical claims.
- Exact tokenizer file fingerprints and full neutral-NLL/refusal/template
  metrics were not present in the recovered summary and remain unverified.

## Status by evidence class

- **Physically verified:** Dell Latitude 7455/X1E80100 identity; Qwen3-4B
  Genie/HTP through `HardwareRuntimeAdapter`, local gRPC, and the real LAN
  interface; worker launcher; SM-S938U1 identity; QNN 2.45.41/v79 execution;
  Qwen3-0.6B split numerics/timings; PersonaCare phone-origin Request A on the
  thin Android mock followed by immediate low-RAM Request B rerouting to real
  X Elite Genie/HTP and returning its result to PersonaCare.
- **AI-Hub verified:** fixed named QNN steering/alpha inputs; small v79->v73
  boundary chain; S25 base/baked compilation; all eight Qwen3-1.7B stage
  contexts and their cross-target prompt/decode tensor interfaces.
- **Source/API verified:** stock Genie/GenieX named-input boundary and custom
  `InputProvider` extension point.
- **Mocked/local only:** Brain transport tests, Android thin APK build/unit
  tests, adapter unit tests, artifact installation/lifecycle tests.
- **Blocked/unverified:** Qwen3-1.7B stage execution on both devices; QAIRT
  2.45-context/2.48-runtime compatibility on X Elite; Android native context/KV
  binding; complete S25-to-laptop generation; physical baked-profile comparison;
  full-model runtime-vector generation.

## Exact next physical commands

On the X Elite laptop:

```powershell
.\scripts\run_xelite_demo.ps1
```

On the workstation with the S25 attached, after staging the matching licensed
SDK/bundle as described in `android-agent/README.md`:

```powershell
$env:DRAGONNEST_QAIRT_SDK_ROOT='C:\path\to\matching\qairt\2.45'
.\android-agent\gradlew.bat -p android-agent :app:assembleDebug
adb install -r android-agent\app\build\outputs\apk\debug\app-debug.apk
.\scripts\deploy_s25_demo_artifacts.ps1 -CacheRoot C:\DragonNestArtifacts
adb shell am start -n com.dragonnest.agent/.AgentSettingsActivity
```

The commands create no repository secrets. Return only secret-free proof JSON
and Android Agent logs/Brain result, not SDK files or credentials.

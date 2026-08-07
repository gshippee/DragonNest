# DragonNest Hardware Capability Audit

Audit date: 2026-08-05; S25 and X Elite evidence updated 2026-08-07 UTC

This audit uses five evidence labels exactly: **verified on physical hardware**,
**verified through Qualcomm AI Hub**, **verified locally without hardware**,
**inferred from official API/header/source inspection**, and **unverified**.
An AI Hub CRD/QRD result is never counted as an APK or laptop-runtime result.

## Audit environment and evidence boundary

The integration machine is a CyberPowerPC x64 desktop with an Intel
Core i7-14700KF and Windows 11 build 26200. It cannot execute Snapdragon
models. Later physical sessions supplied the evidence used here: the X Elite
record in `docs/results/xelite_worker_status.md` and
`docs/results/qwen3_1_7b_xelite_physical_bringup.md`, plus an attached physical
SM-S938U1 recovery/validation session recorded in
`docs/S25_STEERING_RECOVERY.md`. Evidence is attributed to those sessions;
desktop builds and control-plane tests are never relabeled as hardware runs.

Adjacent `qcom_hackathon` evidence was inspected. It contains a physical S25
QNN transcript, checksummed contexts, AI Hub job records, vector metadata, a
baked-profile release, QAIRT 2.45.41 Android tooling, and current GenieX source.
Proprietary SDK/model files remain outside this repository.

PersonaCare has two intentional build classes. The thin build contains the
mock runtime for portable control-plane testing. The explicit S25 hardware
build packages the licensed GenieX 0.3.5 / QAIRT 2.45 runtime closure, disables
`android-mock-v1`, and keeps model bytes externally provisioned. A successful
desktop APK build is still only **verified locally without hardware**; the
separate on-device load/probe evidence is called out below.

## Device audit

| Capability | Snapdragon X Elite laptop | Galaxy S25 Ultra |
|---|---|---|
| Exact identity | Dell Latitude 7455, Snapdragon X Elite `X1E80100`, **verified on physical hardware** | Samsung `SM-S938U1`, Snapdragon 8 Elite (`SM8750` class), **verified on physical hardware** by the saved ADB/QNN transcript |
| OS / ABI | Windows 11 Pro build 26100 / ARM64, **verified on physical hardware** | Android 15 / ARM64, **verified on physical hardware** for OS and inferred from the executed `aarch64-android` runtime for ABI |
| Accelerator | Hexagon NPU visible and real Genie execution reported HTP, **verified on physical hardware** | Hexagon v79 HTP execution is **verified on physical hardware** |
| Available runtime | Qwen3-4B W4A16 bundle plus `genie-t2t-run.exe`, **verified on physical hardware** through DragonNest. The standalone Qwen3-1.7B QAIRT 2.45 QNN harness is also physically verified | QNN 2.45.41 and GenieX Android 0.3.5 / QAIRT 2.45 / HTP v79 are **verified on physical hardware** |
| QAIRT/QNN/Genie/GenieX | Genie/QAIRT `2.48.40.260702` for 4B and QAIRT 2.45 for the standalone four-stage QNN harness, **verified on physical hardware** | QNN 2.45.41 and GenieX 0.3.5 / QAIRT 2.45, **verified on physical hardware** |
| Executable formats | Qwen3-4B Genie bundle and all four Qwen3-1.7B v73 QNN contexts, **verified on physical hardware** in their respective harnesses | QNN context `.bin` and prompt+decode GenieX bundles, **verified on physical hardware** |
| Stock arbitrary named inputs | No public named-tensor input in Genie/GenieX LLM calls, **inferred from official header/source inspection** | Same GenieX boundary. Direct `qnn-net-run` input lists accept named graph inputs, **verified on physical hardware** for normal model tensors; steering inputs were **verified through AI Hub** only |
| Custom/fork path | Current GenieX QAIRT source exposes `Graph::write(name, data)` and `InputProvider`; a fork is feasible, **inferred from official source inspection** | Same, but rebuilding/package/license practicality on the phone is **unverified** |
| `runtime_vector` | Fixed-shape QNN graph inputs (`steering_vector`, `alpha`) are **verified through Qualcomm AI Hub** on X Elite/8 Elite classes. Runtime-vector injection in the physically verified laptop Genie generation path is **unsupported**, so capability is disabled | Full-model stock GenieX is disabled. Direct-QNN Qwen Part B dynamic input was **verified through Qualcomm AI Hub**, not in the APK or a complete generation loop |
| `baked_profile` | Build/export is feasible by graph rewrite, **verified locally without hardware**; X Elite compiled artifact is **unverified** | Base, Concise layer-7 alpha -4, and Detailed layer-7 alpha +4 prompt+decode bundles executed through GenieX/HTP in the preserved reference app, **verified on physical hardware** |
| Fixed split stage | Qwen3-1.7B v73 S0-S3 prefill plus eight decode steps with stage-local KV and coherent text are **verified on physical hardware** in the reproducible standalone harness; normal Agent/provider 4+0 acceptance remains pending | Qwen3-0.6B two-stage execution is **verified on physical hardware**. Qwen3-1.7B v79 contexts are AI-Hub/interface verified but not physically executed |
| Install/load | The 4B DragonNest path is physically verified. The new 1.7B production provider is desktop-tested and explicitly gated, but awaits Agent-path physical acceptance | `com.dragonnest.agent` physically loaded/probed and advertised the exact Detailed bundle through GenieX/HTP. Integrated end-to-end text was not completed before the phone left; Base/Concise remain to be provisioned from the repaired cache and accepted |
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

- The explicit DragonNest S25 hardware build loaded the real GenieX 0.3.5 /
  QAIRT 2.45 runtime and passed the Detailed artifact execution-ready probe on
  SM-S938U1 HTP. The session ended before a normal Brain task returned text.
  Base and Concise must be reprovisioned from the repaired dual-phase cache and
  all three profiles must complete the integrated Brain -> Agent acceptance.
- The new Android QNN JNI boundary and app-private provisioning path are present,
  but context deserialization, named tensor binding, tokenizer integration, and
  stage-local KV updates still require the licensed QAIRT headers/libraries and
  physical S25 validation.
- The X Elite QAIRT 2.45 contexts and runtime algorithm are physically proven
  by `scripts/xelite_bringup/`, including correct delta-KV append/window
  semantics. The newly promoted production Agent provider still requires the
  normal 4+0 Brain-path acceptance; it is not physically verified by the
  standalone result.
- The S25 is no longer attached. Current desktop test/build results cannot add
  new phone execution evidence.
- Exact tokenizer file fingerprints and full neutral-NLL/refusal/template
  metrics were not present in the recovered summary and remain unverified.

## Status by evidence class

- **Physically verified:** Dell Latitude 7455/X1E80100 identity; Qwen3-4B
  Genie/HTP through `HardwareRuntimeAdapter`, local gRPC, and the real LAN
  interface; worker launcher; standalone X Elite 1.7B S0-S3 prefill/eight-step
  decode on HTP; SM-S938U1 identity; QNN 2.45.41/v79 execution; Qwen3-0.6B
  Base/Concise/Detailed GenieX/HTP reference execution and behavior comparison;
  DragonNest Detailed load/probe/advertisement; PersonaCare phone-origin Request A on the
  thin Android mock followed by immediate low-RAM Request B rerouting to real
  X Elite Genie/HTP and returning its result to PersonaCare.
- **AI-Hub verified:** fixed named QNN steering/alpha inputs; small v79->v73
  boundary chain; S25 base/baked compilation; all eight Qwen3-1.7B stage
  contexts and their cross-target prompt/decode tensor interfaces.
- **Source/API verified:** stock Genie/GenieX named-input boundary and custom
  `InputProvider` extension point.
- **Mocked/local only:** Brain transport tests, Android builds/unit tests,
  production-provider unit tests, artifact installation/lifecycle tests, and
  simulated 4+0/3+1/2+2 routing.
- **Blocked/unverified:** normal Agent-path X Elite 1.7B 4+0; S25 1.7B direct
  QNN; real 3+1 and 2+2; integrated `com.dragonnest.agent` Base/Concise/Detailed
  text responses; full-model runtime-vector generation.

## Exact next physical commands

On the X Elite laptop for the physically verified Act 1/2 topology:

```powershell
.\scripts\run_xelite_demo.ps1
```

For the production-provider 1.7B 4+0 acceptance, use the exact staged command
in `docs/QWEN3_1_7B_HANDOFF.md`. It keeps QAIRT 2.45 QNN separate from the
verified 4B QAIRT 2.48 Genie capability and requires an explicit enable flag.

On the workstation with the S25 attached, after staging the matching licensed
SDK/bundle as described in `android-agent/README.md`:

```powershell
$env:JAVA_HOME='C:\path\to\jdk-17'
$env:ANDROID_HOME='C:\path\to\android-sdk'
.\android-agent\gradlew.bat -p android-agent :app:assembleDebug -PincludeS25GenieXRuntime=true --no-daemon
adb install -r android-agent\app\build\outputs\apk\debug\app-debug.apk
.\scripts\deploy_s25_local_artifacts.ps1 -CacheRoot C:\DragonNestArtifacts\qwen3-0.6b\s25 -AdbPath "$env:ANDROID_HOME\platform-tools\adb.exe"
adb shell am start -n com.dragonnest.agent/.AgentSettingsActivity
```

The commands create no repository secrets. Return only secret-free proof JSON
and Android Agent logs/Brain result, not SDK files or credentials.

# GenieX Runtime Activation Steering (SteerLab)

Runtime-adjustable activation steering for Qwen on Snapdragon HTP through a
forked GenieX: one compiled bundle, `alpha` and `steering_vector` change
between requests as ordinary named graph inputs — no recompile, no relink, no
reload.

## Forked runtime (branch `steering/aux-inputs`)

- https://github.com/shubhamx64/GenieX — public C ABI (`geniex_NamedTensor`,
  aux fields on generate/forward-logits, `geniex_llm_get_aux_stats`), QAIRT
  plugin mapping, python bindings. Base: `qualcomm/geniex@3de683b`.
- https://github.com/shubhamx64/geniex-qairt-plugin — the generic aux
  named-tensor mechanism (`GenerationConfig::aux_inputs`,
  `AuxTensorInputProvider`, `aux_inputs.json` sidecar, zero-fill-at-load,
  per-phase diagnostics) + 17 new unit tests + Linux-host build fixes +
  `aux_steering_probe` example. Base: `qualcomm/geniex-qairt-plugin@abf961f`.

## In this repo

| Path | Content |
|---|---|
| `docs/geniex_runtime_steering_plan.md` | implementation plan (pre-coding audit) |
| `docs/geniex_runtime_steering_implementation.md` | files changed, ABI, compiled-model contract, upstreaming strategy |
| `docs/geniex_runtime_steering_demo.md` | demo script + setup |
| `docs/geniex_runtime_steering_final_report.md` | final report: what was proven, evidence classes |
| `geniex-steering/proofs/` | machine-readable proofs (host suite, S25 mechanism proof, S25 Qwen semantic proof), APK screenshots, `steerlab-app-debug.apk` |
| `geniex-steering/apk/` | SteerLab app sources (`com.dragonnest.geniexsteeringlab`, diagnostics-first UI, ±10 alpha slider) |
| `geniex-steering/device/` | bundle/build tooling: tiny steering test-context builder, runtime-input Qwen bake + ai-hub-models patch, JNI/APK build scripts, C-ABI probe |

Model bundles are not committed (700 MB); they are reproducible from the
recorded AI Hub jobs (part-2 link `jgznmqz6g`, sibling part-1 link
`jpeyz0605` — same 2026-08-07 04:48:53-56 batch) via
`geniex-steering/device/make_qwen_steered_bundle/`.

A restorable local copy (with full provenance, checksums, and the
reconstructed `genie_config.json`/`aux_inputs.json`/tokenizer files) lives
outside this repo at
`C:\DragonNestArtifacts\geniex-steering\qwen3-0.6b-runtime-steerable\` —
treat that as the canonical bundle, not the SteerLab app's private storage,
which gets wiped by any `adb uninstall` (e.g. a forced reinstall after a
debug-signing-key mismatch).

## Rebuilding the APK

The Java/Kotlin layer builds with a stock Android SDK (no NDK needed) as long
as two non-source trees are staged into `apk/app/src/main/` first — both can
be lifted straight out of `proofs/steerlab-app-debug.apk`:

- `jniLibs/arm64-v8a/` — the 12 prebuilt `.so` files (forked GenieX stack +
  QNN runtime + the JNI shim). Not committed; ~106 MB.
- `assets/steering_vector_layer7_unit.bin` — the layer-7 verbosity vector,
  unit L2 norm, 1024 float32. This one IS committed (`git add -f`, since the
  root `.gitignore` blanket-ignores `*.bin`).

```
unzip -j proofs/steerlab-app-debug.apk 'lib/arm64-v8a/*' -d apk/app/src/main/jniLibs/arm64-v8a/
gradle :app:assembleDebug
```

**If `assets/` is missing the app still builds and runs, but silently does not
steer** — `MainActivity` logs `ERROR loading vector asset`, sends only `alpha`
(1 aux tensor instead of 2, 1 prefill write instead of 2), and `alpha` with no
vector is a mathematical no-op. Check the diagnostics panel for
`PREFILL graphs got 2 aux writes` before trusting a steering result.

Rebuilt APKs are signed with the building machine's debug key. Installing over
a copy signed by a different machine requires `adb uninstall`, **which wipes
the app-private model bundle** — make sure the external bundle cache is
populated first (see the note above).

Proven on a physical Galaxy S25 Ultra (SM8750, HTP v79): fp16-exact mechanism
numerics on a test graph, and direction-specific verbose/concise steering of
real Qwen3-0.6B W4A16 at ~124 tok/s with one context load. The known-good
baked-profile demo and PersonaCare were not modified.

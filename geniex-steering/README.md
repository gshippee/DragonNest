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

Proven on a physical Galaxy S25 Ultra (SM8750, HTP v79): fp16-exact mechanism
numerics on a test graph, and direction-specific verbose/concise steering of
real Qwen3-0.6B W4A16 at ~124 tok/s with one context load. The known-good
baked-profile demo and PersonaCare were not modified.

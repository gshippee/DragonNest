# Galaxy S25 Ultra runtime activation-steering physical acceptance

Date: 2026-08-08. Branch: `codex/qwen17-variable-split`.

Physical acceptance for `Compute = Local` with runtime activation steering on
the real Galaxy S25 Ultra (SM-S938U1, SM8750, HTP v79), through the normal
DragonNest Brain gRPC path — not a GenieX bypass.

## Result

One prompt, three profiles, one loaded context:

| Profile | Artifact | Runtime | Steering | Output |
|---|---|---|---|---|
| Balanced | `qwen3-0.6b-s25-base` | `genie` (stock GenieX 0.3.5) | none | normal multi-sentence explanation |
| Concise | `qwen3-0.6b-s25-runtime-steerable` | `genie_aux` (forked) | `runtime_vector`, α −8, layer 7 | "The sky is blue because it is the sky." (11 tokens) |
| Detailed | `qwen3-0.6b-s25-runtime-steerable` | `genie_aux` (forked) | `runtime_vector`, α +8, layer 7 | full Rayleigh-scattering explanation (96 tokens, hit cap) |

Prompt: `Explain why the sky is blue.` Deterministic (top-k 1, seed 42); the
run below reproduced an earlier run token-for-token.

Accepted task ids: `task-d5848141` (Balanced), `task-7fa7243d` (Concise),
`task-b0cf7525` (Detailed).

## Device-reported aux evidence

Logged by `GenieXSteeringRuntimeBridge` from `geniex_llm_get_aux_stats`:

```
task-7fa7243d  alpha=-8.0  prefill_aux_writes=2  decode_aux_writes=20   generated_tokens=11  tps=120.772  context_loads=1  ms=118
task-b0cf7525  alpha=+8.0  prefill_aux_writes=2  decode_aux_writes=192  generated_tokens=96  tps=117.573  context_loads=1  ms=836
```

- Both prompt/prefill **and** decode graphs received the aux tensors on every
  steered request (`prefill_aux_writes` 2, `decode_aux_writes` = 2 × decode
  steps), which is the designed binding.
- `context_loads: 1` across both requests: the alpha changed between them with
  **no context reload**.
- Balanced produced **no** `DragonNestGenieXAux` line at all — it never touched
  the forked runtime.

## Acceptance conditions

| Condition | Result |
|---|---|
| real physical S25, normal Brain path | met |
| correct artifact (`qwen3-0.6b-w4a16-sm8750-runtime-vector-l7-qairt245`) | met |
| `steering.enabled = true`, `mode = runtime_vector` | met |
| `vector_id = concise-vs-verbose-layer-7`, `target_layer = 7` | met |
| expected alpha (−8 / +8) | met |
| GenieX fork, HTP | met |
| nonempty coherent output | met |
| directionally appropriate effect | met (11 vs 96 tokens on the same prompt) |
| prefill aux writes > 0 | met (2) |
| decode aux writes > 0 | met (20 / 192) |
| no context reload between sequential requests | met (`context_loads: 1`) |
| no mock | met (not compiled into the hardware build) |
| no prompt-conditioning fallback | met |
| no baked artifact used during the runtime-vector proof | met |

## Balanced regression gate

Re-run after the integration, unchanged:

```
task-17ec1ee8  Compute=Local  Profile=Balanced  "What is the capital of Japan?"
  -> android-13cda486-.../qwen3-0.6b-s25-base   "Japan's capital is Tokyo."
```

## Runtime isolation

The stock GenieX AAR and the forked closure ship **side by side** in one APK.
Five forked libraries collide by soname with
`com.qualcomm.qti:geniex-android:0.3.5`, so
`scripts/artifact_tools/stage_steering_native_closure.py` renames them in place
to equal-length private names (`libgeniex.so` → `libgnxfrk.so`, etc.). Equal
length keeps every `.dynstr` offset valid, so no ELF rewriting or `patchelf` is
needed and the physically proven binaries are reused byte-for-byte apart from
those single soname strings. Each colliding name occurs exactly once per file,
asserted before any byte is written.

Verified on-device, both closures extracted and resident:

```
libgeniex.so  libgeniex_core.so  libgeniex-proc.so  ...   <- stock, serves Base
libgnxfrk.so  libgnxfrk_core.so  libgnxfrk-proc.so  ...   <- fork, serves genie_aux
libgeniex_plugin.so  libsteeringlab_jni.so                <- fork-only names
```

QNN libraries were deliberately not staged: four of the fork's five are
byte-identical to the AAR's, and `libQnnHtpV79Skel.so` — whose filename the DSP
resolves and so cannot be renamed — is left to the stock AAR, whose
`libQnnHtp.so` and `libQnnHtpV79Stub.so` are byte-identical to the fork's.

`libsteeringlab_jni.so` is the exact binary from
`geniex-steering/proofs/steerlab-app-debug.apk`, reused rather than rebuilt. It
resolves native methods by symbol name, which is why
`com.dragonnest.geniexsteeringlab.GenieXBridge` keeps that package.

## Provenance

Bundle (`C:\DragonNestArtifacts\qwen3-0.6b\s25\runtime-steerable`, reached by a
junction from the SteerLab canonical copy):

- `sha256_tree e7ae40e91db941a7d3c68db01abf35c6a1a91248f0d32eb9481974d6aba9afae`
- `part1_of_2.bin 87ee07c2f24f10a75662b579061c29d6725d9a84edaf654c723429e473b1c978` (AI Hub link `jpeyz0605`)
- `part2_of_2.bin c189fe74ae05b44695534191fbdeaed9f23f577d9e9d1695882fec206b6c863e` (AI Hub link `jgznmqz6g`)
- dual-phase: `prompt_ar128_cl512_{1,2}_of_2` + `token_ar1_cl512_{1,2}_of_2`
- `aux_inputs.json` declares `alpha` and `steering_vector`

Vector: `steering_vector_layer7_unit.bin`,
`sha256 800f3248289f97164d0936a567a16b93e16ecbfabbb5d0ec08a811ca0cb76305` —
1024 float32 little-endian, L2 norm 1.0 (measured 0.99999997). This is the
SteerLab APK asset, the L2-normalised form of the recorded research vector
`7d69ff39…` (which is the unnormalised `.pt` source).

## Alpha selection

±8 was adopted directly rather than swept. Fable's CLI evidence used ±8 and the
APK used ±10; on this bundle ±8 produces an unmistakable separation on the same
prompt (11 vs 96 tokens), so a −4/0/+4 escalation sweep was not needed. The
validated alpha range in `configs/steering-vectors.yaml` was widened from ±4 to
±10 for the runtime realization; ±4 remains the baked-profile bake value.

**Not established:** a systematic multi-prompt sweep, and a norm-matched random
vector control on *this* integration (Fable established direction-specificity
on the same bundle through the SteerLab APK). Treat ±8 as a working demo value,
not a tuned optimum.

## Profile UI

`configs/behavior-profiles.yaml` now resolves Concise/Detailed as
`runtime_vector` first, `baked_profile` second, under
`allow_baked_equivalent`. Balanced remains `none` on the stock Base artifact.
Selecting a profile in PersonaCare therefore drives runtime steering with no
UI change, and the response metadata truthfully reports
`persona realized: concise (mode=runtime_vector)`.

A **Steering strength** slider was added to the profile screen at product
request (−10…+10, centre = "Profile default"). Off-centre it sends an explicit
`SteeringSpec` with the request; Brain still validates vector/layer/alpha and
fails closed. This deviates from the original brief's "expose semantic labels
only" rule, on an explicit product decision.

**Not established:** the slider is built, installed, and unit-tested, but no
request has been driven through it from the UI on hardware. The α ±8 evidence
above came from the profile ladder, not the slider.

## Fail-closed behaviour

- Unknown vector id, unsupported layer, non-`runtime_vector` mode, non-finite
  alpha, wrong-width or non-unit-norm vector, or a bundle without
  `aux_inputs.json` are all rejected rather than executed unsteered.
- A steered request whose device reports zero prefill or decode aux writes
  raises rather than returning text that merely looks successful.
- An artifact whose only realization is `runtime_vector` is excluded from
  unsteered/prompt-conditioned requests, so Balanced can never be served from
  the steering bundle.
- Realization `compatible_model_families` / `compatible_runtimes` /
  `compatible_quantizations` are now enforced; they were previously parsed and
  ignored.

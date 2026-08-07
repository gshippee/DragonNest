# GenieX Runtime Auxiliary-Input Implementation

Feature description (upstream-facing): **runtime binding of auxiliary named
graph inputs for GenieX LLM execution**. Activation steering (`alpha`,
`steering_vector`) is the demonstration application, not the mechanism.

## Upstream repositories and revisions

| Repo | Base revision | License | Fork branch |
|---|---|---|---|
| `github.com/qualcomm/geniex` | `3de683b86112bfe3028adbc2fd2f615d5af937b5` (2026-08-03) | BSD-3-Clause | `steering/aux-inputs` |
| `github.com/qualcomm/geniex-qairt-plugin` (submodule `third-party/geniex-qairt`) | `abf961f047436b4593bb1cbb21bda13d9443ef3b` (2026-07-30) | BSD-3-Clause | `steering/aux-inputs` |
| `github.com/qualcomm/geniex-proc` (nested submodule) | `a4558f0014edfb42899358984caa548e108d1987` | unmodified | — |

Both licenses permit modification and redistribution. Fork worktrees live at
`qcom_hackathon2\geniex-steering\geniex` (+ nested geniex-qairt); the baseline
checkouts under `qcom_hackathon\artifacts\deps` are untouched.

## Architecture

```
Demo app (SteerLab APK / aux_steering_probe CLI)
    │  prompt, alpha f32[1], steering_vector f32[1,1,H]
    ▼
geniex.h C ABI                geniex_LlmGenerateInput.aux_inputs (geniex_NamedTensor[], tail fields)
    ▼                         geniex_LlmForwardLogitsInput.aux_inputs / geniex_llm_get_aux_stats
sdk/src/llm.cpp bridge        (no change needed — struct passes through wholesale)
    ▼
QAIRT plugin (QairtLlm)       mapAuxInputs(): validate name/dtype/size/dupes vs loaded
    ▼                         graphs, copy into GenerationConfig::aux_inputs
LLMPipeline                   aux carried inside GenerationConfig (generate) or
    ▼                         explicit param (forwardLogits)
LLMModel::generate            validateAuxInputs() + AuxBindingScope (set/clear,
    ▼                         exception-safe, per request)
LLMModel::runShard            provider loop — SINGLE choke point for prefill,
    ▼                         decode, forwardLogits and sliding-window re-prefill
AuxTensorInputProvider        hasInput() guard per shard, exact-size full
    ▼                         overwrite, per-phase write counters
Graph::write(name, f32*, n)   float write with on-write quantization to the
    ▼                         tensor's own dtype/scale/offset
QNN → HTP
```

## Exact files changed

### geniex-qairt (core) — branch `steering/aux-inputs`

| File | Change |
|---|---|
| `core/include/types.h` | `struct NamedTensor {name, shape, data}` (owned copies); `GenerationConfig::aux_inputs`; `ModelConfig::aux_input_names` |
| `core/include/llm/input_provider.h` + `core/src/llm/input_provider.cpp` | `AuxTensorInputProvider` (set/clear, hasInput-guarded exact-size writes, per-phase counters, debug logging of name/phase/graph/first-element) |
| `core/include/llm/llm_model.h` + `core/src/llm/llm_model.cpp` | unconditional provider registration in `onInitialized()` AFTER `createInputProviders()` (subclass-proof); `all_graph_input_names_` set; `validateAuxInputs` (unknown/duplicate/empty → `std::invalid_argument` naming the tensor and the declared list); `AuxBindingScope` RAII set/clear in `generate` and `forwardLogits`; `forwardLogits` gains defaulted aux param; `findGraphInputSpec`; `AuxStats` (prefill/decode writes, aux/total requests); **spec-inference guard**: `isSpecialOrAux` consults `ModelConfig::aux_input_names` at the 5 heuristic call sites in `inferSpecFromGraphs` |
| `core/include/pipeline/llm_pipeline.h` + `core/src/pipeline/llm_pipeline.cpp` | `graphInputSpec()`, `auxStats()`, `forwardLogits(..., aux)` pass-through |
| `core/src/llm/llm_spec_loader.cpp` + `.h` | `readAuxInputNames()` parsing the optional `aux_inputs.json` bundle sidecar; wired into `modelConfigFromDirectory` |
| `tests/testing/llm_fixture.hpp` | `AuxLLMFixture` — fixture graphs exposing `alpha`/`steering_vector`, deliberately listed FIRST (adversarial for spec inference) |
| `tests/core/llm/input_provider_test.cpp` | 5 provider tests incl. quantized-target write |
| `tests/core/llm/llm_model_test.cpp` | 8 end-to-end + 3 sidecar tests |
| `tests/CMakeLists.txt`, `tests/core/{llm,vlm}/*_test.cpp` | Linux-host build fixes (DmaBufAllocator link, portable setenv) — required for any non-Windows build, unrelated to steering |
| `examples/aux_steering_probe/` | device probe CLI (JSON-line output for host-side assertions) |

### geniex (SDK) — branch `steering/aux-inputs`

| File | Change |
|---|---|
| `sdk/include/geniex.h` | `geniex_TensorDataType`, `geniex_NamedTensor`; tail fields `aux_inputs`/`aux_input_count` on `geniex_LlmGenerateInput` AND `geniex_LlmForwardLogitsInput`; `geniex_LlmAuxStats` + `geniex_llm_get_aux_stats`; `<stddef.h>` include |
| `sdk/include/plugin/ILlm.h` | defaulted virtual `get_aux_stats` (PARAM_NOT_SUPPORTED) |
| `sdk/src/llm.cpp` | `geniex_llm_get_aux_stats` bridge only — generate/forward-logits pass structs through unchanged |
| `sdk/plugins/qairt/{include/llm.h, src/llm.cpp}` | `mapAuxInputs` validation/conversion; wiring into `generate` + `forward_logits`; `get_aux_stats` |
| `sdk/plugins/llama_cpp/src/llm.cpp` | rejects aux inputs with `PARAM_NOT_SUPPORTED` (no silent discard). Compile-verified only on x64 config parsing level — the llama.cpp submodule is not checked out in this fork |

## Generic vs steering-specific

The mechanism is **generic named-tensor support**: nothing in geniex core or
the SDK knows the strings "alpha" or "steering_vector" except tests, examples
and docs. Model bundles self-declare their auxiliary inputs via
`aux_inputs.json`; the runtime writes whatever validated tensors the caller
supplies to whatever graphs expose them.

## Ownership / lifetime / thread-safety

- The C ABI copies caller buffers during the call (`mapAuxInputs` →
  `std::vector<float>` owned copies inside `GenerationConfig`); callers may
  free immediately after `geniex_llm_generate` returns.
- Provider state is per-request: bound after sampler prep, cleared by RAII on
  every exit path including exceptions. A stock request after an aux request
  performs zero aux writes (unit-tested).
- No process-global state. GenieX handles keep their existing "not
  thread-safe per handle" contract; SteerLab serializes requests with a mutex.
- Per-token (mid-generation) updates are not implemented but not precluded:
  the provider is consulted before every decode step, so a future setter hook
  could swap values between steps.

## Prefill/decode propagation

`LLMModel::runShard` (`core/src/llm/llm_model.cpp`) is the single point where
every graph execution binds inputs; the registered provider list is consulted
for prefill chunks, every decode step, `forwardLogits`, and sliding-window
re-prefill. Verified: unit tests assert identical alpha bytes in the prefill
and decode graphs' input buffers; on-device the per-phase counters report
non-zero writes for both phases, surfaced in the APK diagnostics panel via
`geniex_llm_get_aux_stats`.

## Required compiled-model input contract

The runtime change alone is insufficient: the compiled QNN context must expose
the tensors. For a steering-capable Qwen bundle:

1. Every shard graph to be steered — BOTH the prompt graph(s)
   `prompt_ar<N>_cl<M>_<s>_of_<t>` and the decode graph(s)
   `token_ar1_cl<M>_<s>_of_<t>` — must expose additional APP_WRITE inputs:
   - `alpha`: float32 `[1]`
   - `steering_vector`: float32 `[1, 1, hidden_size]`
   (For Qwen3-0.6B split exports these land on part 2 only — the split derives
   subgraph inputs from dataflow, and the injection point, layer 7's residual,
   lives in part 2.)
2. The bundle directory must carry `aux_inputs.json`:
   `{"aux_inputs": ["alpha", "steering_vector"]}` — this both enables
   validation and protects spec inference from mistaking the float `[1,1,H]`
   vector input for the inter-shard hidden state (a real failure mode,
   unit-tested both ways).
3. Quantized (W4A16 + `--quantize_io`) exports need activation encodings for
   the new inputs/intermediates. The prototype uses symmetric INT16:
   alpha range ±12 (slider is ±10), vector range ±0.5 (unit vectors' max
   |component| ≈ 0.18), and widens the steered residual's cloned encoding by
   the max steering delta so extreme alphas don't clip.
4. The injected ops are `Reshape(alpha,[1,1,1]) → Mul → Add` after the target
   residual — the pattern already proven not to constant-fold on HTP.

Production pipeline: `device/make_qwen_steered_bundle/bake_runtime_steering.py`
(ONNX surgery on the published W4A16 AIMET checkpoint) + a 15-line patch to
ai-hub-models' `LLMPartBase._extra_graph_inputs` / `get_graph_sample_inputs`
(`device/make_qwen_steered_bundle/qai_hub_models_steering.patch`) so the
AI Hub compile gives the new inputs their fixed shapes, then the stock
`qai_hub_models.models.qwen3_0_6b.export --runtime geniex_qairt
--sequence-lengths 128,1` flow. One-time recompile; value changes at runtime
never recompile.

Minimal proof context: `device/make_test_context/` builds a tiny LLM-shaped
graph pair (fixture tensor set + aux inputs, H=8, V=16) with the QAIRT
2.45.41 x86_64-linux converter + offline HTP prepare (v79) — compiled once in
WSL, exercised on the S25 across alpha/vector sweeps.

## Compatibility with ordinary GenieX models

Preserved exactly:
- Zero aux inputs ⇒ no behavioral change (162/162 unit tests green, incl. the
  pre-existing 147; stock device generation runs unchanged).
- Bundles without `aux_inputs.json` load and behave stock.
- The C structs grow only at the tail, matching the project's own extension
  convention (e.g. the sliding-window fields); zero-initialized structs from
  old code preserve stock behavior. Old compiled binaries passing a smaller
  struct by pointer would read garbage tail fields — same caveat as every
  prior tail extension in this ABI; recompile against the new header.
- Pre-existing upstream issue (NOT introduced here, reproduced with unpatched
  baseline): current geniex-qairt cannot load prompt-only bundles (the
  2026-08-04 era S25 releases lack `token_ar1` graphs and loading throws an
  out-of-range during init). Fresh exports with `--sequence-lengths 128,1`
  include both phases and avoid it.

## Android build

Cross-compiled with NDK r27c from WSL Ubuntu (gcc host tools, rustup with
`aarch64-linux-android` target for tokenizers-cpp/model-manager):
- geniex-qairt core + probe: `cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DGENIEX_BUILD_EXAMPLES=ON`
- SDK (`libgeniex.so` + qairt `libgeniex_plugin.so`): same toolchain,
  `-DGENIEX_PLUGIN_LLAMA_CPP=OFF`, `ANDROID_NDK_ROOT` exported for the Rust
  model-manager cross-build.
- APK packaging: all .so files flat in `jniLibs/arm64-v8a`; JNI sets
  `GENIEX_PLUGIN_PATH` and `ADSP_LIBRARY_PATH` to `nativeLibraryDir`, which
  drives both plugin discovery and QNN backend/skel resolution on Android.
- Host-side unit tests are Linux/Windows; two upstream VLM teardown tests are
  flaky on Linux hosts (pre-existing, unrelated).

## Smallest DragonNest integration surface (future work — DragonNest untouched)

1. Ship the forked `libgeniex*`/plugin libraries + a steering-capable bundle +
   `aux_inputs.json` as a new artifact (`steering_mode: runtime_vector`,
   new compatibility class).
2. `genie_runner.py` gains an in-process path (ctypes on the fork's ABI) or the
   probe CLI as subprocess with `--alpha/--vector` args; `SteeringSpec` already
   carries `vector_id` + `alpha` — the adapter resolves the vector file and
   passes bytes.
3. `hardware_adapter.execute_runtime_steered()` stops raising and forwards to
   that path when the artifact advertises `runtime_vector`.
4. `configs/steering-vectors.yaml` `validated_runtimes` gains the new runtime
   token only after on-target numerical validation (per HARDWARE_CONTRACT §3);
   `tests/test_steering.py` boundary cases updated accordingly.

## Upstreaming strategy

Three separable series: (1) Linux-host test build fixes (independently
useful); (2) generic aux named-tensor mechanism (core + ABI + tests + docs);
(3) steering demo assets (probe example, bundle tooling) — optional. The
`aux_inputs.json` sidecar and `_extra_graph_inputs` recognition would land in
ai-hub-models as a companion PR.

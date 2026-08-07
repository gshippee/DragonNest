# GenieX Runtime Auxiliary-Input (Activation Steering) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend a fork of Qualcomm GenieX so an application can supply arbitrary named auxiliary tensors (`alpha` float32 `[1]`, `steering_vector` float32 `[1,1,H]`) on every prompt/decode graph invocation, proving runtime-variable Qwen activation steering on HTP without recompilation between value changes.

**Architecture:** Generic named-tensor list appended (tail, backward-compatible style already used upstream) to `geniex_LlmGenerateInput` / `geniex_LlmForwardLogitsInput` → carried in geniex-qairt `GenerationConfig::aux_inputs` (owned copies) → new `AuxTensorInputProvider` registered on `LLMModel`, whose single `runShard` choke point already services prefill, decode, `forwardLogits`, and sliding-window re-prefill → `Graph::write(name, float*, n)`. Bundles self-declare aux input names via an `aux_inputs.json` sidecar so load-time spec inference (`isSpecialTensor` heuristics) never mistakes a steering tensor for the hidden-state input.

**Tech Stack:** C++17 (geniex-qairt core + GenieX SDK), gtest with link-time QNN stub (host proof, WSL Ubuntu x64), Android NDK arm64 build (device proof on Galaxy S25 Ultra SM8750/HTP v79), AI Hub for the steered Qwen context compile, ctypes Python bindings.

## Global Constraints

- Baseline revisions (verbatim): `qualcomm/geniex` @ `3de683b86112bfe3028adbc2fd2f615d5af937b5`; `qualcomm/geniex-qairt-plugin` @ `abf961f047436b4593bb1cbb21bda13d9443ef3b`; `qualcomm/geniex-proc` @ `a4558f0014edfb42899358984caa548e108d1987`. Licenses: BSD-3-Clause (modification + redistribution allowed).
- Work ONLY in the fork worktrees under `C:\Users\shubh\Downloads\qcom_hackathon2\geniex-steering\` (branch `steering/aux-inputs`). Do NOT modify `C:\Users\shubh\Downloads\qcom_hackathon2\DragonNest`, the baseline checkout at `qcom_hackathon\artifacts\deps\geniex` (other than git worktree metadata), or any artifact under `qcom_hackathon\artifacts\` (read-only inputs).
- Zero aux inputs ⇒ byte-identical stock behavior. All 111 existing geniex-qairt unit-test cases must stay green.
- No process-global steering state; per-request owned copies; provider state set/cleared inside the generate call. GenieX handles are already documented "not thread-safe" per handle — keep that contract, document it.
- Errors must be actionable: unknown name, dtype mismatch, element-count mismatch, null data, duplicate name each produce a distinct logged error and `GENIEX_ERROR_COMMON_INVALID_INPUT` / `PARAM_NOT_SUPPORTED` (never silently dropped).
- Steering-vector assets are read-only: `qcom_hackathon\artifacts\vector_layer_7.pt` (layer 7, used by baked demo) and `vector_layer_21.pt` (layer 21, runtime-proof layer), both float32 `(1024,)` for Qwen3-0.6B (H=1024). Do NOT regenerate vectors.
- Recompiling once to add graph inputs is acceptable; recompiling when alpha/vector values change is not.
- Never log full prompts or whole steering vectors — names, shapes, dtypes, checksums, summary stats only. No credentials/proprietary files in commits.

## File Structure (fork)

```
geniex-steering/
  geniex/                                  # worktree of qualcomm/geniex, branch steering/aux-inputs
    sdk/include/geniex.h                   # + geniex_TensorDataType, geniex_NamedTensor, tail fields on generate/forward-logits inputs
    sdk/plugins/qairt/src/llm.cpp          # + validation & mapping C aux list -> GenerationConfig::aux_inputs (generate + forward_logits)
    sdk/plugins/llama_cpp/src/llm.cpp      # + reject aux inputs with PARAM_NOT_SUPPORTED (no silent discard)
    bindings/python/geniex/modeling.py     # + auxiliary_inputs= dict param, strict kwargs
    bindings/python/geniex/_ffi/_api.py    # + ctypes structs/fields
    third-party/geniex-qairt/              # worktree of qualcomm/geniex-qairt-plugin, branch steering/aux-inputs
      core/include/types.h                 # + struct NamedTensor; GenerationConfig::aux_inputs; ModelConfig::aux_input_names
      core/include/llm/input_provider.h    # + class AuxTensorInputProvider
      core/src/llm/input_provider.cpp      # + impl (hasInput guard, full-capacity write, phase logging)
      core/include/llm/llm_model.h         # + aux_provider_ member, findGraphInputSpec, aux name validation
      core/src/llm/llm_model.cpp           # + provider registration, set/clear in generate/forwardLogits, isSpecialTensor aux set
      core/src/llm/llm_utils.cpp(+h)       # + isSpecialTensor overload taking declared-aux set
      core/include/pipeline/llm_pipeline.h # + graphInputSpec() accessor, forwardLogits aux param (defaulted)
      core/src/pipeline/llm_pipeline.cpp   # + impl
      core/src/llm/llm_spec_loader.cpp     # + read aux_inputs.json sidecar into ModelConfig (or plugin-side read)
      tests/core/llm/aux_input_provider_test.cpp   # NEW gtest target
      tests/core/llm/llm_model_test.cpp    # + aux end-to-end cases against stub graphs
      tests/testing/llm_fixture.hpp        # + fixture variant with alpha/steering_vector graph inputs
      examples/aux_steering_probe/         # NEW device CLI: bundle dir + alpha sweep + logit dump
  docs/                                    # plan, implementation notes, demo doc, final report
  artifacts/geniex_runtime_steering/       # machine-readable test results JSON
  device/                                  # adb push/run scripts, tiny test-graph build scripts
```

Compiled-model input contract (documented in Task 10, consumed by Tasks 8–9): every shard graph that should be steerable — both the prompt graph(s) `(prompt_|prefill_)?ar<N>_cl<M>_<s>_of_<t>` and the decode graph(s) `(token_)?ar1_cl<M>_<s>_of_<t>` — must expose additional APP_WRITE inputs `alpha` (float32 `[1]`) and `steering_vector` (float32 `[1,1,H]`), appended after canonical inputs, and the bundle dir must carry `aux_inputs.json`: `{"aux_inputs": ["alpha", "steering_vector"]}`.

---

### Task 0: Fork workspace + submodules + baseline build (WSL)

**Files:** none modified — workspace setup only.

**Interfaces — Produces:** worktrees at `geniex-steering/geniex` (done) and `geniex-steering/geniex/third-party/geniex-qairt` on branches `steering/aux-inputs`; populated `third-party/geniex-proc`; WSL Ubuntu-24.04 toolchain (gcc/cmake/ninja/cargo); a green **unpatched** `ctest -L unit` run = the stock-behavior baseline.

- [ ] Create the geniex-qairt worktree inside the fork (reference the existing submodule checkout to avoid a 500 MB network fetch), then branch: `git -C <fork>/third-party/geniex-qairt checkout -b steering/aux-inputs abf961f`.
- [ ] `git submodule update --init third-party/geniex-proc` inside the fork's geniex-qairt (network fetch of `a4558f00`).
- [ ] WSL: install `build-essential cmake ninja-build cargo` (apt) if absent.
- [ ] Configure + build tests from WSL against the fork path (`/mnt/c/...`), out-of-tree build dir `build-wsl`: `cmake -G Ninja -B build-wsl -DGENIEX_BUILD_TESTS=ON -DGENIEX_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release` then `ctest -L unit`.
- [ ] Record baseline results to `artifacts/geniex_runtime_steering/baseline_unit_tests.json`. Commit nothing yet (no changes).

### Task 1: Core types + AuxTensorInputProvider (TDD against stub graphs)

**Files:**
- Modify: `core/include/types.h` (NamedTensor struct after TensorSpec; `std::vector<NamedTensor> aux_inputs;` at tail of GenerationConfig; `std::vector<std::string> aux_input_names;` at tail of ModelConfig)
- Create: provider class in `core/include/llm/input_provider.h` + `core/src/llm/input_provider.cpp`
- Test: `tests/core/llm/aux_input_provider_test.cpp`, registered in `tests/CMakeLists.txt` next to `input_provider_test`
- Modify: `tests/testing/llm_fixture.hpp` — add optional aux tensors (`alpha` f32 `[1]`, `steering_vector` f32 `[1,1,4]`) to the fixture graph builder.

**Interfaces — Produces:**
```cpp
struct NamedTensor {                 // core/include/types.h
    std::string name;
    std::vector<uint32_t> shape;     // informational; validation is element-count based
    std::vector<float> data;         // owned copy; float32-only for now
};
class GENIEX_API AuxTensorInputProvider : public InputProvider {
   public:
    void set(std::vector<NamedTensor> tensors);   // per-request; copies owned here
    void clear();
    size_t writeCount() const;                    // diagnostics/tests: total graph writes performed
    void write(Graph& g, const LLMRunContext& ctx) override;
};
```
`write()` behavior: for each tensor — `if (!g.hasInput(t.name)) continue;` (per-shard skip, mirrors DeepstackInputProvider contract); element-count mismatch vs `g.inputSpec(t.name).elementCount()` throws `std::invalid_argument` naming tensor, expected and got counts; otherwise write **full capacity** every call (RPC buffers persist across executes — zero-fill then copy, exactly like DeepstackInputProvider does); `GENIEX_LOG_DEBUG` name/phase(prefill|decode)/graph-name/count/first-element each write.

- [ ] Write failing tests: writes reach `g.inputPtr("alpha")` bytes on a fixture graph; both prefill (`ctx.phase==0`) and decode (`ctx.phase==1`) writes happen; absent input is skipped silently; element-count mismatch throws with the tensor name in the message; `clear()` stops writing; second `set()` fully overwrites the first (no stale bytes).
- [ ] Run: `ctest -R AuxTensorInputProvider` → FAIL (class absent).
- [ ] Implement struct + provider.
- [ ] Run new tests → PASS; run full `ctest -L unit` → all green.
- [ ] Commit (geniex-qairt fork): `feat: generic auxiliary named-tensor input provider`.

### Task 2: LLMModel wiring — registration, per-request set/clear, spec-inference guard, name validation

**Files:**
- Modify: `core/include/llm/llm_model.h` (members `AuxTensorInputProvider* aux_provider_ = nullptr;` and `std::unordered_set<std::string> all_graph_input_names_`; declare `const TensorSpec* findGraphInputSpec(const std::string&) const;`)
- Modify: `core/src/llm/llm_model.cpp` (register provider in `initialize()` after `createInputProviders()` so subclass overrides can't drop it; build `all_graph_input_names_` in `onInitialized`; set/clear around `generate` body and `forwardLogits`; pass declared-aux set into spec-inference calls)
- Modify: `core/src/llm/llm_utils.cpp` + `core/include/llm/llm_utils.h` (`isSpecialTensor(name, const std::unordered_set<std::string>& declared_aux)` overload)
- Modify: `core/include/pipeline/llm_pipeline.h` + `core/src/pipeline/llm_pipeline.cpp` (`const TensorSpec* graphInputSpec(const std::string&) const;` and `forwardLogits(input_ids, all_positions, const std::vector<NamedTensor>* aux = nullptr)`)
- Test: extend `tests/core/llm/llm_model_test.cpp`.

**Interfaces — Consumes:** Task 1's `NamedTensor`, `AuxTensorInputProvider`. **Produces:** `GenerationConfig::aux_inputs` honored by `LLMModel::generate`; unknown aux name throws `std::invalid_argument` listing available graph input names; `LLMPipeline::graphInputSpec(name)` for plugin-side validation; spec inference ignores tensors named in `ModelConfig::aux_input_names`.

- [ ] Failing tests (fixture graphs with aux inputs + `ModelConfig::aux_input_names = {"alpha","steering_vector"}`):
  - `GenerateWithAuxWritesPrefillAndDecode` — generate 3 tokens with `aux_inputs={alpha:[2.5]}`; assert prefill and decode graphs' `inputPtr("alpha")` both contain 2.5f.
  - `ChangingAlphaBetweenRequestsChangesBufferWithoutReload` — generate with 0.0 then 4.0 on the same model instance; assert buffer content changed and the same `Graph` objects were reused.
  - `NoAuxPreservesStockBehavior` — bit-compare a no-aux generate's emitted tokens against a pre-patch recording (the existing `GreedyDecodeEmitsStubToken` family already pins this).
  - `UnknownAuxNameThrows`, `AuxClearedAfterGenerate` (a second no-aux generate leaves zeros in aux buffers), `SteeringVectorNotMistakenForHiddenState` (spec inference on a fixture whose graph lists `steering_vector` [1,1,4] float **before** the real in_state still picks the right in_state and hidden_size when the name is declared aux).
- [ ] Run → FAIL; implement; run new + full suite → PASS.
- [ ] Commit: `feat: per-request aux tensor binding through LLMModel generate/forwardLogits`.

### Task 3: Bundle self-declaration — `aux_inputs.json` sidecar

**Files:**
- Modify: `core/src/llm/llm_spec_loader.cpp` (+ its header) — `std::vector<std::string> readAuxInputNames(const fs::path& bundle_dir)` parsing optional `aux_inputs.json` (`{"aux_inputs": [...]}`); returns empty on absence (stock behavior); malformed file throws with path in message.
- Test: `tests/core/llm/llm_spec_loader` cases in `llm_model_test.cpp` (same target): absent → empty; well-formed → names; malformed → throw.

**Interfaces — Produces:** `readAuxInputNames()` used by the qairt plugin (Task 4) to populate `ModelConfig::aux_input_names`.

- [ ] Failing tests → implement → suite green → commit: `feat: bundle-declared auxiliary input names (aux_inputs.json)`.

### Task 4: C ABI + QAIRT plugin mapping + llama_cpp guard

**Files:**
- Modify: `sdk/include/geniex.h` — after `geniex_ProfileData`:
```c
typedef enum { GENIEX_TENSOR_DTYPE_FLOAT32 = 0 } geniex_TensorDataType;
typedef struct {
    const char*           name;       /* graph input name (non-NULL) */
    geniex_TensorDataType dtype;      /* FLOAT32 only for now */
    const int64_t*        dims;       /* rank entries; informational */
    int32_t               rank;
    const void*           data;       /* non-NULL; must stay valid for the duration of the call */
    size_t                data_byte_size;
} geniex_NamedTensor;
```
  and tail fields on BOTH `geniex_LlmGenerateInput` and `geniex_LlmForwardLogitsInput`: `const geniex_NamedTensor* aux_inputs; int32_t aux_input_count;` (append-at-tail matches the project's own convention, e.g. the sliding_window addition; zero/NULL preserves stock behavior).
- Modify: `sdk/plugins/qairt/src/llm.cpp` — in `create()`: `model_cfg.aux_input_names = readAuxInputNames(model_dir);`. New helper `mapAuxInputs(const geniex_NamedTensor*, int32_t, LLMPipeline&, std::vector<NamedTensor>& out) -> int32_t` used by both `generate()` and `forward_logits()`: NULL-data/zero-size/NULL-name → `GENIEX_ERROR_COMMON_INVALID_INPUT`; dtype ≠ FLOAT32 → `PARAM_NOT_SUPPORTED`; duplicate name → `INVALID_INPUT`; name with no matching graph input (`pipeline_->graphInputSpec(name) == nullptr`) → `INVALID_INPUT` logging the available aux-capable names; byte-size ≠ `spec->byteCount()`-equivalent element count × 4 → `INVALID_INPUT` logging expected vs got. On success copy into `gen_cfg.aux_inputs` / pass to `forwardLogits`.
- Modify: `sdk/plugins/llama_cpp/src/llm.cpp` — `if (input->aux_input_count > 0) return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;` (with error log) in generate; never silently discard.
- SDK bridge `sdk/src/llm.cpp`: NO change needed (struct copied wholesale; verify by reading).

**Interfaces — Consumes:** `readAuxInputNames`, `graphInputSpec`, `GenerationConfig::aux_inputs`. **Produces:** the public ABI used by the probe CLI (Task 6), Python bindings (Task 11), APK (Task 9).

- [ ] Implement; compile-check plugin + SDK x64 in WSL (`sdk` preset default, `GENIEX_PLUGIN_LLAMA_CPP=OFF`, cargo present for model-manager; if model-manager blocks, patch it out of the build locally and note it).
- [ ] geniex-qairt suite still green (`ctest -L unit`).
- [ ] Commit both repos: `feat: public auxiliary named-tensor ABI + qairt plugin mapping`.

### Task 5: Device probe CLI example

**Files:** Create `third-party/geniex-qairt/examples/aux_steering_probe/{aux_steering_probe.cpp, CMakeLists.txt}` (linked like `auto_llm`: `geniex_core` + `geniex-proc`).

**Behavior:** args `--bundle <dir> [--prompt <txt>] [--ids 1,2,3] [--alpha <f> ...] [--vector <path.bin>] [--logits]`. Loads the pipeline ONCE (prints a load timestamp + object address as context identity), then for each `--alpha` value in sequence: builds `GenerationConfig.aux_inputs` (`alpha` `[1]`, plus `steering_vector` `[1,1,H]` from the raw float32 `.bin` if given), runs `generate` (or `forwardLogits` with `--logits`) and prints per-run JSON lines: alpha, first 8 logits or generated text, latency ms, `aux_write_count` (from provider diagnostics via env `GENIEX_AUX_DEBUG=1` logging), plus explicit `reload_count: 0`. With no `--alpha` args it runs stock (no aux) — the unsteered baseline.

- [ ] Implement + build in WSL (x64 compile check only — it cannot execute on x64).
- [ ] Commit: `feat: aux_steering_probe example CLI`.

### Task 6: Android arm64 build + physical stock-regression on S25

**Files:** Create `geniex-steering/device/build_android.sh` (WSL: NDK r27 clang, `cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DGENIEX_BUILD_TESTS=OFF -DGENIEX_BUILD_EXAMPLES=ON` building `aux_steering_probe`), `geniex-steering/device/run_probe.ps1` (adb push probe + vendored `third-party/android` QNN libs + hexagon v79 skel to `/data/local/tmp/geniex-steering/`, set `LD_LIBRARY_PATH` + `ADSP_LIBRARY_PATH`, run).

- [ ] Download NDK r27 (Linux) into WSL if absent; build.
- [ ] Push + run the probe on the S25 against the EXISTING baked bundle (extract `qwen3-0.6b-steered-l7-alpha-m4-sm8750.zip` to a scratch dir on device — read-only source, scratch copy):
  - stock run (no aux): generates text → record as physical stock-behavior proof of the fork.
  - `--alpha 1.0` against this bundle (which has no aux graph inputs): expect the documented actionable `INVALID_INPUT` error naming the problem — proves no silent discard.
- [ ] Save both transcripts to `artifacts/geniex_runtime_steering/s25_stock_and_reject.json`. Commit scripts.

### Task 7: Tiny steering-capable LLM-shaped test context (the decisive mechanism proof)

**Files:** Create `geniex-steering/device/make_test_context/` — ONNX builders for two graphs mirroring `tests/testing/llm_fixture.hpp`'s tensor set (in_state f32 `[1,ar,8]`-style inputs, 1 KV layer in/out, attention_mask, logits `[1,ar,16]`) **plus** `alpha` f32 `[1]` and `steering_vector` f32 `[1,1,8]`, math `logits = MatMul(in + alpha*vec, W)`; graph names `prefill_ar4_cl16_1_of_1` / `token_ar1_cl16_1_of_1`; converted via QAIRT 2.45.41 x86_64-linux tools (from the SDK zip, extracted to scratch — NOT into artifacts) `qairt-converter` + `qnn-context-binary-generator` with HTP v79 offline prepare; fallback: AI Hub compile+link jobs. Plus `aux_inputs.json`, minimal `genie_config.json`, tokenizer copy, `expected_logits.py` (numpy reference).

- [ ] Build context; assemble a mini-bundle; run probe on S25: alphas `0, 1, -2, 5`, two different vectors, same loaded context.
- [ ] Assert on-host: device logits match numpy reference within HTP tolerance (≤1e-2 rel), logits differ across alphas/vectors, `reload_count: 0`, load happened once (timestamps).
- [ ] Save `artifacts/geniex_runtime_steering/s25_test_graph_proof.json`. Commit scripts + results. **This is the go/no-go gate: runtime mechanism physically proven.**

### Task 8: Steered Qwen3-0.6B bundle with runtime inputs (AI Hub, one-time recompile)

**Files:** Create `geniex-steering/device/make_qwen_steered_bundle/` — adaptation of the existing baked-export pipeline (reference: `qcom_hackathon/src/steering_poc/qualcomm/qwen_runtime_steering.py` injection math `hidden + alpha.reshape(1,1,1) * steering_vector`, and the prior GenieX export scripts): insert the Mul/Add after layer 7 (DragonNest's validated layer, vector `vector_layer_7.pt`) with `alpha`/`steering_vector` as graph *inputs* on both prompt and decode graphs, export via ai-hub-models qwen3_0_6b GenieX/QAIRT W4A16 path for `snapdragon_8_elite_for_galaxy`, QAIRT 2.45; keep aux inputs float32 if the converter allows mixed I/O, else document the quantization params chosen (alpha calibration range must cover −10..+10 for the slider).
- [ ] Submit jobs; while waiting, proceed with Tasks 10–11.
- [ ] Assemble bundle + `aux_inputs.json`; device runs: alpha sweep −10..+10 on fixed prompts, vector A (layer-7 real) vs zero vector; deterministic sampler; record text + logit deltas; verify one context load.
- [ ] Save `artifacts/geniex_runtime_steering/s25_qwen_steered_proof.json`. Commit.

### Task 9: Standalone slider APK (SteerLab)

**Files:** Create `geniex-steering/apk/` standalone Gradle project (NOT in DragonNest): single activity — bundle picker (defaults to `/sdcard/geniex-steering/qwen-bundle`), alpha slider −10..+10, prompt box, generate button, output pane, tech panel (context loads: 1, requests: N, aux writes: N, latency). JNI layer: reuse/adapt `geniex/bindings/android` if serviceable, else a thin JNI wrapper over `geniex.h` calling `geniex_llm_create/generate` with aux inputs. Ships forked `libgeniex`/plugin `.so`s + vendored QNN libs in `jniLibs`.
- [ ] Build (JDK17 + Android SDK; NDK from Task 6), install on S25, manual demo run; record screen/log evidence.
- [ ] Commit.

### Task 10: Documentation + patch inventory + final report

**Files:** Create `geniex-steering/docs/geniex_runtime_steering_implementation.md` (files changed w/ rationale, ABI, ownership/lifetime, thread-safety, prefill/decode propagation map incl. the runShard diagram, compiled-model input contract, Android build steps, upstreaming strategy), `geniex_runtime_steering_demo.md` (demo script), final report section in the implementation doc (proven/unproven/evidence-class table). Update `artifacts/geniex_runtime_steering/` index JSON.

### Task 11: Python bindings (host-side API completeness)

**Files:** Modify `bindings/python/geniex/_ffi/_api.py` (ctypes `GeniexNamedTensor`, new struct fields), `bindings/python/geniex/modeling.py` (`generate(..., auxiliary_inputs: dict[str, np.ndarray] | None = None)`): validate float32/contiguity (`np.ascontiguousarray`), keep refs alive across the call, unknown kwargs raise `TypeError`. Test file `bindings/python/tests/test_aux_inputs.py` (struct-marshalling unit tests with a fake DLL entry point where feasible; on-device/X Elite execution deferred).
- [ ] Implement + pytest green (marshalling-level) + commit.

## Known Risks

1. **W4A16 converter handling of float aux inputs (Task 8)** — mixed-precision I/O may force quantized aux inputs; mitigation: explicit io-config/calibration; worst case aux inputs are ufixed with documented scale (slider granularity ~0.08 at ±10 range) — mechanism proof unaffected (Task 7 is fp32).
2. **LLMModel load-tolerance of the tiny test context (Task 7)** — spec inference may demand tensors the mini-graph lacks; mitigation: mirror `llm_fixture.hpp`'s exact tensor set (that set is proven loadable) and iterate with `GENIEX_DUMP_IO`.
3. **Local QAIRT x86_64-linux offline HTP prepare may not produce v79 contexts** — fallback: AI Hub link jobs (proven path, `link_jgd2z0wz5` precedent).
4. **WSL path/perf issues building under `/mnt/c`** — mitigation: copy source into WSL FS (`~/geniex-steering-build`) with rsync back of changed files, or build there and keep git in Windows.
5. **`modified_input = *input` struct copy in bridge** — verify tail fields copied (plain struct copy: yes) and no other code stack-allocates the struct with designated init that would zero them (grep).
6. **APK JDK/SDK availability** — no JDK found yet; mitigation: temurin JDK17 download; Android SDK exists at `qcom_hackathon/artifacts/tools/android-sdk` (add build-tools/platform as needed via sdkmanager).

## Smallest usable milestone

Task 2's `ChangingAlphaBetweenRequestsChangesBufferWithoutReload` green on the stub = the runtime mechanism exists end-to-end in GenieX code. Task 7 = the same, physically on HTP. Everything after is demo polish.

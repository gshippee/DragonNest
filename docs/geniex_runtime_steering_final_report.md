# Final Report — GenieX Runtime Activation Steering Prototype

Date: 2026-08-07. Workspace: `qcom_hackathon2\geniex-steering`. DragonNest and
all existing demo artifacts untouched.

## Answers to the task's closing questions

**Exact API boundary changed.** The GenieX public C ABI (`sdk/include/geniex.h`):
`geniex_NamedTensor` + tail fields `aux_inputs`/`aux_input_count` on
`geniex_LlmGenerateInput` and `geniex_LlmForwardLogitsInput`, plus
`geniex_llm_get_aux_stats` (per-phase diagnostics). Below it, geniex-qairt's
`GenerationConfig` gained `aux_inputs` and `ModelConfig` gained
`aux_input_names`; `LLMPipeline::forwardLogits` gained a defaulted aux
parameter. No other signatures changed; the SDK bridge passes structs through
unchanged.

**Exact GenieX files changed.** 14 source files + 4 test files + 1 new example
across the two forked repos — full table with rationale in
`geniex_runtime_steering_implementation.md`. Commits: geniex-qairt fork
`68ed0f5` (Linux test-build fixes), `428adc0` (aux mechanism + 15 tests),
`afd4afc`/`fea4cce` (probe example), `+` per-phase AuxStats; geniex fork
`c33af05` (ABI + plugin mapping), `09f9686` (stddef fix), `cdbf0b2`
(aux-stats ABI).

**Generic or steering-specific?** Generic named-tensor support. Core/SDK know
no steering names; bundles self-declare auxiliary inputs via an
`aux_inputs.json` sidecar. Steering is the demo application.

**Prompt/decode propagation path.** One choke point: every graph execution
(prefill chunks, each decode step, forwardLogits, sliding-window re-prefill)
runs `LLMModel::runShard`'s provider loop; the always-registered
`AuxTensorInputProvider` writes every matching named input with an exact-size
full overwrite before `Graph::execute`. Per-phase counters prove both phases
receive the tensors (unit tests assert buffer bytes; the APK displays the
counters live).

**Required compiled-model input contract.** Both prompt and decode graphs must
expose APP_WRITE inputs `alpha` float32 `[1]` and `steering_vector` float32
`[1,1,H]`; the bundle ships `aux_inputs.json` naming them (also protects
spec-inference heuristics); quantized exports need activation encodings for
the new tensors (alpha ±12 for a ±10 slider; steered-residual encoding
widened by the max delta). Full detail + the ai-hub-models pipeline changes in
the implementation doc.

**Ordinary GenieX models compatible?** Yes. Zero aux inputs ⇒ stock behavior
(162/162 unit suite green incl. all 147 pre-existing tests; stock Qwen
generation exercised on-device). Bundles without the sidecar load stock.
Struct growth is tail-only per the project's own convention (recompile against
the new header, as with every prior tail extension).

**Android build implications.** Whole stack cross-compiles with NDK r27c
(geniex-qairt core, SDK + qairt plugin, Rust model-manager/tokenizers via
`aarch64-linux-android` target). APK packaging works with all `.so` files flat
in jniLibs; `GENIEX_PLUGIN_PATH`+`ADSP_LIBRARY_PATH` = nativeLibraryDir covers
plugin discovery and QNN/skel resolution. Two upstream test-build portability
fixes were needed for Linux hosts (also benefit CI diversity upstream).

**Smallest DragonNest integration surface.** Ship forked libs + a
steering-capable bundle as a new artifact with `steering_mode: runtime_vector`;
give `genie_runner.py` an in-process ctypes path (or the probe CLI) accepting
alpha/vector; un-stub `hardware_adapter.execute_runtime_steered()`; update
`steering-vectors.yaml` validated lists + `test_steering.py` boundaries after
on-target validation. Four files, no proto changes (SteeringSpec already
carries `vector_id` + `alpha`).

**Physical runtime proof achieved?** **Yes — mechanism proven on hardware.**
Galaxy S25 Ultra (SM-S938U1, Snapdragon 8 Elite, HTP v79), forked runtime,
steering-capable test context compiled ONCE offline (QAIRT 2.45.41 x86_64
tools, WSL):

- alpha ∈ {0, 1, −2, 5} and two vectors on one loaded context: device logit
  deltas match the analytic `alpha·(vec@W)` to ≤ 6.2e-3 (fp16-level), all
  4 argmax changes as predicted; `alpha=0` bit-identical to the no-aux stock
  run; 0–2 ms per steered forward pass; decode-phase generate() runs produce
  different token sequences per alpha ("/%/%" vs "$$$$" vs "&&&&");
  `load_count: 1`, `reload_count: 0` throughout.
  Machine-readable: `artifacts/geniex_runtime_steering/s25_test_graph_proof.json`.

**Full-Qwen semantic proof — also achieved on hardware.** A fresh steered
Qwen3-0.6B W4A16 export with runtime inputs at layer 7
(`bake_runtime_steering.py` + patched ai-hub-models export, seq lengths 128,1
so BOTH phases exist) compiled via AI Hub and ran on the S25 through BOTH the
geniex-qairt CLI path and the full public C ABI inside the SteerLab APK
(`com.dragonnest.geniexsteeringlab`, ±10 slider, real/random/off vector
selector, live per-phase diagnostics):

- One context load (997–1083 ms), zero reloads across every request.
- Greedy fixed prompt: α=0 → normal explanation; α=+10 → elaborate structured
  prose to the 96-token cap; α=−10 → one terse sentence, early EOS at 21
  tokens, 188 ms. Norm-matched random-vector control does NOT reproduce the
  concise/verbose axis — the effect is direction-specific.
- Diagnostics from `geniex_llm_get_aux_stats` per request: 2 prefill aux
  writes + 2×decode-steps decode writes, exactly as designed; ~124 tok/s.
- Stock requests on the steered bundle produce clean unsteered text (0 aux
  writes) thanks to the zero-fill-at-load guarantee.
  Machine-readable: `artifacts/geniex_runtime_steering/s25_qwen_steered_proof.json`
  (+ APK screenshots under `screenshots/`).

**Concrete blockers (none fatal).**
1. Pre-existing upstream defect (reproduced with UNPATCHED baseline): current
   geniex-qairt cannot load the 2026-08-04-era prompt-only S25 bundles (no
   `token_ar1` graphs → out-of-range during init). Consequence: the old
   baked-demo bundles cannot serve as the runtime-steering base; the fresh
   dual-phase export avoids it. Old bundles remain untouched and continue to
   work with the runtimes they shipped with.
2. W4A16 aux-input quantization (INT16 symmetric) bounds alpha resolution to
   ~3.7e-4 — irrelevant at slider granularity.
3. Host x64 cannot execute QNN (no CPU backend exists in this stack) — host
   proof runs against the link-time QnnApi stub; physical truth comes from the
   S25.

## Evidence classes

- **Physically verified (S25/HTP v79):** forked runtime loads/generates stock
  Qwen path attempts, tiny-model steering sweeps incl. decode phase, aux
  rejection errors on non-steering bundles, one-load/no-reload counters.
- **Host-tested (x64 Linux, QnnApi stub):** 162-test unit suite; per-phase
  binding; validation/lifecycle/spec-inference-guard behaviors; sidecar parse.
- **Statically verified:** llama_cpp plugin rejection (submodule not built);
  Windows-ARM64/X Elite builds (not attempted this session).
- **Inferred:** old-bundle loader defect root cause (missing decode graphs
  matches the failure signature; not yet bisected upstream).

## Thread-safety and per-token steering

Per existing GenieX contract a handle serves one request at a time; aux state
is request-scoped RAII, no globals; SteerLab serializes with a mutex.
Per-token updates are out of scope but the provider is consulted every decode
step, so a per-step setter is a natural future extension.

## Test inventory

`ctest -L unit`: 162 tests (147 upstream + 15 new) — see
`artifacts/geniex_runtime_steering/baseline_unit_tests.json` and suite runs in
the build logs. New coverage: provider round-trips (float + quantized
targets), absent-input skip, size-mismatch errors, clear/replace lifecycle,
end-to-end prefill+decode buffer assertions, alpha-change-without-reload,
unknown/duplicate/empty rejection, post-request cleanup, forwardLogits
binding, per-phase stats, spec-inference guard positive + negative controls,
sidecar parse (absent/wellformed/malformed).

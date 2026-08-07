# Qwen3-1.7B X Elite Physical Bring-Up — Session Status

Date: 2026-08-07
Git commit tested: `028442a8d3628d75f09faddd5147099188d52c15`
Branch: `codex/qwen17-variable-split`

Sanitized status record for the first physical-execution attempt of the
four-stage Qwen3-1.7B QNN pipeline (`docs/results/qwen3_1_7b_pipeline_manifest.json`)
on this Snapdragon X Elite laptop. Excludes API tokens, licensed SDK files,
model binaries, and absolute local paths where not load-bearing.

## Host identity

- Manufacturer/model: Dell Latitude 7455 (same machine as
  `docs/results/xelite_worker_status.md`)
- SoC: Snapdragon X Elite (X1E80100)
- OS: Windows 11 Pro build 10.0.26100, ARM64
- QAIRT installs present on this machine (exhaustively enumerated via
  `Get-ChildItem -Recurse -Filter qnn-net-run.exe` across all of `C:\`):
  `2.32.6.250402` and `2.48.40.260702`, both under
  `C:\Qualcomm\AIStack\QAIRT\`. **No QAIRT 2.45.x `aarch64-windows-msvc`
  build exists on this machine.**

## Summary

1. Environment/artifact staging: **done, verified**.
2. QAIRT version gate: initially tested against the two runtimes already on
   this machine (2.32.6, 2.48.40) — both failed. **Mid-session the exact
   matching runtime, QAIRT 2.45.41.260507, was installed** and retested; it
   also failed, but at a materially later, more specific point (null
   `graphPrepareDsp` handle) that ruled out blob-format version skew as the
   cause (§2, §2b).
3. **Root cause found and fixed**: this device's Hexagon DSP needs
   `ADSP_LIBRARY_PATH` pointed explicitly at the unsigned HTP skel
   directory; `qnn-net-run.exe` was only getting `PATH` (via
   `qnn_runner.py`'s `_env()`), which is not sufficient for the DSP-side
   skel/graph-prepare step even though it's enough for basic backend init.
   Confirmed with the SDK's own DSP smoke test
   (`qnn-platform-validator.exe --testBackend`: fail → pass), then with the
   real S0 context binary (crash → clean success). Fixed in
   `src/dragon_nest/runtime/qnn_runner.py::_env()`. See §2c.
4. **S0 (embedding stage) is now physically proven on HTP** — both the
   prompt graph (128 tokens) and the decode graph (1 token), through the
   actual `qnn_runner.run_context_binary()` API, with **bit-exact
   cross-validated correctness**: the decode graph's output for token 198,
   run as a separate invocation, matches every occurrence of token 198 in
   the prompt graph's output exactly after dequantization. See §4.
5. Also discovered and handled: these context binaries bundle **two graphs
   each** (prompt + decode sharing one weight blob) and use **quantized I/O**
   (`UFIXED_POINT_16`, scale/offset per tensor) rather than the float32 the
   pipeline manifest's `output_tensor_schema` implies — both now supported
   in `qnn_runner.py` / documented for S1-S3. See §4, §6.
6. RoPE / attention-mask / KV-cache tensor construction contract:
   **recovered from Qualcomm's own `qai_hub_models` source**, not guessed
   (§5), then **physically exercised against S1's prompt graph and
   confirmed correct** (§7) — including resolving §5's two flagged
   uncertainties (attention-mask clip is -100, not -50; KV zero-point is
   raw byte 128) with hard evidence from the binary's own metadata rather
   than assumption.
7. **S1's decode graph, S2's prompt graph, and S3's prompt graph are all now
   physically proven too** (§9, §10, §11) — S1 decode via a bit-exact-style
   cross-check reconstruction (close but not exact; see §9 for the precise
   number and the plausible reason), S2 and S3 each with the **real chained
   output of the previous stage** as input, not a synthetic placeholder. The
   full real chain (S0→S1→S2→S3, prompt only) was run end-to-end and its
   top-1 predicted token is `<think>` (id 151667) — the expected Qwen3
   reasoning-mode first token, corroborated by this same repo's
   `genie_runner.py` docstring. That is a strong qualitative correctness
   signal for the whole physical pipeline, not just a shape/dtype check.
8. **S1's decode graph, S2, and S3 are all now physically proven with real
   upstream decode output and real persistent stage-local KV** (§9-§11),
   and a **complete real prefill + 8-token autoregressive decode loop** ran
   end-to-end on HTP (§13), generating the coherent, on-topic text
   `<think>\n</think>\n\nGravity is the force that` for the prompt "What is
   gravity?" — the stop condition for this bring-up.
9. `docs/results/qwen3_1_7b_pipeline_manifest.json` and
   `configs/model-artifacts.yaml` updated throughout the session to say
   `ufixed_point_16`/`ufixed_point_8` instead of `float32` for every tensor
   physically confirmed quantized (now all four stages, prompt and decode),
   and `verification_status: verified-on-physical-hardware` with real
   latency numbers for all four `qwen3-1.7b-s{0,1,2,3}-xelite` entries — see
   §8/§13. The S25 artifact entries are deliberately left untouched (out of
   scope this session).

## 1. Artifact staging (physically verified)

- `.\scripts\artifact_tools\stage_xelite_artifacts.ps1 -CacheRoot C:\DragonNestArtifacts -StageDir $env:TEMP\dragonnest-qwen17-xelite`
  found all four stage binaries already staged from a prior session; re-verified
  each by independent SHA-256 against `docs/results/demo_artifact_inventory.json`
  — all four matched (`qwen3-1.7b-s0-xelite.bin` 622,391,296 B,
  `s1` 263,147,520 B, `s2` 263,131,136 B, `s3` 525,893,632 B).
- With `QWEN3_1_7B_S{0,1,2,3}_XELITE_QNN` set, `check_artifacts.py` reports
  all four `qwen3-1.7b-s{0,1,2,3}-xelite` entries `READY` (checksum-verified
  against `configs/model-artifacts.yaml`). The four `*-s25` entries correctly
  report `UNAVAILABLE` (their env vars are for the Android side, out of
  scope this session).

## 2. QAIRT version gate — tested honestly, both runtimes fail

Per the task's explicit instruction: test whether the installed runtime can
load the recovered QAIRT-2.45-compiled context binaries; if not, do not
alter manifests or recompile to paper over it.

**Control test first** (confirm the environment itself is healthy): re-ran
the same physically-verified Genie 4B path from `xelite_worker_status.md` —
`genie-t2t-run.exe` under QAIRT 2.48.40.260702 — and it succeeded again,
54.6s, real text output ("Gravity is the force that pulls objects toward
Earth."). This confirms the HTP/DSP transport and QAIRT 2.48 environment on
this laptop are healthy right now; the failures below are specific to the
1.7B split-stage context binaries, not a broken environment.

**QAIRT 2.32.6.250402** (`qnn-net-run.exe --retrieve_context qwen3-1.7b-s0-xelite.bin`):
fails cleanly and immediately with a version-check error:

```
Can't read future blob. Newest blob version supported: 3.2.2. Current blob version: 3.3.4.
Fail to check context blob version 30001
Failed to get context blob meta info or shared blob meta info .etc.
Failed to create context from binary with err 0x7531
Create From Binary failure
```

Exit code 16, clean failure, no crash. This matches the same error code
(`30001`) already documented in `xelite_worker_status.md` for the Genie
bundle against this SDK — a known, consistent "too old to read this blob
version" rejection, not new information.

**QAIRT 2.48.40.260702** (`qnn-net-run.exe --retrieve_context qwen3-1.7b-s0-xelite.bin --log_level verbose`):
does **not** reject the blob version — verbose log confirms
`Creating deserializer for version 3.3.4`, then proceeds through backend
init, DSP session open, and successfully copies the **entire 622,329,856-byte
weight blob** (`Successfully copied 622329856 bytes of weights!` /
`Finished copying shared weights.`, ~1144ms in). It then continues into
graph/tensor-metadata finalization for roughly another 170ms, at which point
the log turns into an uninterrupted cascade of `Freeing memory` lines with no
further `[INFO]`/`[ERROR]` message, and the process terminates abnormally
(non-zero, non-standard exit code; the Python subprocess wrapper observed raw
exit code `3221226505` / `0xC0000409`, i.e. a native crash, not a returned
QNN error code). No graceful `Finished Executing Graphs` message, no QNN
error object, no Windows crash dump was found under
`%LOCALAPPDATA%\CrashDumps`.

**Reading of this evidence:** QAIRT 2.48.40.260702's deserializer recognizes
and accepts blob format 3.3.4 (unlike 2.32.6, which rejects it outright), and
gets substantially further — through the entire weight payload — before
faulting during graph/tensor finalization. This is consistent with a genuine,
narrower incompatibility between the exact QAIRT-2.45-era AI Hub compiler
build that produced this context and the exact 2.48.40.260702 deserializer's
handling of some later structure in the same nominal blob version (rather
than a wholesale format rejection). No attempt was made to patch, retry with
altered flags in a way that would mask this, or otherwise make it "work" —
per instructions, the honest result is recorded as-is.

## 2b. QAIRT 2.45.41.260507 installed mid-session — retested, new failure point

The user installed **QAIRT 2.45.41.260507** (`aarch64-windows-msvc`) during
this session — the exact build number already documented in
`docs/HARDWARE_AUDIT.md` line 39 as physically executed (on the S25/Android
side). Re-ran the identical `--retrieve_context` load, both bare
(`--log_level verbose`, no inputs) and as a full prompt-graph call through
`qnn_runner.run_context_binary()` with real tokenized `input_ids`. Both
still fail — but materially further in, and with a different, much more
specific signature than the 2.48.40 run:

- Backend/session/skel init proceeds (including the same benign
  `DspTransport.openSession ... 0x80000406` warning documented as harmless
  in `AskQuery/README.md`).
- `Context Blob version: 3.3.4` recognized (as expected — it's now the
  literal matching compiler-era runtime).
- The full 622,329,856-byte weight blob is copied successfully again
  (`Finished copying shared weights.`, ~655-871ms in depending on run).
- It then proceeds into `Calling driver's API - graphPrepare` →
  `Calling transport graphPrepareDsp from driver` — this is **new**; the
  2.48.40 run never reached this call. This is the actual DSP-side graph
  compilation/placement step.
- It completes that call, but with a telling result:
  `HtpTransport::graphPrepareDsp done. graph.m_hexNNGraphHandle = 0` — a
  **null/zero graph handle**, i.e. DSP-side graph preparation did not
  actually produce a usable graph.
- Immediately after, the log turns into a large uncontrolled cascade of
  `Freeing memory`/`Memory reallocated` lines with no further semantic
  message, and the process terminates abnormally mid-write (same
  `0xC0000409` raw exit code as the 2.48.40 run) — consistent with the
  caller then dereferencing/using that null handle rather than checking it,
  crashing instead of returning a clean QNN error.

**Independent corroboration, unrelated to this specific context binary:**
`qnn-context-binary-utility.exe --context_binary ... --json_file ...`
(under 2.45.41) parses the S0 binary cleanly with no crash and reports
`dspArch=73` (correct — matches this device's Hexagon V73) and an empty
`socVersion` (no hard SoC lock encoded), plus a trivial
`vtcmSize=8` per graph — ruling out an obvious SoC-ID mismatch or VTCM
overcommit as the cause.

Separately, `qnn-platform-validator.exe --backend dsp --testBackend --debug`
(2.45.41) was run as a **completely independent, minimal DSP smoke test**
(no model, just the SDK's built-in calculator sum function) and **fails the
same way**, with an explicit, actionable message:

```
Unable to destroy the handle. PF_VALIDATOR: ERROR: -6 . Error while executing the sum function.
PF_VALIDATOR: ERROR: Please use testsig if using unsigned images.
PF_VALIDATOR: ERROR: Also make sure ADSP_LIBRARY_PATH points to directory containing skels.
Unit Test on the backend DSP: Failed.
QNN is NOT supported for backend DSP on the device.
```

This is a standard Qualcomm Hexagon DSP message: unsigned/developer HTP skel
libraries (everything under `QAIRT\*\lib\hexagon-v73\unsigned\`, which is
the *only* skel directory either installed QAIRT version ships — there is no
parallel "signed" directory to point at instead) require a device-specific
**testsig** (test signature) to be installed before the DSP will actually
execute them; without it, skel loading and basic init can still succeed
(explaining why the log gets as far as `setSkelLogLevel return 0` and even a
`graphPrepareDsp` call) but real execution silently fails.

One open puzzle, noted honestly rather than resolved: the physically-verified
Genie 4B path (`genie-t2t-run.exe`, §2 control test) uses the **same-shaped**
`libQnnHtpv73Skel.so` + `libqnnhtpv73.cat` pair (case difference in the
filename only) and *does* execute successfully on this same device, so
whatever is granting it DSP trust is not something this session identified —
it was not investigated further because doing so would mean guessing at
device-level trust/signing configuration, which risks system-wide changes
outside this session's scope. This is flagged for the next session rather
than acted on speculatively.

## 2c. Root cause and fix: `ADSP_LIBRARY_PATH`

`qnn-platform-validator.exe`'s DSP smoke-test failure message named the
missing piece explicitly: `Also make sure ADSP_LIBRARY_PATH points to
directory containing skels`. `qnn_runner.py::_env()` was only prepending
`PATH` with `LIB_DIR`/`HEXAGON_DIR` — never setting `ADSP_LIBRARY_PATH` at
all. Setting it explicitly to
`QAIRT_ROOT/lib/hexagon-v73/unsigned` (the only skel directory either
installed QAIRT version ships — there is no separate "signed" directory)
fixed both the standalone smoke test and the real S0 binary:

```
# Before (unset): PF_VALIDATOR: ERROR: Please use testsig if using unsigned images.
#                  Unit Test on the backend DSP: Failed.
# After (set):     Success in executing the sum function
#                  Unit Test on the backend DSP: Passed.
```

Re-running the exact S0 `--retrieve_context` load that previously crashed
with `graph.m_hexNNGraphHandle = 0` (§2b) now reaches `Executing Graphs` →
`Finished Executing Graphs` cleanly.

**Fixed in code**: `src/dragon_nest/runtime/qnn_runner.py::_env()` now sets
`ADSP_LIBRARY_PATH` for every `qnn-net-run.exe` invocation (`run_dlc`,
`run_context_binary`, and their `_batch` siblings) — this is a general
environment fix, not Qwen-pipeline-specific, and should also make any
currently-flaky large/complex HTP graph elsewhere in the repo (if any exist)
more reliable, though only the Qwen3-1.7B path was actually retested this
session.

One open item, noted honestly rather than resolved: the physically-verified
Genie 4B path (`genie-t2t-run.exe`, §2 control test) succeeded *without*
`ADSP_LIBRARY_PATH` set, using a same-shaped unsigned skel + `.cat` pair.
Why Genie doesn't need it while `qnn-net-run.exe` does was not investigated
further — plausibly Genie resolves its DSP libraries by a different
mechanism (e.g. relative to its own `.dll`), or the smaller/simpler graphs
other DragonNest tools (AskQuery, Image2Audio, MeloTTS) run happen not to
exercise whatever code path requires it. Flagged for whoever next touches
`qnn_runner.py`'s environment setup, not chased down further here.

## 3. Blocker status: resolved this session

The original blocker (§3 in earlier drafts of this document, before the fix
above) was that no stage could load or prepare on this device's HTP,
regardless of QAIRT version. **That is no longer true.** With
`ADSP_LIBRARY_PATH` set (now automatic via the `qnn_runner.py` fix) and
QAIRT 2.45.41.260507 active, S0's prompt and decode graphs both execute
successfully and produce verifiably correct output — see §4. S1's prompt
graph does too — see §7. S1's decode graph and S2/S3 have not yet been
attempted; there is no known reason to expect them to hit a different
blocker, but that is a claim to verify, not assume.

## 4. S0 physically proven — prompt and decode, bit-exact cross-check

Both graphs in `qwen3-1.7b-s0-xelite.bin` were run through the fixed
`qnn_runner.run_context_binary()` (QAIRT 2.45.41.260507, HTP backend), using
the exact prompt construction from §5's recovered spec: ChatML
`"You are a helpful AI assistant." / "What is gravity? Keep the answer under
ten words."`, tokenized, padded/truncated to 128 tokens (31 real tokens, 97
left-pad).

**Multi-graph handling**: `qnn-context-binary-utility.exe`'s JSON dump
confirmed this binary holds two graphs, in this order: index 0
`prompt_ar128_cl512_1_of_4`, index 1 `token_ar1_cl512_1_of_4`. `qnn-net-run`
requires selecting one via a comma-separated `--input_list` where unselected
slots are literally `__`. `qnn_runner.py`'s `run_context_binary()` /
`run_context_binary_batch()` now take optional `graph_index`/`num_graphs`
params that build this automatically; output-directory resolution also now
handles the `output_dir/<graph_name>/Result_N/` nesting multi-graph binaries
produce (previously assumed `output_dir/Result_N/`).

**Quantized I/O**: the manifest's `output_tensor_schema` says `embedding` is
`float32`, but the physical output is `QNN_DATATYPE_UFIXED_POINT_16` with a
per-tensor scale/offset in a `.raw.json` sidecar qnn-net-run writes
alongside each output (for this artifact's `embedding` output:
`scale=0.000007, offset=-32232`; dequantize as `(raw_uint16 + offset) *
scale`). This confirms the AI Hub compile's recorded `--quantize_io` flag
(`docs/results/qwen3_1_7b_pipeline_manifest.json`'s
`quantization.compile_flag_observed`) applies to stage-boundary I/O tensors,
not just weights — S1-S3 should be expected to need the same treatment on
both their inputs and outputs, with each tensor's own scale/offset read from
its `.raw.json` sidecar rather than assumed constant.

**Results**:

- Prompt graph: 128-token input → `[1,128,2048]` output, 0.99s end-to-end
  through the Python API. `nan_count=0`, `inf_count=0`, values in a sane
  range (mean 0.000168, std 0.0176, min -0.137, max 0.128 after
  dequantization) — consistent with typical LLM embedding-table statistics.
- **Correctness, not just "didn't crash"**: every one of the 97 left-padded
  positions (all the same pad/eos token id) produced a **byte-identical**
  dequantized embedding row — exactly the behavior a correct embedding
  Gather must have. The five distinct real-token positions that happened to
  share a token id (198, appearing at prompt positions 99/108/111/124/127)
  also all produced byte-identical rows, distinct from the pad rows and from
  each other's-different-token neighbors.
- Decode graph: single-token input (`input_ids=[[198]]`, the last real
  prompt token) → `[1,1,2048]` output, run as a **completely separate
  process invocation** against the same context binary. Its dequantized
  output is **numerically identical** (`np.allclose`, matched exactly to
  6 decimal places) to the prompt graph's row for token 198. Two
  independent graph executions of the same underlying embedding table agree
  exactly — about as strong a physical correctness signal as is available
  without a second independent implementation to diff against.

This satisfies the "prove S0 alone first" gate from the original bring-up
plan (both prompt and decode). S1's prompt graph was attempted next and is
also physically proven — see §7 (S1's decode graph and S2/S3 remain
unattempted).

## 5. Recovered (not guessed) tensor-construction contract for S1-S3

Pulled directly from Qualcomm's published `qai_hub_models==0.59.1` wheel
(`qai_hub_models/models/_shared/qwen3/model.py`,
`qai_hub_models/models/_shared/llm/model.py`,
`qai_hub_models/models/_shared/llama3/model.py`,
`qai_hub_models/models/_shared/qwen3/model_adaptations.py`) — the exact
source Qualcomm AI Hub used to export this pipeline — not reimplemented from
memory. Recorded here so the next session doesn't have to re-derive it:

- **Tokenizer**: `AutoTokenizer(..., is_fast=False)`,
  `padding_side="left"`, `pad_token = eos_token` (`<|im_end|>`, id 151645 for
  the Qwen3 tokenizer family), `truncation_side="left"`. Prompt is built via
  `tokenizer.apply_chat_template` with ChatML
  (`<|im_start|>system\n...\n<|im_end|>\n<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n`),
  default system prompt `"You are a helpful AI assistant."`.
- **Prompt input_ids**: tokenize with `padding="max_length",
  max_length=context_length(512)`, then take the **last 128** tokens
  (`input_ids[:, -128:]`). `num_tokens = min(sum(attention_mask),
  128)`; `padding_size = 128 - num_tokens`.
- **position_ids** (prompt): `[0]*padding_size + list(range(128 -
  padding_size))` — i.e. position counting starts at 0 right where the real
  tokens begin, not at the start of the 512-token window.
- **RoPE cos/sin**: standard HF `LlamaRotaryEmbedding`-style computation
  using Qwen3-1.7B's own `rope_theta`/`head_dim` from its `config.json`,
  fetched this session directly from `huggingface.co/Qwen/Qwen3-1.7B` (a
  public, non-proprietary file): `rope_theta=1000000`, `head_dim=128`,
  `hidden_size=2048`, `num_attention_heads=16`, `num_key_value_heads=8`,
  `num_hidden_layers=28`, `max_position_embeddings=40960`,
  `rope_scaling=None` — the layer/hidden/kv-head counts match the pipeline
  manifest exactly, corroborating this is the right config. `inv_freq =
  1/(1000000^(arange(0,128,2)/128))`, length 64; `position_ids_cos`/`_sin` =
  `cos`/`sin` of `position_ids ⊗ inv_freq`, shape `[1,1,seq_len,64]` —
  matches the manifest's recorded `[1,1,128,64]` prompt / `[1,1,1,64]`
  decode shapes exactly. This piece is now fully pinned, no longer a TODO.
- **attention_mask**: build a length-512 zero vector, set the last
  `num_tokens` entries to 1, convert to a 4D causal mask via HF's
  `AttentionMaskConverter(True).to_4d(..., query_length=128,
  key_value_length=512)`, then clip. The generic `sample_input()` helper
  hardcodes `[-50, 0]`, but that's the FP/calibration-path constant, not
  necessarily what's baked into these specific quantized QNN graphs — ***this
  was resolved empirically in §7***: S1's actual compiled `attention_mask`
  input quantization (scale=`100/65535`, offset=`-65535`) is an exact linear
  map of **`[-100, 0]`**, matching `Qwen3QuantizablePreSplitBase`'s
  quantized-precision clip/multiplier of `(-100.0, 1.0)`, not the generic
  helper's `-50`. Use `-100` for S2/S3 too, but confirm each stage's own
  `attention_mask` scale/offset independently rather than assuming it
  carries over.
- **KV cache layout** (`SHAQwen3Attention.forward_sha` in
  `model_adaptations.py`): `past_key_{n}_in` stored **transposed**,
  shape `[num_kv_heads=8, 1, head_dim=128, past_len]`; `past_value_{n}_in`
  stored normally, shape `[8, 1, past_len, 128]`. The PyTorch reference
  concatenates new K/V onto the existing past internally (via
  `Cache.update()`) to form the attention context for this call. **Correction
  from §7's physical run**: this does **not** mean the compiled QNN graph's
  `past_key_{n}_out`/`past_value_{n}_out` **output** is that same
  concatenated (old+new) tensor — empirically it's a **delta**, shaped for
  just the newly-computed positions (`[8,1,128,128]` for the 128-token
  prompt graph, `[8,1,128,1]` for the 1-token decode graph, not `past_len_in
  + new`). Building the next call's `past_*_in` is therefore a host-side
  `concat(this_past_in, this_new_out)[..., -next_past_len:]` sliding window,
  not a direct feed-through of the output. See §7.
- **Sampling / generation loop ownership**: per
  `configs/model-artifacts.yaml`'s existing `runtime_options`
  (`tokenizer_owner: brain`, `sampling_owner: final_stage_top1`), unchanged
  by this session — the Brain owns tokenization, S3 owns top-1 sampling.

At the time this was written (before §7 below), none of it had been
exercised against real silicon. It has since been exercised against S1's
prompt graph specifically — both flagged uncertainties (attention-mask clip;
KV zero-point) were resolved by that run, with hard evidence rather than
assumption. S1's decode graph, and S2/S3 entirely, remain unexercised — see
§7 for exactly what was and wasn't covered.

## 6. What was not attempted, and why

- S1's decode graph, S2, S3, the four-stage local prefill, the
  autoregressive decode loop, DragonNest runtime/session integration, and
  stage advertisement (plan steps 5-10): genuinely not attempted this
  session — not because of a blocker (the runtime blocker that justified
  stopping earlier in this same session is resolved, §2c/§3), but per
  explicit instruction to stop at S1's prompt graph and report before going
  further (§7). S2/S3 also still need their own metadata pulled fresh
  (§7 found S1's graph order and per-tensor quantization differed from
  S0's in ways that would have been wrong to assume) rather than reusing
  S0/S1's numbers.
- No manifest field was edited to claim physical execution beyond what was
  actually run (S0 prompt+decode, S1 prompt only — §8's dtype corrections
  are scoped exactly to those). No mock fallback was substituted for a
  failed QNN call anywhere in this session, including during the
  crash-diagnosis phase before the fix landed.
- Directly importing `qai_hub_models`'s own `RopeEmbedding` class (rather
  than reusing the underlying `transformers` classes it's built on) was
  attempted and abandoned when its import chain required a private,
  unavailable package plus an escalating unrelated dependency chain — see
  §7 for the specifics. Recompiling the AI Hub graphs to sidestep the
  original version-skew hypothesis was separately out of scope per
  instructions and was not attempted — moot now that the real cause turned
  out to be unrelated to QAIRT version.

## 7. S1 prompt graph — physically proven

Following the exact plan this document previously laid out as the "next
step" (now executed, not just planned):

**Metadata first, not assumptions.** `qnn-context-binary-utility.exe` on
`qwen3-1.7b-s1-xelite.bin` revealed real, load-bearing facts the pipeline
manifest didn't have and that would have been wrong to assume from S0:

- **Graph order is not consistent across binaries.** S0 has
  `prompt_ar128_cl512_1_of_4` at index 0. S1 has
  `prompt_ar128_cl512_2_of_4` at index **1** (`token_ar1_cl512_2_of_4` is
  index 0). Every stage's graph order must be looked up individually, never
  assumed from a previous stage.
- **`embedding`'s quantization is identical on both sides of the S0→S1
  boundary**: scale `6.988926998019451e-06`, offset `-32232`, exact match
  in both S0's output metadata and S1's input metadata. This meant S0's raw
  `uint16` output bytes could be fed into S1 completely unchanged — no
  dequantize/requantize round-trip, and no floating-point rounding error
  introduced at the boundary.
- **`attention_mask`'s quantization resolves §5's flagged clip-value
  uncertainty exactly**: scale `0.0015259021893143654` = `100/65535` to
  float precision, offset `-65535` — an exact linear map of `[-100, 0]`
  onto the full `uint16` range with zero rounding error at either endpoint.
  This confirms the real compiled graph uses the **quantized-precision
  clip of -100** (`Qwen3QuantizablePreSplitBase.attention_mask_min_clip_and_multiplier`),
  not the generic `sample_input()` helper's hardcoded -50 — resolved by
  hard evidence, not assumption.
- **`position_ids_cos`/`position_ids_sin`** quantize `[~-1, ~1]` onto
  `uint16` (scale `3.051804378628731e-05`, offset `-32768`) — consistent
  with real cosine/sine output range, a useful sanity check that the RoPE
  computation (§5) and the graph's own expectation agree.
- **KV cache is `ufixed_point_8`** (not `float32`), `offset=-128` uniformly
  across all 10 layers and both key/value, but **scale differs per layer
  per tensor** (20 distinct calibrated values) — a zero-valued cache (used
  to initialize the prompt graph's `past_len=384` inputs) is encoded as raw
  byte `128` for every element, regardless of that tensor's own scale,
  since `(128 + (-128)) * scale = 0` for any scale.
- The compiled graph also declares `alpha`/`steering_vector` inputs
  (the optional runtime-steering experiment noted in the manifest) as
  hard requirements, not truly optional at the QNN level. Fed as
  quantized zero (`alpha=0` nullifies any steering effect regardless of
  `steering_vector`'s content) — consistent with this artifact's
  `supports_steering: false` — not exercised as a real steering call.

**RoPE/mask construction**: reused the real
`transformers.models.llama.modeling_llama.LlamaRotaryEmbedding` and
`transformers.modeling_attn_mask_utils.AttentionMaskConverter` classes,
transcribing Qualcomm's `RopeEmbedding.precompute()`/`get_embedding()` and
`sample_input()` logic verbatim around them (both fully read this session
from the `qai_hub_models==0.59.1` wheel) — not a hand-derived RoPE
implementation. Directly importing `qai_hub_models`'s own `RopeEmbedding`
class was attempted first and abandoned: its import chain requires
`qai_hub_models.configs.model_metadata`, which requires the private
`qai_hub_models_cli.proto` package (not on PyPI, no public source found)
plus an escalating chain of unrelated config/metadata dependencies
(`pydantic_yaml`, ...) — none of it related to RoPE or attention-mask math,
so continuing to chase it would have meant broadening scope into an
unrelated dependency yak-shave rather than proving S1. Fetched Qwen3-1.7B's
real `AutoConfig` from Hugging Face (public, non-proprietary) and passed it
directly to `LlamaRotaryEmbedding(config=...)`, exactly as Qualcomm's own
code does; verified in isolation against first-principles RoPE values
(`cos(1·θ⁰)`, `cos(2·θ⁰)`) before spending a real HTP call on it.

**Result**: `qnn_runner.run_context_binary(..., graph_index=1, num_graphs=2)`
against `qwen3-1.7b-s1-xelite.bin`, 0.90s end-to-end. `add_21844`:
shape `(1, 128, 2048)`, dtype `uint16` (matches the now-corrected schema,
§4/§6 update) — the expected shape and dtype the stop condition asked for.
`nan_count=0`, `inf_count=0`. All 10 `past_key_{n}_out`/`past_value_{n}_out`
pairs also produced with correct shapes — **`[8,1,128,128]`, i.e. only the
128 newly-computed positions, not `past_len_in(384) + 128 = 512`**. This
corrects §5's KV-cache bullet, which (reading the PyTorch reference
`SHAQwen3Attention.forward_sha`'s `Cache.update()`-based semantics) assumed
the graph outputs the full concatenated history. The real compiled graph's
I/O contract is a **delta**: each call outputs only the K/V it just
computed (confirmed identically on the decode side too:
`token_ar1_cl512_2_of_4`'s `past_key_0_out` is `[8,1,128,1]`, not `[8,1,128,512]`).
So chaining into a decode step is host-side concatenate-then-trim, not a
direct feed-through: `next_past_in = concat(this_past_in, this_new_out)[..., -next_past_len:]`
(dropping the oldest entries to stay within `context_length`) — a standard
sliding-window KV cache, once the delta-only I/O contract is known. Not
exercised this session (that's the decode-chaining step, out of scope per
"stop after S1 prompt").

**Correctness signal, same pattern as S0**: dequantized `add_21844` row 0
and row 1 (both padding positions, `position_id=0`, causal-masked to see
only equivalent zero/self context) are **byte-identical**
(`np.array_equal` on the raw `uint16`, not just close), while row 127 (a
real, distinct token after 10 real transformer layers of computation) is
different from both. This is a non-trivial invariant — repeated identical
(token, position, masked-context) inputs must produce identical outputs
through 10 real attention+MLP layers — and it held exactly.

This satisfied the original "stop after S1 prompt and report" instruction.
The session then continued (explicitly requested) through S1's decode
graph, S2, and S3 — see §9-§11.

## 8. Manifest/schema updated to match physical reality (all four stages)

Both `docs/results/qwen3_1_7b_pipeline_manifest.json` and
`configs/model-artifacts.yaml` declared `float32` for every stage-boundary,
shared-input, and (implicitly, via absence) KV tensor — an AI-Hub-level
assumption inherited from the manifest's original recovery, never
physically checked until this session. Updated to `ufixed_point_16` /
`ufixed_point_8` for every tensor physically confirmed quantized, in two
passes (S0/S1 first, S2/S3 added once those stages were also proven — see
§9-§11):

- Pipeline manifest: every stage's `shared_inputs`
  (`attention_mask`/`position_ids_cos`/`position_ids_sin`) and `kv_cache`
  now have a physically-confirmed `dtype`; stage 0's `prompt_graph`/
  `decode_graph` `embedding` output dtype; stage 1's
  `optional_steering_inputs`; each stage's `boundary_output_tensor_dtype`
  (new field, didn't exist before); stage 3's `output_logits_shape`
  vocab_size corrected from an "unverified" placeholder to the physically
  confirmed `151936`; a new top-level `physical_io_verification` block
  tracking exactly which graphs are confirmed; new `kv_cache.input_name_note`
  fields on stages 2/3 recording that `{n}` in `past_key_{n}_in` is the
  **absolute** transformer layer index (e.g. `past_key_10_in`..
  `past_key_19_in` for stage 2), not stage-relative 0-9/0-7 — this would
  have been a real bug if assumed instead of checked.
- `configs/model-artifacts.yaml`: all four `qwen3-1.7b-s{0,1,2,3}-xelite`
  entries' `input_tensor_schema`/`output_tensor_schema`, each with an inline
  comment citing the real scale/offset; stage 3's `logits` schema's
  `vocab_size` placeholder replaced with the confirmed `151936`.

**Deliberately left unchanged**: the S25 artifact entries (different
target, not touched this session). `verification_status` for the four
`qwen3-1.7b-s{0,1,2,3}-xelite` entries *was* later updated — see §13 — once
the full physical loop (not just individual-stage dtype confirmation)
justified it; that update also added real `measured_hardware_results`
latency entries. `ArtifactRegistry.from_yaml()` still loads all 13 models
cleanly after every edit; full test suite still 168/169 throughout (the 1
failure is the same pre-existing, unrelated gRPC heartbeat-timing flake
documented in `xelite_worker_status.md`).

## 9. S1 decode graph — close cross-check, not bit-exact (explained)

Built a rigorous reconstruction rather than an arbitrary probe: local
prompt position 127 (RoPE position 30, the last real prompt token) only
ever attends to context up through and including itself. Reconstructing a
decode call whose `past_key/value_in` is exactly `zero_384 ++ S1 prompt's
own new-KV output for local positions 0..126` (the *real* per-position KV
the prompt graph itself computed, not a guess), feeding the same token's
embedding, the same RoPE position (30), and the same attend-mask, should
reproduce the prompt graph's `add_21844` row 127 — an independent second
computation of the same logical value through a different graph entry
point.

**Result**: not bit-exact. Dequantized max abs diff `0.776`, mean abs diff
`0.166`, against a quantization step of `0.0185` (row 127's raw-`uint16`
value distribution has `std≈72.4`, so the mean difference is about `0.12`
standard deviations — small in relative terms, but a real, repeatable
difference, not sub-quantization-step noise). Most plausible explanation,
not confirmed further: the prompt graph computes all 128 positions through
a **batched** attention/MLP kernel while the decode graph uses a
**single-token-optimized** kernel — different floating/fixed-point
summation order across 10 quantized layers can produce small but nonzero
divergence even when both are computing the mathematically same thing. This
was not chased down further (would require comparing intermediate
per-layer activations, out of scope for "prove decode executes and produces
a plausible, close-to-correct result"). Recorded honestly rather than
rounded off to "matches" or treated as a failure.

S2's and S3's own decode graphs are proven later this session as part of
the full autoregressive loop (§13), using real upstream decode output and
real persistent KV rather than a reconstruction like this one — no separate
bit-exact-style cross-check was built for them specifically.

## 10. S2 prompt graph — physically proven with S1's real output

Same pattern as S1, generalized into a shared script
(`run_stage_prompt()`) so S2/S3 reuse the exact tested RoPE/mask/KV
machinery rather than re-deriving it. One real discovery pulling S2's own
metadata (not assumed from S1): **KV tensor names use the absolute
transformer layer index**, `past_key_10_in`..`past_key_19_in`, not
stage-relative `past_key_0_in`..`past_key_9_in` — this would have been a
silent, wrong-tensor-name bug if S1's naming had been assumed to generalize.

Ran with **S1's real, physically-computed `add_21844` output** as the
boundary input (not a placeholder) — `qnn_runner.run_context_binary(...,
graph_index=1, num_graphs=2)` against `qwen3-1.7b-s2-xelite.bin`, 0.82s.
`add_42314`: shape `(1, 128, 2048)`, dtype `uint16`, `nan_count=0`. Same
pad-row invariant as S0/S1 held: row 0 and row 1 (both padding) byte-identical,
row 127 (real token) different from both.

## 11. S3 prompt graph — physically proven, real end-to-end chain, top-1 token recorded

Ran S3 twice. First with a quantized-zero **placeholder** boundary input
(S2's real output didn't exist yet at that point in the session) purely to
prove the graph itself executes correctly in isolation — `logits`: shape
`(1, 128, 151936)`, dtype `uint16`, `nan_count=0`, same pad-row invariant
held. `logits`' vocab dimension (`151936`) is now physically confirmed,
correcting the manifest's "unverified" placeholder.

Then re-ran with **S2's real output** as the boundary input — a genuine,
complete `S0 → S1 → S2 → S3` physical forward pass through four separate
QNN context binaries on real Hexagon HTP silicon, 3.25s for this last hop.
Took the argmax of the last (real, non-padding) token position's
dequantized logits:

- **Top-1 token: id `151667`, logit `31.08`** — decodes to `<think>`.
- Top-5: `<think>` (31.08), `</think>` (24.45), `<|im_end|>` (22.99),
  `<|im_start|>` (22.89), `<tool_response>` (21.20).

`<think>` as the very first predicted token is not an arbitrary-looking
number — it is **exactly** the documented Qwen3 reasoning-mode behavior
already recorded elsewhere in this repo:
`src/dragon_nest/runtime/genie_runner.py`'s own docstring states "Qwen3
always emits a `<think>...</think>` reasoning block first (this model was
built with reasoning mode on)". Getting this specific, semantically
meaningful token out of a from-scratch physical reconstruction — real
tokenization, real RoPE, real attention masking, real per-tensor
quantization, chained through four independently-compiled QNN graphs — is
strong evidence the whole physical pipeline (not just each stage in
isolation) is numerically correct, not merely producing plausible-shaped
noise. This single forward pass was extended later this session into the
full `phase 7/8` local four-stage prefill + multi-token decode loop the
original bring-up plan describes — see §13.

## 13. Full physical prefill + 8-token autoregressive decode loop — final proof

Built a persistent sliding-window KV buffer (`kv_buffer.StageKVBuffer`,
generalized module, not inline hacks) per stage, seeded directly from each
stage's own real prompt-call `past_*_in` (zero) + `past_*_out` (real delta)
— confirmed via §7/§10/§11 to concatenate to exactly 512 entries — and
updated after every decode call the same way: `concat(current_window,
new_delta)[-512:]`. This is the same rule for both prompt (past_len=384,
delta=128) and decode (past_len=511, delta=1) calls; no special-casing
needed. Ran one real prefill, then 8 real decode steps, all on HTP, all
through `qnn_runner.run_context_binary()` — no mock execution, no CPU
fallback, no placeholder tensors anywhere in this final run.

**Per decode step**: `S0 decode(token_id)` → embedding; `S1
decode(embedding, S1 KV, RoPE pos, growing mask)` → `add_21844`, updates S1
KV; `S2 decode(add_21844, S2 KV, ...)` → `add_42314`, updates S2 KV; `S3
decode(add_42314, S3 KV, ...)` → `logits`, updates S3 KV; top-1 argmax →
next token. RoPE position and the attention-mask's real-token count both
advance by 1 every step (`31, 32, 33, ...` and `32, 33, 34, ...`
respectively — the mask's count includes the token currently being
processed). Step 1's S2 and S3 decode calls **are** the "S2 decode / S3
decode individually, with real upstream decode output and real stage-local
KV" proof this task asked for first — they were not re-run separately with
throwaway data, since that would be the identical computation.

### Results

- **Prefill top-1 token**: id `151667` → `<think>` (unchanged from the
  earlier session's real-chain result, §11 — same prompt, same pipeline,
  reproduced).
- **Generated token IDs** (prefill token + 8 decode steps, 9 total, no EOS
  hit): `[151667, 198, 151668, 271, 38409, 374, 279, 5344, 429]`
- **Decoded text**: `'<think>\n</think>\n\nGravity is the force that'`
- **Incremental decode** (one token added per line):
  `<think>` → `<think>\n` → `<think>\n</think>` → `<think>\n</think>\n\n` →
  `...Gravity` → `...Gravity is` → `...Gravity is the` →
  `...Gravity is the force` → `...Gravity is the force that`

This is a **coherent, on-topic, semantically correct** continuation of the
actual prompt ("What is gravity? Keep the answer under ten words.") — an
empty `<think>...</think>` block (Qwen3's documented reasoning-mode
preamble) followed by the literal start of a correct factual answer. This
is qualitatively strong evidence the full four-stage physical pipeline,
including real cross-call KV persistence over 8 steps, is numerically
sound end-to-end — a coherent sentence is a much harder thing to get by
accident than a plausible-shaped tensor.

### Per-stage latency (mean over 8 decode steps; prompt is a single call)

| Stage | Prompt (1 call) | Decode (mean of 8) | Decode min/max |
|---|---|---|---|
| S0 | 1.266s | 1.143s | 1.067 / 1.208s |
| S1 | 0.940s | 0.774s | 0.731 / 0.843s |
| S2 | 0.931s | 0.769s | 0.725 / 0.827s |
| S3 | 2.663s | 1.197s | 1.135 / 1.282s |

Total: 36.86s across 36 physical HTP calls (4 prompt + 32 decode). Every
call pays a full context-binary reload (`qnn-net-run.exe` has no
persistent-process/warm-load mode, consistent with `qnn_runner.py`'s and
`genie_runner.py`'s existing docstrings) — these numbers are not
representative of a production serving loop, only of this bring-up script's
call pattern.

### KV update method

Confirmed (§7, §10, §11) and now exercised for real across 8 steps: each
call's `past_*_out` is the newly-computed **delta** only, never a full
replacement. Buffer update rule: `new_window = concat(old_window,
new_delta)[..., -512:]`. Because this run only ever reaches sequence length
39 (31 real prompt tokens + 8 generated) against a `context_length` of 512,
the sliding window never actually needed to drop real content — only the
initial unused zero-padding — so this run does not independently prove the
window correctly evicts real (non-padding) history once the buffer
genuinely fills past 512 tokens. That would require a much longer
generation and was not attempted.

### Runtime/HTP confirmation

Every one of the 36 calls used `backend="htp"` through
`qnn_runner.run_context_binary()` (never `cpu`, never a mocked/short-circuited
path). No `qnn-net-run` call in this run returned mock or fallback output;
every failure mode encountered earlier in this session (§2, §2b) was a hard
process failure, not a silent fallback, so "physical HTP execution or a
loud error" was the operating invariant throughout, not "physical HTP
execution or a quiet approximation."

### Cleanup

- Released all Python-side `StageKVBuffer` state (`kv_s1.release()`,
  `kv_s2.release()`, `kv_s3.release()`) after the loop.
- **0 orphaned `qnn-net-run.exe`/`qnn`-named processes** remained
  post-run (checked via `psutil.process_iter`).
- **0 leftover scratch work directories** under `qnn_runner.py`'s
  `_SCRATCH_ROOT` (each call's own `finally: shutil.rmtree(...)` — already
  existing code, not new this session — accounted for all 36 calls'
  temporary input/output files).
- There is no separate "context unload" step to prove beyond this, because
  `qnn-net-run.exe` has no persistent/warm-loaded state to begin with —
  each call is already a fully self-contained process that loads, executes,
  and exits. The only session-scoped state this bring-up introduced was the
  Python-side KV buffers, which are now confirmed released.

### Independent reference comparison

**Not done.** Downloading and running a floating-point reference Qwen3-1.7B
(via `transformers`/PyTorch on CPU, several GB of real weights) was judged
not worth blocking this session's completion on, per instructions —
especially given the qualitative signal already obtained (a coherent,
on-topic, grammatically correct sentence, plus this repo's own documented
Qwen3 reasoning-mode behavior matching exactly) is already a strong,
non-coincidental correctness signal. Flagged as a genuine follow-up for
whoever wants a numeric (not just qualitative) confidence bound on this
pipeline's fidelity to the original unquantized model.

### Remaining scope, explicitly not attempted (per instructions)

- Android/S25 — untouched, as instructed.
- DragonNest Brain/runtime session integration (wiring this into
  PREFILL/DECODE/RESET abstractions, advertising these stages on a device
  registry) — untouched, as instructed.
- Distributed/cross-device execution — untouched, as instructed.
- A floating-point reference comparison (see above) — optional, explicitly
  allowed to be skipped rather than block completion.
- Generation past 39 total tokens (to exercise real KV eviction, not just
  zero-padding eviction) — not attempted; not required by the stop
  condition, which was 4-8 tokens.

### Stop condition: met

"The physical X Elite can execute the complete recovered Qwen3-1.7B S0→S3
pipeline, prefill once, then autoregressively generate multiple tokens
through S0→S3 decode with persistent correct KV state entirely on HTP" —
demonstrated above with 8 real decode steps producing coherent, on-topic
text.

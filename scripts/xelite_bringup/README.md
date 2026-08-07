# X Elite Qwen3-1.7B physical bring-up harness

Reproduces the physical prefill + multi-token autoregressive decode proof
recorded in `docs/results/qwen3_1_7b_xelite_physical_bringup.md`: one real
prompt pass through all four split-pipeline stages (S0 embedding, S1 layers
0-9, S2 layers 10-19, S3 layers 20-27 + LM head), followed by several real
decode steps with persistent per-stage KV state, entirely on Hexagon HTP.

This is a reproducibility harness, not production code -- no batching, no
warm/persistent context loading (each call pays a full `qnn-net-run.exe`
context-binary reload, consistent with `src/dragon_nest/runtime/qnn_runner.py`'s
existing architecture), and no DragonNest Brain/session integration.

## Prerequisites

- A Snapdragon X Elite (or same `windows-arm64-x1e-v73` class) Windows host.
- **QAIRT 2.45.x** (`aarch64-windows-msvc`) installed. Version matters: this
  pipeline's contexts were compiled against QAIRT ~2.45-era tooling; QAIRT
  2.32.6 rejects the context blob format outright, and QAIRT 2.48.40 loads
  it but crashes during graph finalization (see the bring-up doc, §2/§2b) --
  neither substitutes for the matching version.
- The `ADSP_LIBRARY_PATH` fix in `qnn_runner.py::_env()` (already in this
  repo) -- without it, `qnn-net-run.exe` crashes trying to execute unsigned
  HTP skel libraries instead of returning a clean error. See the bring-up
  doc §2c if retrofitting this onto an older checkout.
- The four X Elite Qwen3-1.7B split-pipeline QNN context binaries, staged
  and checksum-verified via the existing
  `scripts/artifact_tools/stage_xelite_artifacts.ps1` (see that script and
  `configs/model-artifacts.yaml` for provenance/checksums). This harness
  does **not** download or compile them -- they are external, licensed
  artifacts, intentionally not committed to this repository.
- Python deps beyond this repo's normal `requirements`: `torch`,
  `transformers` (with `jinja2` for chat templates), `psutil`. Network
  access to `huggingface.co` to fetch the public `Qwen/Qwen3-1.7B`
  tokenizer/config (no proprietary data; only the tokenizer and RoPE
  parameters are used from it, not model weights).

## Required environment variables

| Variable | Meaning |
|---|---|
| `QAIRT_ROOT` | Path to the QAIRT 2.45.x install root, e.g. `C:\Qualcomm\AIStack\QAIRT\2.45.41.260507` |
| `QWEN3_1_7B_S0_XELITE_QNN` | Path to the staged `qwen3-1.7b-s0-xelite.bin` context binary |
| `QWEN3_1_7B_S1_XELITE_QNN` | Path to the staged `qwen3-1.7b-s1-xelite.bin` context binary |
| `QWEN3_1_7B_S2_XELITE_QNN` | Path to the staged `qwen3-1.7b-s2-xelite.bin` context binary |
| `QWEN3_1_7B_S3_XELITE_QNN` | Path to the staged `qwen3-1.7b-s3-xelite.bin` context binary |

These are the same variable names `configs/model-artifacts.yaml` already
expects, so `scripts/artifact_tools/stage_xelite_artifacts.ps1`'s printed
`$env:...=` lines can be used directly.

## Optional environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DRAGONNEST_XELITE_BRINGUP_SCRATCH` | `%TEMP%\dragon_nest\xelite_bringup` | Where per-binary `qnn-context-binary-utility.exe` metadata dumps are cached between runs |
| `DRAGONNEST_XELITE_DECODE_STEPS` | `8` | Number of autoregressive decode steps to run after the prefill |

## Running it

```powershell
$env:QAIRT_ROOT = 'C:\Qualcomm\AIStack\QAIRT\2.45.41.260507'
$env:QWEN3_1_7B_S0_XELITE_QNN = '<path to staged qwen3-1.7b-s0-xelite.bin>'
$env:QWEN3_1_7B_S1_XELITE_QNN = '<path to staged qwen3-1.7b-s1-xelite.bin>'
$env:QWEN3_1_7B_S2_XELITE_QNN = '<path to staged qwen3-1.7b-s2-xelite.bin>'
$env:QWEN3_1_7B_S3_XELITE_QNN = '<path to staged qwen3-1.7b-s3-xelite.bin>'
.\.venv\Scripts\python.exe scripts\xelite_bringup\run_physical_smoke_test.py
```

Expect ~35-45s total (36 physical HTP calls: 4 prompt + 4×8 decode, each
paying a full context-binary reload) and, for the fixed built-in prompt
("What is gravity? Keep the answer under ten words."), a prefill top-1 token
of `<think>` (id `151667`) followed by 8 more real decode tokens.
Byte-for-byte token IDs are not guaranteed to reproduce exactly across
different QAIRT builds/host silicon revisions, but a coherent,
`<think>...</think>`-prefixed, on-topic continuation is the expected shape
of the result.

## Files

- `kv_buffer.py` -- `StageKVBuffer`: the sliding-window per-stage KV cache.
  Each stage's compiled graph outputs only the newly-computed KV positions
  (a delta), never a full replacement cache -- confirmed physically, not
  assumed. See its docstring.
- `stage_runner.py` -- per-tensor quantization lookup, RoPE (via the real
  `transformers.LlamaRotaryEmbedding`, not a hand-derived implementation),
  causal attention-mask construction, and the `run_s0_prompt` /
  `run_s0_decode` / `run_stage_prompt` / `run_stage_decode` functions that
  drive one physical `qnn-net-run.exe` call each.
- `run_physical_smoke_test.py` -- entry point: tokenizes the fixed prompt,
  runs the four-stage prefill, seeds the KV buffers from real prefill
  state, then runs the decode loop and prints token IDs, decoded text, and
  latency.

## What this harness does *not* do

- Does not integrate with DragonNest's Brain/Agent/session
  (PREFILL/DECODE/RESET) abstractions -- it is a standalone physical proof.
- Does not touch Android/S25.
- Does not include an independent floating-point reference comparison.
- Does not exercise real KV-window eviction (the fixed prompt + default 8
  decode steps stay well under the 512-token context window, so the
  sliding buffer only ever drops initial zero-padding, never real history).

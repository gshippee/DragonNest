# On-device NPU inference: EasyOCR + MeloTTS + Whisper

Hand-built local pipelines that run Qualcomm AI Hub's `easyocr`, `melotts_en`,
and `whisper_tiny` models fully offline on the Snapdragon X Elite's Hexagon
NPU (HTP), using the QAIRT SDK's `qnn-net-run.exe` directly against the
downloaded `.dlc`/`.bin` model artifacts. No cloud/AI Hub Workbench dependency.

> **Offline behavior:** once the one-time downloads below have happened, all
> three pipelines run with **no network access at all** — every inference
> call is local (NPU or CPU on this machine). MeloTTS and Whisper pull small
> companion assets from Hugging Face Hub (`bert-base-uncased` tokenizer/model
> for MeloTTS's BERT features; `openai/whisper-tiny`'s config/tokenizer/
> feature-extractor files, plus its full PyTorch weights now that the decoder
> runs on CPU — see the Whisper note below) the **first time** each pipeline
> runs, then caches them under `C:\Users\harisury\.cache\huggingface\hub\`
> (confirmed already present: `models--bert-base-uncased`,
> `models--myshell-ai--MeloTTS-English`, `models--openai--whisper-tiny`).
> With that cache already populated — which it is on this machine — you can
> disconnect entirely and every pipeline keeps working. EasyOCR has no such
> dependency; it's pure `.dlc` files plus the local `easyocr` package.

## Why hand-built pipelines, and why one per model?

**`qai_hub_models`' own demo/export scripts can't run these locally.** Every
demo (`easyocr/demo.py`, `melotts_en/demo.py`, `whisper_tiny/demo.py`) funnels
through `qai_hub_models/utils/args.py::demo_model_components_from_cli_args`,
which is a hard binary choice:
- default (`eval_mode=FP`): run the **original PyTorch model on CPU** — never
  touches the downloaded `.dlc`/`.bin` NPU artifacts at all, so it doesn't
  prove (or use) local NPU execution;
- `eval_mode=ON_DEVICE`: call `compile_model_from_args` →
  `qai_hub_models/utils/export/pipeline.py`, which requires AI Hub cloud
  credentials (raises `RuntimeError` without them) and drives everything —
  compile, quantize, profile, **and inference** — via
  `qai_hub.submit_inference_job(...)` against Qualcomm's cloud device farm.
  Even with a paid AI Hub Workbench account, the model still runs on
  Qualcomm's remote devices, not this laptop.

There is no third option, no `--local` flag, and no code path anywhere in
`qai_hub_models` that shells out to `qnn-net-run`/`qnn-context-binary-utility`
or any other local QAIRT runtime binary. `export.py`'s `--target-runtime` and
`--device` flags only choose what to compile *for*, not where to run it. The
downloaded `.dlc`/`.bin` files are meant to be picked up by AI Hub's cloud
pipeline or by a device-side app you build yourself — this repo *is* that
device-side app, written by hand against the raw QAIRT SDK tools instead.

**Why one pipeline script per model, rather than one generic runner:** each
model is architecturally distinct — different number of graphs, different
input/output tensor contracts, different CPU-side pre/post-processing — so
there's no shared orchestration logic beyond the low-level `qnn-net-run`
wrapper (`qnn_runner.py`, genuinely shared across all three):

| Model | Graphs | Why it needs bespoke glue |
|---|---|---|
| EasyOCR | 2 (`detector.dlc`, `recognizer.dlc`) | Box geometry between stages (`craft_utils.getDetBoxes`, `group_text_box`, `four_point_transform`) is bespoke CV code; recognizer runs a variable number of times (once per detected box). |
| MeloTTS | 3 (`encoder.bin`, `flow.bin`, `decoder.bin`) | `generate_path()` alignment math between encoder and flow; uint16 quantize/dequantize at each NPU boundary using per-tensor scale/zero_point from `metadata.json`; decoder runs in sliding overlapping chunks. |
| Whisper-tiny | 1 NPU graph (`encoder.bin`) + CPU decoder | Autoregressive KV-cache decode loop (self-attention cache updated every step, cross-attention cache computed once); HF tokenizer/feature-extractor glue; decoder runs on CPU, not NPU (see note below). |

Each pipeline reuses `qai_hub_models`' own `*App` classes (`EasyOCRApp`,
`MeloTTSApp`, `HfWhisperApp`) for this orchestration/pre/post-processing logic
unmodified — only the PyTorch `forward()` calls are swapped for NPU-backed
callables that shell out to `qnn_runner.py`. This means the *hard* part
(CV geometry, phoneme alignment, KV-cache bookkeeping, tokenization) is never
reimplemented; only a thin adapter layer per model is hand-written.

**Why not Genie (for these three models):** Qualcomm's Genie SDK (also shipped
in this QAIRT release, `docs\QAIRT-Docs\Genie\`) is explicitly scoped to
single-model LLM-style text generation — its own intro doc states "*support is
currently limited to large language models*." Its `genie-t2t-run` tool and
Dialog JSON config schema expect exactly one tokenizer + one decoder-only
transformer's context binaries + one sampler + a token-by-token generate loop
— there's no concept of chaining heterogeneous graphs or modalities. There's a
more experimental `GeniePipeline`/`GenieNode` API that can chain an "Encoder" +
"Generator", but its only documented examples (a vision-language model, a
translation encoder-decoder) are still single-modality-pair, text-in/text-out
cases with no tutorial or config for a 3-stage TTS pipeline, a
detector+recognizer OCR pair, or an ASR audio-encoder+autoregressive-decoder
pipeline like these. None of these three models fit Genie's model — they need
custom multi-graph orchestration with real CV/DSP/audio glue code in between,
which is exactly what `qai_hub_models`' `*App` classes already provide and
what this repo wires up to the NPU.

Genie *is* used elsewhere in this repo, though — see
[Voice Q&A pipeline](#voice-qa-pipeline) below, which is exactly the
single-model text-generation case Genie is built for (a live Q&A loop over a
transcribed question), chained after Whisper and before MeloTTS by a Python
orchestrator rather than by Genie itself.

## Prerequisites (already set up on this machine)

- QAIRT SDK extracted at `C:\Users\harisury\Downloads\v2.48.40.260702\qairt\2.48.40.260702`
- Python env `qai_env` (`C:\Users\harisury\qai_env`) with `qai_hub_models`, `easyocr`,
  `melo` (MeloTTS), `torch`, `numpy`, `soundfile`, `transformers`, `pip-system-certs` installed
- Model downloads:
  - `C:\Users\harisury\Downloads\easyocr-qnn_dlc-float\easyocr-qnn_dlc-float\` (`detector.dlc`, `recognizer.dlc`)
  - `C:\Users\harisury\Downloads\melotts_en-voice_ai-mixed_with_float-qualcomm_snapdragon_x_elite\...\` (`encoder.bin`, `flow.bin`, `decoder.bin`, `metadata.json`)
  - `C:\Users\harisury\Downloads\whisper_tiny-qnn_context_binary-float-qualcomm_snapdragon_x_elite\...\` (`encoder.bin`, `decoder.bin`, `metadata.json`)
  - `C:\Users\harisury\Downloads\qwen3_4b-genie-w4a16-qualcomm_snapdragon_x_elite\...\` (`genie-t2t-run.exe`, `genie_config.json`, tokenizer files, `part*_of_4.bin` context binaries) — used by the voice Q&A pipeline's Genie Q&A step, see below.
- One-time setup for fast EasyOCR HTP inference: run `python prepare_context_binaries.py` once to precompile `detector.dlc`/`recognizer.dlc` into `context_cache\detector_htp.bin`/`recognizer_htp.bin` — see [Context binaries for EasyOCR](#context-binaries-for-easyocr-why-prepare_context_binariespy-exists) below for why this is required, not optional, for HTP.

## Setting this up on a different device

Everything above (and the voice Q&A pipeline further down) was built and
tested on one specific machine, and several files hardcode that machine's
absolute paths and two specific Python virtual environments rather than
resolving anything relative to the repo folder or discovering paths at
runtime. Moving this folder to a different Windows machine means
re-satisfying all of the following, not just copying the `.py` files.

### Hardware requirement (non-negotiable)

A **Windows-on-ARM laptop with a Qualcomm Snapdragon X Elite (or another
Hexagon-NPU-equipped Snapdragon X-series) SoC**. The entire NPU/HTP execution
path (`qnn-net-run.exe` + `QnnHtp.dll`, and the ONNX Runtime QNN execution
provider) is Qualcomm-Hexagon-specific — none of it runs on Intel/AMD x86_64
hardware or non-Qualcomm ARM SoCs. `--backend cpu` still requires this same
hardware; it only swaps which QNN backend DLL runs the graph, not the
machine's architecture.

### Software that must be installed fresh

1. **QAIRT SDK** — download/extract Qualcomm's QAIRT SDK (this build used
   `v2.48.40.260702`; a newer version should work but hasn't been tested) and
   note the extracted path — every `*_pipeline.py`/`qnn_runner.py` needs to
   know where its `bin\aarch64-windows-msvc\qnn-net-run.exe` and
   `lib\aarch64-windows-msvc\Qnn{Htp,Cpu}.dll` live.
2. **Two Python virtual environments** — this repo's ONNX Runtime path
   (Whisper-base, `--runtime onnx` for EasyOCR) needs a *second*, native
   ARM64 Python because `qai_env` itself runs under x64 emulation and can't
   reach the HTP backend directly (see
   [The key constraint](#the-key-constraint-this-machines-arm64-python-is-not-actually-arm64)
   below for why):
   - **`qai_env`** (x64-emulated Python 3.x): `qai_hub_models`, `easyocr`,
     `melo` (MeloTTS), `torch`, `numpy`, `soundfile`, `sounddevice`,
     `opencv-python` (`cv2`), `Pillow`, `transformers`, `pip-system-certs`.
   - **`qai_env_arm64`** (genuinely native ARM64 Python — e.g. the official
     ARM64 Windows installer from python.org, *not* `qai_env`): `numpy`,
     `onnx`, `onnxruntime-qnn==2.4.0`. No `torch` here — no Windows ARM64
     PyTorch wheel exists, which is exactly why two environments are needed.
   - There's no `requirements.txt` in this bundle today; install the above
     via `pip install` into each venv, matching package names as closely as
     possible to what's listed (exact pinned versions weren't captured
     beyond `onnxruntime-qnn==2.4.0`, which matters — see the README section
     on ORT's provider API below).
3. **Model downloads** (not copied into this folder — re-download each from
   Qualcomm AI Hub / Hugging Face on the new device):
   - EasyOCR: `detector.dlc`, `recognizer.dlc` (`.dlc` runtime) — this
     bundle's `context_cache\*.bin` (copied alongside these scripts) are
     precompiled **for this specific machine's QAIRT build**; regenerate
     them on the new device with `prepare_context_binaries.py` rather than
     assuming the copied `.bin` files will load (context binaries are
     compiled artifacts, not portable across arbitrary QAIRT/device
     combinations).
   - MeloTTS: `encoder.bin`, `flow.bin`, `decoder.bin`, `metadata.json`
   - Whisper-base (ONNX path, used by `voice_qa_pipeline.py --whisper base`,
     the default): `encoder.onnx`/`decoder.onnx` + their sibling
     `*_qairt_context.bin` files
   - Whisper-tiny (`.dlc`/`.bin` path, only needed for `--whisper tiny`):
     `encoder.bin`, `decoder.bin`, `metadata.json`
   - Qwen3-4B Genie bundle: `genie-t2t-run.exe`, `genie_config.json`,
     tokenizer files, `part*_of_4.bin` context binaries — this is the
     largest download (~3.1GB of context binaries), required since Genie
     answers every question in the voice Q&A loop.
   - MeloTTS and Whisper also pull small companion assets from Hugging Face
     Hub on first run (`bert-base-uncased` for MeloTTS, the Whisper
     tokenizer/config) — the new device needs network access at least once
     for this, or its own pre-populated
     `%USERPROFILE%\.cache\huggingface\hub\`.

### Hardcoded paths that must be edited

Every one of these is a module-level constant, not a CLI flag or environment
variable — grep for `Path(r"C:\Users\harisury` across this folder to confirm
nothing was missed after editing:

| File | Constant | Points to |
|---|---|---|
| `qnn_runner.py` | `QAIRT_ROOT` | Extracted QAIRT SDK root |
| `genie_runner.py` | `GENIE_DIR` | Qwen3-4B Genie bundle folder |
| `chunking.py` | `QWEN_DIR` | Same Genie bundle folder (only used to load its tokenizer for token counting) |
| `easyocr_pipeline.py` | `EASYOCR_DIR`, `EASYOCR_ONNX_DIR` | EasyOCR `.dlc` folder and `.onnx`/w8a8 folder |
| `melotts_pipeline.py` | `MELOTTS_DIR` | MeloTTS `.bin` folder |
| `whisper_pipeline.py` | `WHISPER_DIR` | Whisper-tiny `.bin` folder (only used by `--whisper tiny`) |
| `whisper_onnx_pipeline.py` | `WHISPER_ONNX_DIR` | Whisper-base `.onnx` folder (the default Whisper path) |
| `prepare_context_binaries.py` | `EASYOCR_DIR` | Same EasyOCR `.dlc` folder as above |
| `onnxrt_runner.py` | `ARM64_PYTHON` | Full path to `qai_env_arm64`'s `python.exe` |

`qnn_runner.py`'s and `genie_runner.py`'s `_SCRATCH_ROOT` (temp-file
directory for prompts/tensors) already resolves relative to each script's
own location (`Path(__file__).parent / "scratch"`), so those don't need
editing — only the paths above, which point at external SDK/model
directories, do.

The `& "C:\Users\harisury\qai_env\Scripts\python.exe" ...` prefix on every
command in this README also needs updating to the new device's own venv
paths (both `qai_env` for everything below, and `qai_env_arm64` only if you
ever run `onnxrt_worker.py` directly rather than through
`onnxrt_runner.py`'s auto-launch).

### One-time setup step to re-run

After the QAIRT SDK, model downloads, and `EASYOCR_DIR` path are all in
place on the new device, run `prepare_context_binaries.py` once (see
[Context binaries for EasyOCR](#context-binaries-for-easyocr) below) —
**don't** rely on the `context_cache\*.bin` files copied alongside these
scripts; they're compiled for the original machine's QAIRT build/device and
may fail to load or silently mis-execute on different hardware.

### Known device-specific quirk worth re-checking

`record_audio.py` monkeypatches `platform.machine()` to `"AMD64"` before
importing `sounddevice`, because on the original machine `qai_env`'s Python
process is actually x64 (running under Windows-on-ARM emulation) despite
`platform.machine()` reporting `"ARM64"`. If the new device's `qai_env`
Python is genuinely native ARM64 instead (unlikely, since `torch` has no
ARM64 Windows wheel — see
[The key constraint](#the-key-constraint-this-machines-arm64-python-is-not-actually-arm64)
below — but worth stating explicitly), this monkeypatch would need removing
or `sounddevice` would try to load the wrong bundled PortAudio DLL.


## How to run

Both scripts live in `C:\Users\harisury\qnn_pipeline\` and share `qnn_runner.py`,
the wrapper around `qnn-net-run.exe`.

### EasyOCR (image -> detected text boxes)

```powershell
& "C:\Users\harisury\qai_env\Scripts\python.exe" `
  "C:\Users\harisury\qnn_pipeline\easyocr_pipeline.py" `
  "<path-to-image.png>" `
  --backend htp `
  --output_image "<optional-annotated-output.png>"
```

- `--backend htp` runs on the NPU (default); `--backend cpu` runs the same
  compiled graphs on CPU for a sanity check.
- `--runtime dlc` (default, see [Context binaries for EasyOCR](#context-binaries-for-easyocr) —
  requires the one-time `prepare_context_binaries.py` setup for fast HTP
  inference) or `--runtime onnx` (see [ONNX Runtime execution path](#onnx-runtime-execution-path-easyocr-w8a8-whisper-base) below).
- Prints one line per detected text region: `<confidence>\t<box>\t<recognized text>`.
- If `--output_image` is given, saves a copy of the input with detection boxes drawn.

### MeloTTS (text -> spoken WAV)

```powershell
& "C:\Users\harisury\qai_env\Scripts\python.exe" `
  "C:\Users\harisury\qnn_pipeline\melotts_pipeline.py" `
  "Text to speak out loud." `
  "<path-to-output.wav>" `
  --backend htp
```

- Produces a 44.1kHz mono WAV file with the "EN-US" MeloTTS voice.
- `--backend cpu` will NOT work for MeloTTS — see note below.

> **Note:** unlike EasyOCR's `.dlc` files, MeloTTS's `.bin` files are QNN
> **context binaries** pre-compiled for a specific backend at export time. They
> can only be deserialized by the HTP backend they were compiled for —
> `--backend cpu` fails with "Context de-serialization failed". Always use
> `--backend htp` for MeloTTS.

### Whisper-tiny (speech -> transcribed text)

```powershell
& "C:\Users\harisury\qai_env\Scripts\python.exe" `
  "C:\Users\harisury\qnn_pipeline\whisper_pipeline.py" `
  "<path-to-audio.wav>" `
  --backend htp
```

- Prints `Transcription: <text>`.
- Runs the **encoder** (`encoder.bin`, ~20MB) on the NPU; the **decoder** runs
  as plain PyTorch (`openai/whisper-tiny` weights, downloaded/cached from
  Hugging Face on first run) on CPU — see note below for why.
- `--backend cpu` runs the encoder's compiled graph on the CPU backend instead
  of HTP as a sanity check; the decoder is unaffected by this flag either way.

> **Note:** this model ships a `decoder.bin` (~97MB context binary) that
> **cannot be loaded on this device's HTP backend at all**. Every attempt —
> through the full pipeline, via a direct isolated `qnn-net-run.exe` call, and
> with/without `--use_mmap` — fails identically:
> `Skel failed to process context binary` / `err 5005` on every protection
> domain (PD 0, 1, 2, 3), then `Failed to find available PD ... with context
> size estimate 120426752`. `qnn-context-binary-utility.exe` confirms the file
> itself is well-formed (not corrupted), so this is a runtime resource
> allocation failure, not a bad download. QAIRT's HTP backend docs describe a
> per-PD memory ceiling but expose no CLI/JSON/registry knob to raise it (only
> VTCM size and I/O-estimation options are configurable). `encoder.bin` (~20MB)
> and every MeloTTS binary (all <30MB) load fine on the same device/backend, so
> this looks like a hard per-PD capacity limit on this specific device that
> `decoder.bin`'s size exceeds — not a bug in `qnn_runner.py` or the pipeline.
> Since the decoder is small per-step compute (4 transformer blocks,
> `d_model=384`, one token at a time), running it on CPU has negligible cost
> compared to the NPU-accelerated encoder pass.

## Repository layout

| File | Purpose |
|---|---|
| `qnn_runner.py` | Thin wrapper: writes raw tensors + `input_list.txt`, shells out to `qnn-net-run.exe`, reads back raw output tensors as NumPy arrays. Supports both raw `--dlc_path` graphs and precompiled `--retrieve_context` binaries, plus batched multi-input calls (`run_dlc_batch`/`run_context_binary_batch`) and automatic retry on transient failures. |
| `easyocr_pipeline.py` | Builds `qai_hub_models`' `EasyOCRApp` with two NPU-backed callables (`detector.dlc`, `recognizer.dlc`) standing in for the PyTorch modules; all box detection/grouping/decoding logic reused unmodified. Recognizer runs as one batched `qnn-net-run.exe` call over all detected boxes rather than one process per box. Uses precompiled context binaries (`context_cache\*.bin`) automatically when present — see note below. |
| `melotts_pipeline.py` | Builds `qai_hub_models`' `MeloTTSApp` with three NPU-backed callables (`encoder.bin`, `flow.bin`, `decoder.bin`); text preprocessing, alignment, and chunked-decoder orchestration reused unmodified. |
| `whisper_pipeline.py` | Builds `qai_hub_models`' `HfWhisperApp` with an NPU-backed encoder callable (`encoder.bin`) and a plain PyTorch decoder (`HfWhisperDecoder.from_pretrained`, CPU) — see note above; feature extraction, autoregressive KV-cache decode loop, and tokenization reused unmodified. |
| `prepare_context_binaries.py` | One-time setup script: precompiles `detector.dlc`/`recognizer.dlc` into HTP context binaries via `qnn-context-binary-generator.exe`, so `easyocr_pipeline.py` never has to pay `qnn-net-run.exe`'s full graph-recompile cost per call. See note below. |
| `genie_runner.py` | Thin subprocess wrapper around `genie-t2t-run.exe` (Qwen3-4B, Qualcomm's Genie SDK) — used by the voice Q&A pipeline for its Q&A step. See below. |
| `chunking.py` | Shared size-measurement/splitting helpers (Genie token-budget chunking, TTS char-budget chunking) used by the voice Q&A pipeline. See below. |
| `doctor_note_pipeline.py` | Not run directly here — kept only because `voice_qa_pipeline.py` imports its `_synthesize_chunk_with_retry` TTS helper. |
| `record_audio.py` | Push-to-talk microphone recorder used by the voice Q&A pipeline. See below. |
| `voice_qa_pipeline.py` | End-to-end orchestrator: mic recording -> Whisper transcription -> Genie Q&A (with a clarification loop) -> MeloTTS -> auto-play. See below. |
| `patient_records.py` | Patient visit-history lookup used by the voice Q&A pipeline's optional `--records-dir` fallback. See below. |

---

## End-to-end flow: EasyOCR (image in -> text out)

```
image file
   │  PIL.Image.open().convert("RGB")
   ▼
[CPU] resize_pad to 608x800, normalize to [0,1], NCHW->NHWC permute
   ▼
[NPU/HTP]  detector.dlc   (qnn-net-run --dlc_path)
   │  in:  image  [1,608,800,3] float32
   │  out: results [1,304,400,2] float32  (per-pixel text/link scores, half-res)
   ▼
[CPU] craft_utils.getDetBoxes() -> denormalize -> group_text_box()
      -> per-box crop (four_point_transform) from the original grayscale image
   ▼
[NPU/HTP]  recognizer.dlc   (qnn-net-run --dlc_path, once per detected box)
   │  in:  image [1,64,800,1] float32 (cropped/padded grayscale line)
   │  out: output_preds [1,199,97] float32  (per-char-position class logits)
   ▼
[CPU] softmax -> mask ignored chars -> greedy argmax decode -> strip artifacts
   ▼
recognized text + confidence, per box
```

**Hardware split:** every pixel-level convolutional computation (detection
CNN, recognition CNN) runs on the Hexagon Tensor Processor (HTP/NPU). All the
"glue" logic — image resize/pad, box geometry (NMS-like grouping, rotation
correction), and greedy CTC-style decoding — is plain NumPy/PyTorch on the
CPU, because it's cheap, branchy, and not worth compiling to a fixed NPU graph.

### Context binaries for EasyOCR

`qnn-net-run.exe --dlc_path` recompiles a `.dlc`'s **entire graph** (graph
optimization, VTCM allocation, graph sequencing, parallelization) from
scratch on every single process invocation — measured at **~185-200s for
`recognizer.dlc` alone**. Early testing misread this as the recognizer
"hanging" against a 120s timeout, and separately misread a red-herring
`DspTransport.openSession qnn_open failed, 0x80000406` stderr message (logged
around the same time) as the cause — that error is harmless and
auto-recovered (falls back to the "user driver path"); verbose logging
confirms the run proceeds normally into graph compilation right after it.
**The real cost is the recompile itself, not a hang or a DSP error.**

The fix: run `python prepare_context_binaries.py` once (or whenever
`detector.dlc`/`recognizer.dlc` change) to precompile both graphs into HTP
context binaries via `qnn-context-binary-generator.exe`, written to
`context_cache\detector_htp.bin` / `context_cache\recognizer_htp.bin`:

```powershell
& "C:\Users\harisury\qai_env\Scripts\python.exe" `
  "C:\Users\harisury\qnn_pipeline\prepare_context_binaries.py"
```

This pays the ~185-200s compile cost once, up front, instead of on every
call. `easyocr_pipeline.py --runtime dlc --backend htp` then automatically
picks up `context_cache\*.bin` (via `ContextBinaryModule`, loaded with
`qnn-net-run.exe --retrieve_context`, the same fast path already used by
MeloTTS) if both files exist, falling back to the slow raw-`.dlc` path
(`DlcModule`) only if they don't — reducing recognizer inference from
~185-200s/call to ~1-2s/call. `--backend cpu` always uses the raw-`.dlc`
path, since HTP context binaries can't be loaded by `QnnCpu.dll`.

If you ever see the recognizer appear to hang again after regenerating or
replacing the `.dlc` files, this is the first thing to check: delete/rerun
`prepare_context_binaries.py` rather than assuming a DSP session problem.

---

## End-to-end flow: MeloTTS (text in -> speech out)

```
input text string
   │
   ▼
[CPU] melo package: text normalization, g2p (grapheme->phoneme), tone
      assignment, BERT tokenization + inference (bert-base-uncased, plain
      PyTorch) -> phones, tones, language ids, bert features, ja_bert features
   ▼
[NPU/HTP]  encoder.bin   (qnn-net-run --retrieve_context)
   │  in:  x, x_lengths, tone, sid, language, bert, ja_bert,
   │       sdp_ratio, length_scale, noise_scale_w   (int32/float32, unquantized)
   │  out: m_p, logs_p, w_ceil, y_lengths, x_mask, g
   │       (phoneme-level latent stats + per-phoneme predicted duration
   │        + speaker embedding g)
   ▼
[CPU] generate_path(): turns w_ceil (durations) + masks into an explicit
      phoneme->frame alignment/attention matrix (pure indexing math, no NN)
   ▼
[CPU] quantize m_p/logs_p/y_mask/g/attn_squeezed/noise_scale to uint16
      (scale/zero_point read from metadata.json)
   ▼
[NPU/HTP]  flow.bin   (qnn-net-run --retrieve_context, uint16 quantized I/O)
   │  in:  m_p, logs_p, y_mask, g, attn_squeezed, noise_scale
   │  out: z   (frame-level latent "acoustic code", upsampled 3x from phoneme rate)
   ▼
[CPU] dequantize z to float32, slice into overlapping ~40-frame chunks
      (40 frames + 12-frame overlap each side), re-quantize each chunk to uint16
   ▼
[NPU/HTP]  decoder.bin   (qnn-net-run --retrieve_context, uint16 quantized I/O,
   │        called once per chunk — this is the vocoder / waveform synthesizer)
   │  in:  z (chunk), g
   │  out: audio (chunk)
   ▼
[CPU] dequantize each audio chunk, crop overlap regions, concatenate chunks,
      trim to y_lengths * upsample_factor, write WAV at 44100 Hz
   ▼
output .wav file
```

**Hardware split:** three separate NPU graphs run in sequence per utterance —
`encoder` (text->latent alignment stats, run once), `flow` (latent
upsampling/normalizing-flow, run once), `decoder` (neural vocoder, run
repeatedly over sliding chunks to bound the NPU graph's fixed input size).
Everything upstream of the encoder (tokenization, G2P, BERT) and the
generate_path/quantization/chunk-stitching glue between NPU stages runs on
CPU. `flow.bin`/`decoder.bin` are int-quantized (uint16) at the NPU boundary —
the CPU-side code quantizes right before each NPU call and dequantizes right
after, using the exact scale/zero_point constants QAIRT recorded when it
compiled these graphs (in `metadata.json`).

### Why the decoder runs in chunks

The decoder graph's compiled input shape is fixed at `z: [1, 192, 64]` frames
— it can't accept an arbitrarily long utterance in one call. `melotts_pipeline.py`
(inherited from `MeloTTSApp.tts_to_file()`) slides a 40-frame window with
12-frame overlap on each side across the full `z` sequence, decodes each
window separately on the NPU, and stitches the results by cropping away the
overlap regions before concatenation — this avoids audible clicks/discontinuities
at chunk boundaries.

## In-depth analysis: profiling and layer inspection

`qnn_runner.py` exposes two opt-in Python APIs, both built on top of QAIRT
SDK tools already shipped in `qairt\...\bin\aarch64-windows-msvc\`. Both are
off by default (zero overhead on normal runs) and are toggled globally, so
they apply to every `qnn-net-run.exe` call made while enabled — including
every stage of a full pipeline run (e.g. both `detector.dlc` and
`recognizer.dlc` for EasyOCR, once per detected box).

### KPI profiling (`enable_profiling`)

```python
import qnn_runner
qnn_runner.enable_profiling(r"C:\Users\harisury\qnn_pipeline\profile_out")

import easyocr_pipeline
easyocr_pipeline.run("test_image.png", backend="htp")

qnn_runner.disable_profiling()
```

Every `qnn-net-run` call made while enabled gets `--profiling_level detailed`
added, and writes three files to a numbered subfolder of `profile_out` (one
subfolder per call, e.g. `0000_detector/`, `0001_recognizer/`,
`0002_recognizer/`, ... — recognizer runs once per detected text box):

| File | Contents |
|---|---|
| `qnn-profiling-data_0.log` | Raw binary QNN profiling log (the artifact `--profiling_level` produces). |
| `profile.txt` | Human-readable rendering via `qnn-profile-viewer.exe --input_log <log>`. |
| `profile.csv` | Same data as CSV (`Msg Timestamp, Message, Time, Unit of Measurement, Timing Source, Event Level, Event Identifier`) — pull into a spreadsheet/pandas for aggregation across many calls. |

`profile.txt` is the most useful starting point. Key numbers to look at:

- **`NetRun IPS (includes IO and misc. time)`** — end-to-end inferences/sec for that single call, the top-line throughput KPI.
- **`Backend (Accelerator (execute) time (cycles))`** — total HTP cycles for that call's graph, broken down **per op** by name (e.g. `/model/basenet/slice1/0/Conv:OpId_27 (cycles): 0 cycles`) — this is how you find which layer is the bottleneck.
- **`Backend (Number of HVX threads used)`** — parallelism the backend actually used.
- **`Backend (Time for HVX + HMX power on and acquire)`** — fixed per-call overhead of waking the accelerator; large relative to a short graph's execute time if you're calling very small graphs very frequently (relevant for MeloTTS's chunked `decoder.bin`, called once per audio chunk).
- **Init/Finalize Stats** — one-time graph-load cost (only paid once per `qnn-net-run.exe` process; since each `qnn_runner._run()` call spawns a fresh process, this cost is paid on every call today — see note below).

You can also profile the raw command directly, without Python, for a single ad hoc check:

```powershell
& "...\bin\aarch64-windows-msvc\qnn-net-run.exe" --dlc_path detector.dlc `
  --input_list input_list.txt --backend "...\lib\aarch64-windows-msvc\QnnHtp.dll" `
  --output_dir out --use_native_input_files --use_native_output_files `
  --profiling_level detailed

& "...\bin\aarch64-windows-msvc\qnn-profile-viewer.exe" --input_log out\qnn-profiling-data_0.log
```

**Aggregate throughput (IPS) across repeated calls:** to measure steady-state
throughput rather than a single call's IPS (which includes one-time
init/finalize overhead), loop the same input multiple times in one process
via `--num_inferences` and pin a performance mode with `--perf_profile`:

```powershell
& "...\qnn-net-run.exe" --dlc_path detector.dlc --input_list input_list.txt `
  --backend "...\QnnHtp.dll" --output_dir out --use_native_input_files `
  --use_native_output_files --num_inferences 10 --perf_profile burst
```

This produces `Result_0` through `Result_9` and one aggregate `NetRun IPS`
line reflecting steady-state throughput (confirmed: `7.0011 inf/sec` for
`detector.dlc` on HTP over 10 back-to-back inferences). `qnn_runner.py`
doesn't wrap this today — each `_run()` call is a single inference in its own
process — so use the raw command above if you need a steady-state number.

### Per-layer inspection (`enable_layer_dump`)

```python
import qnn_runner
qnn_runner.enable_layer_dump(r"C:\Users\harisury\qnn_pipeline\debug_out")

import easyocr_pipeline
easyocr_pipeline.run("test_image.png", backend="htp")

qnn_runner.disable_layer_dump()
```

Adds `--debug` to every `qnn-net-run` call made while enabled, and copies
every intermediate tensor's raw output (one `.raw` file per op in the graph,
named after the ONNX/graph op — e.g. `_model_basenet_slice1_0_Conv_output_0.raw`)
into a numbered subfolder of `debug_out`, mirroring the same
`0000_<graph>/`, `0001_<graph>/`, ... numbering as profiling. Load a `.raw`
file with `numpy.fromfile(path, dtype=np.float32)` and reshape using the
shape recorded for that tensor (see `qnn-context-binary-utility.exe` below,
or just infer from context — most intermediate CNN feature maps are NHWC).

This is useful for "did layer N actually do something sane" debugging — e.g.
this is exactly how the `x_lengths`-corruption bug (see below) was originally
isolated, by comparing an intermediate tensor's dumped values against what
was expected.

**Important limitation:** `--debug` only works for graphs loaded via
`--dlc_path` (EasyOCR's `detector.dlc`/`recognizer.dlc`). MeloTTS's
`encoder.bin`/`flow.bin`/`decoder.bin` are QNN **context binaries** loaded via
`--retrieve_context` — they're already fully compiled/finalized, so
intermediate tensors inside them aren't observable at all; QNN has no
equivalent of `--debug` for context binaries. `qnn_runner.enable_layer_dump()`
knows this and silently skips adding `--debug` for `run_context_binary()`
calls (MeloTTS), so enabling it while running the MeloTTS pipeline is a no-op
rather than an error. If you need to inspect a MeloTTS-internal tensor, the
only lever is `--set_output_tensors=<graphName:tensorName,...>` at **export
time** (before compiling to a context binary) — not something achievable
against the already-downloaded `.bin` files.

For a lighter-weight version that dumps only specific tensors instead of
every layer (also `--dlc_path`-only), use `--set_output_tensors` directly:

```powershell
& "...\qnn-net-run.exe" --dlc_path detector.dlc --input_list input_list.txt `
  --backend "...\QnnHtp.dll" --output_dir out --use_native_input_files `
  --use_native_output_files `
  --set_output_tensors="graph_name:tensor_a,tensor_b"
```

### Inspecting a compiled model's I/O contract

To see exactly what tensors a graph expects/produces (names, shapes, dtypes,
quantization params) without running it — useful for MeloTTS's context
binaries, where you can't get this from `--debug`:

```powershell
& "...\qnn-context-binary-utility.exe" --context_binary encoder.bin --json_file encoder_info.json
```

This is also how MeloTTS's exact graph names and tensor names (documented in
this repo's code comments) were originally confirmed against `metadata.json`.

### A note on the noisy `DspTransport` stderr message

HTP runs sometimes print `DspTransport.openSession qnn_open failed,
0x80000406` (or `IDspTransport: Unable to load lib 0x80000406`) to stderr.
Observed consistently to be non-fatal — every run that printed this still
reached "Finished Executing Graphs" and produced correct output. Safe to
ignore; not something `qnn_runner.py` currently filters out.

## A note on precision: two bugs found and fixed in `qnn_runner.py`

1. **Native vs. float I/O parsing.** `qnn-net-run` defaults to reading/writing
   raw I/O files as floating point regardless of a tensor's real dtype. This
   is harmless for EasyOCR (its DLCs are float32 end-to-end) but silently
   corrupted MeloTTS's `int32`/`uint16` tensors (e.g. a phoneme count of 45
   reinterpreted as a raw float32 bit pattern is a near-zero denormal, so the
   encoder always saw `x_lengths=0`). Fixed by always passing
   `--use_native_input_files --use_native_output_files`.
2. **Native output filenames.** With native output mode, QNN writes
   `<tensor>_native.raw` instead of `<tensor>.raw`. `qnn_runner.py` now checks
   for the native filename first, falling back to the plain name.

## Where inference actually runs, and how it gets there

Each pipeline call ends up executing on one of three physically different
places, depending on the call: the **CPU** (this laptop's Snapdragon X Elite
cores, running plain Python/NumPy/PyTorch), the **Hexagon NPU/HTP**
(`qnn-net-run.exe` with `QnnHtp.dll`), or — for the CPU sanity-check backend
only — the same compiled graph re-run through `QnnCpu.dll` on the CPU. No call
in any of these three pipelines ever leaves this machine; there is no network
request, cloud job, or remote device involved at any point.

### The local NPU call path, end to end

For every `qnn_runner.run_dlc()` / `run_context_binary()` call:

1. **Python (`qnn_runner.py`, CPU, host process)** — writes each input tensor
   to a raw binary file, writes `input_list.txt` pointing at them, and spawns
   `qnn-net-run.exe` as a subprocess (one fresh process per call today).
2. **`qnn-net-run.exe` (CPU, host process)** — a QAIRT SDK binary built for
   `aarch64-windows-msvc`, i.e. it runs natively on the Snapdragon X Elite's
   ARM cores (not under x86 emulation). It loads the QNN backend requested via
   `--backend`:
   - `QnnHtp.dll` — the HTP (Hexagon Tensor Processor) backend, i.e. the NPU path.
   - `QnnCpu.dll` — a reference CPU backend that executes the same graph on
     the host CPU cores instead, used here only as a plumbing sanity check
     (see per-model notes above — EasyOCR supports this, MeloTTS/Whisper's
     compiled context binaries generally don't).
3. **`QnnHtp.dll` (CPU, host process)** — the host-side half of the HTP
   backend. It deserializes the `.dlc`/`.bin` graph, and hands the compiled
   graph + input tensors to the DSP side over **FastRPC**, Qualcomm's
   host↔DSP RPC transport (this is the same transport whose harmless
   `DspTransport.openSession qnn_open failed, 0x80000406` warning shows up in
   every run — see note above).
4. **Hexagon cDSP (the NPU silicon itself)** — a physically separate
   processor on the same SoC die, not a CPU core. FastRPC delivers the graph
   into a **Process Domain (PD)** on the DSP — an isolated DSP-side execution
   context with its own address space (per QAIRT's HTP backend docs, each PD
   supports up to 3.75GB of usable virtual address space on this
   architecture). Inside that PD, `libQnnHtpV73Skel.so` (the "skel" library,
   present under
   `qairt\...\lib\hexagon-v73\unsigned\libQnnHtpV73Skel.so` — "v73" matches
   this device's Hexagon core version) is the actual code that runs the
   compiled graph's ops using the DSP's HVX (Hexagon Vector eXtensions,
   wide SIMD) and HMX (Hexagon Matrix eXtensions, systolic matmul) execution
   units — this is the hardware that actually does the NPU's matrix
   multiplies and convolutions.
5. **Result path back** — outputs come back over FastRPC to `QnnHtp.dll`,
   which `qnn-net-run.exe` writes to raw files in `--output_dir`;
   `qnn_runner.py` reads those files back into NumPy arrays and returns them
   to the calling pipeline, which continues on the CPU (box decoding, KV-cache
   bookkeeping, quantization, etc.) until the next NPU call.

Every one of the `[NPU/HTP]` boxes in the end-to-end flow diagrams above (and
the `EncoderModule`/`DecoderModule`/`RecognizerModule`-style classes in each
`*_pipeline.py`) is exactly this 5-step round trip. This is also why the
per-PD memory ceiling documented in the Whisper note above is a **DSP-side**
limit, not a host RAM limit — `decoder.bin`'s ~120MB context has no trouble
existing in this laptop's system RAM (which has far more than 120MB free);
the failure is specifically the Hexagon cDSP's PD being unable to host a
context that large, confirmed by the error recurring identically across all
4 PDs the backend tried.

### What's genuinely local vs. what's "local but not on the NPU"

| Stage | Where it runs | Why |
|---|---|---|
| Image/audio/text preprocessing, box geometry, phoneme alignment, quantize/dequantize, tokenization | CPU (this laptop, plain Python) | Cheap, branchy, or one-off work not worth compiling to a fixed NPU graph — see per-model hardware-split notes above. |
| EasyOCR `detector.dlc` / `recognizer.dlc` | Hexagon NPU (HTP) | Convolutional graphs, exactly what the NPU is compiled/optimized for. |
| MeloTTS `encoder.bin` / `flow.bin` / `decoder.bin` | Hexagon NPU (HTP) | All three stages are compiled QNN context binaries targeting HTP only. |
| Whisper `encoder.bin` | Hexagon NPU (HTP) | Compiled QNN context binary targeting HTP, loads and runs fine. |
| Whisper decoder | **CPU**, not NPU | `decoder.bin` exists and is a valid HTP-targeted context binary, but cannot be loaded into any HTP Process Domain on this device (see note above) — so `whisper_pipeline.py` falls back to running the original `openai/whisper-tiny` PyTorch decoder weights on the CPU instead. This is the one stage, across all three pipelines, that is local but **not** NPU-accelerated. |

So: "fully offline" (no cloud/network dependency, everything in this repo
runs on this physical machine) is true for all three pipelines without
exception. "Fully NPU-accelerated" is true for EasyOCR and MeloTTS, but not
for Whisper — its decoder runs on the CPU due to the device-specific HTP
memory-ceiling limitation described above, while its encoder still runs on
the NPU.

---

## ONNX Runtime execution path (EasyOCR w8a8, Whisper-base)

Two more model downloads — under `C:\Users\harisury\Downloads\ONNX\` — use a
different deployment format from everything above: genuine ONNX graphs
(`.onnx`), meant to run through **ONNX Runtime's own QNN execution provider**
(`onnxruntime-qnn`), not through `qnn-net-run.exe`. This is an additive path;
none of the `.dlc`/`.bin` pipelines above changed.

| Download | Files | Format |
|---|---|---|
| `easyocr-onnx-w8a8\easyocr-onnx-w8a8\` | `detector.onnx`, `recognizer.onnx` | Real QDQ (QuantizeLinear/DequantizeLinear) op graphs, uint8 w8a8-quantized I/O. Same underlying EasyOCR network as the `.dlc` files, re-exported. |
| `whisper_base-precompiled_qnn_onnx-float-qualcomm_snapdragon_x_elite\...\` | `encoder.onnx`, `decoder.onnx` | Each is a one-node **EPContext** wrapper (`source="QNN"`) around a precompiled QNN context binary (`encoder_qairt_context.bin` 47MB, `decoder_qairt_context.bin` 145MB, sitting alongside the `.onnx` files). This is **Whisper-base** (6 decoder blocks), not the whisper-tiny (4 blocks) used above — a different, larger model. |

### The key constraint: this machine's "ARM64" Python is not actually ARM64

`qai_env` (the environment every pipeline above runs in) reports
`platform.machine() == 'ARM64'`, but its Python process is genuinely **x64**,
running under Windows-on-ARM's x86_64 emulation layer — confirmed via its PE
header (`0x8664`, AMD64) and `PROCESSOR_ARCHITECTURE=AMD64`. `platform.machine()`
reflects the OS, not the process. This matters a lot here: **no x64 build of
the Hexagon HTP stub libraries exists anywhere in the QAIRT SDK** — they only
ship for `aarch64-windows-msvc`/`arm64x-windows-msvc` — so an x64 process can
never reach the NPU via QNN's HTP backend, full stop, regardless of which
`onnxruntime-qnn` build is installed. (`qnn-net-run.exe` sidesteps this
entirely for the `.dlc`/`.bin` pipelines above because it's a genuinely
`aarch64`-native binary, invoked as a subprocess — it was never running
inside `qai_env`'s own process.)

A genuinely native ARM64 Python does exist on this machine
(`C:\Users\harisury\AppData\Local\Programs\Python\Python312-arm64\python.exe`,
confirmed via PE header `0xAA64`) — but PyTorch ships no Windows ARM64 wheel
on PyPI, so `qai_hub_models`/`transformers`/`easyocr` (everything this repo's
orchestration logic depends on) can't be installed there.

**Resolution: two processes.** A new native-ARM64 venv,
`C:\Users\harisury\qai_env_arm64` (built from the `Python312-arm64`
interpreter, with `onnxruntime-qnn==2.4.0` + `onnx` + `numpy` — no torch, by
necessity), runs a small persistent worker
(`qnn_pipeline\onnxrt_worker.py`) that's the only thing on this machine that
actually calls into `QnnHtp.dll`/the Hexagon NPU for this ONNX path. `qai_env`
(x64, where all the existing torch/qai_hub_models orchestration lives) talks
to it through a thin client, `qnn_pipeline\onnxrt_runner.py`, over
newline-delimited JSON on stdin/stdout (tensors base64-encoded). The worker is
launched once and reused — `InferenceSession`s are cached per
`(model_path, backend)` inside it, so a model already loaded (e.g. across many
calls in Whisper's autoregressive decode loop) only pays compile/load cost
once.

```
qai_env (x64, torch/qai_hub_models orchestration)
  │  onnxrt_runner.run_onnx(model_path, inputs, output_names, backend)
  ▼  [subprocess, stdin/stdout, base64-framed JSON tensors]
qai_env_arm64 (native ARM64, onnxrt_worker.py)
  │  cached ort.InferenceSession per (model_path, backend)
  ▼
QnnHtp.dll → Hexagon NPU
```

QNN's own graph-compile diagnostics (progress bar, per-stage timings) print
straight to the OS-level stdout file descriptor, which would otherwise
collide with the JSON framing on the same pipe — `onnxrt_worker.py` handles
this by duplicating the real fd 1 out before importing `onnxruntime`, then
redirecting fd 1 itself to `devnull`; all IPC goes exclusively through the
saved duplicate.

### ONNX Runtime 1.28's provider API changed from what the docs assume

The classic `InferenceSession(path, providers=['QNNExecutionProvider'],
provider_options=[{...}])` API **silently fails** in the installed ORT
version (1.28, pulled in by `onnxruntime-qnn`): the EPContext node falls back
to CPU EP instead of binding QNN, then errors with `NOT_IMPLEMENTED: Failed to
find kernel for com.microsoft.EPContext ... ep:'CPUExecutionProvider'`. The
working pattern uses ORT's newer plugin-EP device API:

```python
import onnxruntime as ort
import onnxruntime_qnn

ort.register_execution_provider_library(
    "QNNExecutionProvider", onnxruntime_qnn.get_library_path()
)
npu_devices = [
    d for d in ort.get_ep_devices()
    if d.ep_name == "QNNExecutionProvider"
    and d.device.type == ort.OrtHardwareDeviceType.NPU
]
so = ort.SessionOptions()
so.add_provider_for_devices(npu_devices, {"backend_path": onnxruntime_qnn.get_qnn_htp_path()})
sess = ort.InferenceSession(model_path, sess_options=so, providers=[])
```

This is exactly what `onnxrt_worker.py` does. EPContext-wrapped models
(Whisper's `encoder.onnx`/`decoder.onnx`) load through this identical code
path as QDQ graphs (EasyOCR's `detector.onnx`/`recognizer.onnx`) — no
special-casing needed; ORT resolves the sibling `*_qairt_context.bin`
internally for the EPContext case.

### Whisper-base's decoder runs on-NPU — unlike whisper-tiny's

The open question flagged when this path was planned was whether
`decoder.onnx`'s 145MB context binary would hit the same per-PD memory
ceiling that makes whisper-tiny's `decoder.bin` (120MB) CPU-only (see the
Whisper-tiny note earlier in this README). **It does not.** Tested directly:
session creation for `decoder.onnx` takes ~1s and a full forward pass runs
successfully on the NPU. The likely reason is architectural, not just size —
ORT's QNN EP loads the context in-process through this worker, a completely
different loading path from `qnn-net-run.exe`'s subprocess-per-call,
`--retrieve_context` model that whisper-tiny's `decoder.bin` failed under.
Practically, this means `whisper_onnx_pipeline.py` runs **both** encoder and
decoder on the NPU, with no CPU-decoder fallback needed — simpler than
`whisper_pipeline.py`'s hybrid architecture.

### EasyOCR's QDQ graphs: same NCHW contract, quantized I/O

Unlike the `.dlc` files (NHWC float32 I/O), the ONNX graphs declare **NCHW**
input/output (`image: [1,3,608,800]` for the detector) and **uint8**
quantized I/O (w8a8) — `easyocr_pipeline.py`'s `OnnxModule` quantizes plain
`[0,1]` float input to uint8 and dequantizes the uint8 output back to float,
using the per-tensor scale/zero_point recorded in this download's own
`metadata.json`. Inspecting the graph directly (`onnx.load()`) confirmed the
ImageNet mean-subtraction (`Sub` node, constant `[0.485, 0.456, 0.406]`) is
baked into the graph right after the input dequant, with the std-division
folded into the surrounding QDQ scale — so, same as the `.dlc` path, no
external normalization is needed; feed plain `[0,1]` RGB/grayscale.

First-run session creation (graph compile/partition for the QNN EP) costs
noticeably more for these QDQ graphs than for Whisper's already-precompiled
EPContext graphs — observed ~3.5s for `detector.onnx`, ~53s for
`recognizer.onnx` — but this is a one-time cost per `(model_path, backend)`
per worker process lifetime; subsequent calls reuse the cached session
(~25ms per inference after that).

### How to run

```powershell
& "C:\Users\harisury\qai_env\Scripts\python.exe" `
  "C:\Users\harisury\qnn_pipeline\easyocr_pipeline.py" `
  "<path-to-image.png>" --backend htp --runtime onnx

& "C:\Users\harisury\qai_env\Scripts\python.exe" `
  "C:\Users\harisury\qnn_pipeline\whisper_onnx_pipeline.py" `
  "<path-to-audio.wav>" --backend htp
```

`easyocr_pipeline.py --runtime dlc` (the default) is unchanged and still uses
the original `.dlc` files via `qnn_runner.py`. `--runtime onnx` switches to
the path described above. Verified against the same test image: both
runtimes recognize the same text (`HELLO WORLD`), with the expected small
confidence delta from w8a8 quantization (0.904 dlc vs. 0.728 onnx).
`whisper_onnx_pipeline.py` is a separate script (not a flag on
`whisper_pipeline.py`) since it's a genuinely different model
(whisper-base vs. whisper-tiny); verified end-to-end against the same test
audio used for `whisper_pipeline.py`, producing the same correct transcription
("Hello world, this is a test.") with both encoder and decoder running on-NPU.

### Repository layout additions

| File | Purpose |
|---|---|
| `onnxrt_worker.py` | Runs inside `qai_env_arm64` (native ARM64). Persistent subprocess that owns all real QNN EP `InferenceSession`s (cached per model+backend), reads/writes newline-delimited JSON requests/responses on stdin/stdout. |
| `onnxrt_runner.py` | Runs inside `qai_env` (x64). Client API (`run_onnx()`) — launches/talks to `onnxrt_worker.py`, base64-encodes/decodes tensors over the pipe. Mirrors `qnn_runner.py`'s shape but session-cached rather than per-call-subprocess, since ORT sessions (unlike `qnn-net-run.exe` invocations) are meant to be reused. |
| `whisper_onnx_pipeline.py` | Builds `qai_hub_models`' `HfWhisperApp` with NPU-backed encoder **and** decoder callables (`encoder.onnx`/`decoder.onnx` via `onnxrt_runner.run_onnx()`) for whisper-base; feature extraction and tokenization reused unmodified. |
| `easyocr_pipeline.py` (`OnnxModule` + `--runtime` flag) | Second detector/recognizer adapter, backed by `detector.onnx`/`recognizer.onnx` via `onnxrt_runner.run_onnx()`, with uint8 quantize/dequantize at the NPU boundary. Selected via `--runtime onnx`; `--runtime dlc` (default) is the original, unchanged path. |

---

## Voice Q&A pipeline

Genie (Qualcomm's Genie SDK, `genie-t2t-run.exe`) is used here to run
Qwen3-4B for the Q&A step below. A few things about Genie that matter for
understanding the flow:

- **No persistent-process mode.** Unlike `qnn-net-run.exe`'s context-binary
  path, every `run_genie()` call reloads the full ~3.1GB Qwen3-4B context
  binary from disk — expect noticeably slower turnaround per Genie call than
  the EasyOCR/MeloTTS/Whisper stages; this is expected cold-start cost, not
  a bug.
- **`--prompt_file`, not `-p`.** `genie_runner.py` always writes the prompt
  to a temp file and passes `--prompt_file <path>` rather than `-p
  "<prompt>"`, since a multi-turn conversation (or OCR'd records context)
  can be long enough to hit the Windows command-line length limit as a `-p`
  argument; a temp file has no such limit. The temp file is written under
  `scratch\`, deleted immediately after each call.
- **`<think>` reasoning block.** This Qwen3-4B build was created with
  reasoning mode on — every `genie-t2t-run.exe` reply wraps a
  `<think>...</think>` block before the actual answer.
  `genie_runner._parse_response()` strips it automatically; callers only
  ever see the post-`</think>` answer text.

`voice_qa_pipeline.py` chains a different four of the above building blocks —
a microphone recorder, Whisper-base (via the ONNX Runtime path), Genie/
Qwen3-4B, MeloTTS — into a live spoken Q&A loop: ask a question out loud, get
a spoken answer back, with Genie able to ask a clarifying follow-up (spoken,
answered by recording again) instead of guessing when a question is
ambiguous. Optionally, it can also fall back to a patient's historical visit
records (text summaries + OCR'd scans) for questions the live conversation
alone can't answer — see [Patient records fallback](#patient-records-fallback)
below.

```
microphone
   │  record_audio.record_to_wav() -- press Enter to start/stop
   ▼
question.wav
   │
   ▼
[1] Transcribe  (whisper_onnx_pipeline.run(), whisper-base, NPU encoder+decoder)
   │  onnxrt_runner.shutdown_worker() called right after -- releases the
   │  persistent HTP session Whisper holds open, so Genie/MeloTTS can get
   │  exclusive NPU access next (see note below).
   ▼
[2] Ask Genie  (genie_runner.run_genie(), Qwen3-4B, multi-turn ChatML history)
   │  Always replies with a fenced ```json {"status": ..., "text": ...} block.
   │  status="answer"      -> done, speak parsed["text"]
   │  status="clarify"     -> speak parsed["text"] as a follow-up question,
   │                           record a new reply, loop back to [1]
   │  status="need_records"-> (only if --records-dir set) OCR the patient's
   │                           notes/AVS scans and re-ask [2] with that
   │                           context added -- does NOT re-record or consume
   │                           a clarification round
   ▼
[3] Speak  (melotts_pipeline chunking + synthesis, chunks concatenated)
   │
   ▼
auto-played via winsound.PlaySound() (skip with --no-play)
```

This pipeline keeps a real multi-turn conversation, not one-off single-shot
Genie calls: every user question and every Genie reply is appended to a
ChatML `history` list and replayed on every subsequent call
(`_build_multiturn_prompt`), trimmed by `_trim_history` to fit under
`--max-history-tokens` (default 3200, leaving headroom in Qwen3-4B's
4096-token total budget for the system prompt and generation) by dropping
the oldest (user, assistant) turn-pair at a time.

### How to run

```powershell
& "C:\Users\harisury\qai_env\Scripts\python.exe" `
  "C:\Users\harisury\qnn_pipeline\voice_qa_pipeline.py" `
  --backend htp
```

This records live from the microphone. To feed a pre-recorded question
instead of using the mic (e.g. for a repeatable test), pass `--input`:

```powershell
& "C:\Users\harisury\qai_env\Scripts\python.exe" `
  "C:\Users\harisury\qnn_pipeline\voice_qa_pipeline.py" `
  --input "<path-to-question.wav>" --backend htp
```

Note that if Genie asks a clarifying question, the pipeline still records a
*live* follow-up reply from the microphone regardless of whether the first
question came from `--input` or the mic — `--input` only replaces the very
first recording.

To record a WAV standalone (e.g. to build an `--input` fixture, or just to
capture audio without running the rest of the pipeline), `record_audio.py`
also runs on its own:

```powershell
& "C:\Users\harisury\qai_env\Scripts\python.exe" `
  "C:\Users\harisury\qnn_pipeline\record_audio.py" `
  "<path-to-output.wav>"
```

Press Enter to start recording, press Enter again to stop.

| Flag | Default | Purpose |
|---|---|---|
| `--input PATH` | none (records from mic) | Use an existing WAV as the first question instead of recording live. |
| `--backend {cpu,htp}` | `htp` | Backend for Whisper/MeloTTS NPU calls (Genie is HTP-only). |
| `--whisper {base,tiny}` | `base` | `base` uses the ONNX Runtime path (`whisper_onnx_pipeline.py`, encoder+decoder both on NPU); `tiny` uses the `.dlc`/`.bin` path (`whisper_pipeline.py`, CPU decoder — see the Whisper-tiny note earlier in this README). |
| `--output-dir DIR` | `scratch/voice_qa` (or alongside `--input`) | Where `question.wav`, any `clarify_N.wav`/`reply_N.wav`, `answer.wav`, and `conversation.json` are written. |
| `--no-play` | off | Skip auto-playing spoken output (clarifying questions and the final answer). |
| `--max-clarifications N` | 5 | After this many clarification round-trips, Genie is forced to answer with whatever ambiguity remains rather than asking again. |
| `--max-history-tokens N` | 3200 | ChatML history token budget (`_trim_history`) — kept under Genie's 4096-token total context. |
| `--records-dir DIR` | none | Patient records folder (see [Patient records fallback](#patient-records-fallback)) to fall back on for questions the live conversation can't answer. |
| `--ocr-runtime {dlc,onnx}` | `dlc` | Runtime used to OCR patient record images, only relevant with `--records-dir` set — same meaning as EasyOCR's `--runtime` flag above. |

Each run writes, under `--output-dir`: `question.wav` (the first question,
copied/recorded), `clarify_N.wav`/`reply_N.wav` per clarification round,
`answer.wav` (the final spoken answer), and `conversation.json` (the full
ChatML `history` list, for debugging what Genie actually saw).

### Why a JSON status envelope

`genie-t2t-run.exe` is a plain text-completion CLI with no JSON-schema/
structured-decoding mode, so a JSON envelope here is a convention this
pipeline enforces on top of it, not something Genie guarantees.
`voice_qa_pipeline.py` needs it anyway because it has a real decision to
make on every reply — answer now, or ask the user something — that
plain-text regex parsing can't express as cleanly as a `status` field.
`_extract_json_block` fails open: an unparseable response is treated as a
direct `"answer"` (with a printed warning) rather than crashing the voice
loop over one bad Genie turn.

### NPU device contention between Whisper and Genie/MeloTTS

`whisper_onnx_pipeline.py` (the default `--whisper base` path) keeps a
persistent ARM64 worker process alive with an open QNN HTP `InferenceSession`
across calls, by design, so repeated Whisper calls are fast. If Genie or
MeloTTS then tried to acquire the same HTP device while that session is still
open, it fails with `err 5000` ("Could not create context from binary") /
"Failed to free device: 14003" — real device contention, not the flaky-DSP-
open issue `qnn_runner.py` already retries for. `run()` calls
`onnxrt_runner.shutdown_worker()` immediately after each Whisper transcription
and before the first Genie call to release the device. (`--whisper tiny`
doesn't need this — `whisper_pipeline.py`'s decoder already runs on CPU, so
it never holds the HTP device open.)

### Patient records fallback

With `--records-dir` set, the pipeline can answer questions about a patient's
history — not just the live conversation — by reading a folder laid out as:

```
<records_dir>/
  <visit_1>/
    summary.txt          (or any *.txt -- plain text, read directly)
    <notes image>.png    (any image file -- doctor's notes)
    <avs image>.png      (any image file -- AVS)
  <visit_2>/
    ...
```

Visit subfolders are natural-sorted the same way `doctor_note_pipeline.py`
sorts multi-page images (`patient_records.discover_visits`, "visit_2" before
"visit_10"). This convention is scoped to **one patient's folder per run** —
there's no patient-identification logic — and a query searches across **all**
visits in the folder, not just the most recent.

Retrieval is two-tiered, cheapest first: `load_summaries_text` reads every
visit's `*.txt` summary once at the start of `run()` and includes it in every
system prompt. Only if Genie replies `status="need_records"` — meaning the
summaries alone weren't enough — does the pipeline pay for
`ocr_visit_images_text` (one EasyOCR pass per image, reusing
`doctor_note_pipeline.py`'s OCR/formatting helpers wholesale, since a visit
folder's images are exactly the "directory of page images" shape that
pipeline already handles). That OCR result (`avs_context`) is cached and
reused for the rest of the run rather than recomputed. Both are packed
most-recent-first under a 2000-token budget
(`patient_records.RECORDS_CONTEXT_MAX_TOKENS`, measured via
`chunking.count_genie_tokens`) — Genie's 4096-token total context must also
fit the system prompt, live conversation history, and generation, so if
everything doesn't fit, the *oldest* visits are dropped first (and printed as
a `[warning]`, never silently lost).

`need_records` is a third status value, distinct from `clarify`: `clarify`
means "ask the user something their chart wouldn't contain"; `need_records`
means "the answer might be in this patient's history, fetch more detail
automatically." The inner retry loop in `run()` re-asks Genie with the newly
OCR'd context immediately, without recording new audio or counting against
`--max-clarifications`. If Genie still replies `need_records` after the OCR
context is already loaded — nothing left to escalate to — the pipeline
downgrades it to a real `clarify` instead of looping forever.

With `--records-dir` unset (the default), none of this runs: no records
instructions are added to the system prompt, and behavior is byte-for-byte
identical to the pipeline before this feature existed.

### Repository layout additions

| File | Purpose |
|---|---|
| `record_audio.py` | Push-to-talk mic recorder (`sounddevice`/PortAudio) — press Enter to start/stop, writes a WAV via `soundfile`. Includes a `platform.machine()` monkeypatch working around `qai_env`'s x64-under-ARM64 process reporting the wrong architecture to `sounddevice`'s bundled-DLL selection. |
| `voice_qa_pipeline.py` | End-to-end orchestrator: mic recording -> Whisper transcription -> Genie (JSON-envelope, multi-turn, clarification loop) -> MeloTTS -> auto-play. See above. |
| `patient_records.py` | Patient visit-history lookup used by `voice_qa_pipeline.py`'s `--records-dir` fallback: discovers visit subfolders, packs `*.txt` summaries and (on demand) OCR'd images most-recent-first under a token budget. See [Patient records fallback](#patient-records-fallback) above. |

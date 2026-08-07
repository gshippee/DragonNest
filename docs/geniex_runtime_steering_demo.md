# SteerLab Demo Guide — GenieX Runtime Activation Steering on the S25 Ultra

One compiled Qwen3-0.6B W4A16 bundle; steering strength `alpha` and the
`steering_vector` change **between requests** with no recompile, no relink, no
reload. Package `com.dragonnest.geniexsteeringlab` — fully independent of
PersonaCare / the historical demo apps.

## What the audience sees

1. App loads the bundle ONCE — status line shows load time and
   `context loads: 1 (reused hereafter)`.
2. Fixed prompt, alpha slider at 0, vector "layer-7 verbosity (real)" →
   Generate. Baseline answer.
3. Slide alpha to −10 (concise) → Generate. Same context, new alpha, answer
   shortens. Slide to +10 (verbose) → Generate. Answer lengthens.
4. Switch vector to "random control (same norm)" → Generate. Direction-specific
   effect disappears (norm-matched control).
5. Switch to "unsteered (no aux inputs)" → Generate. Bit-identical stock GenieX
   request path.
6. The diagnostics panel prints, per request, straight from the runtime's
   `geniex_llm_get_aux_stats` counters (not client-side guesses):

```
req #3 ▸ submit  aux=ON  alpha=+10.0  vector=layer7-verbosity  (same loaded context, no reload)
req #3 ◂ done in 2140 ms  (23 prompt tok, 96 gen tok, 44.9 tok/s)
req #3 ◂ PREFILL graphs got 2 aux writes, DECODE graphs got 192 aux writes
totals ▸ prefill_writes=6 decode_writes=384 aux_requests=3/5 context_loads=1
```

The prefill/decode split makes the correctness story visible: BOTH the prompt
pass and every generated token are steered — no prompt-only or decode-only
accidents.

## Setup (shared device rules respected)

All device state lives in our own namespaces; nothing belonging to
PersonaCare / the historical demo is touched. Announce device use first.

```powershell
$ADB = "C:\Users\shubh\Downloads\qcom_hackathon\artifacts\tools\android-sdk\platform-tools\adb.exe"
# 1. Install/refresh our app (safe to reinstall freely)
& $ADB install -r <workspace>\artifacts\steerlab\app-debug.apk
# 2. Push the steering-capable bundle into the app's private external storage
& $ADB shell mkdir -p /sdcard/Android/data/com.dragonnest.geniexsteeringlab/files/qwen-bundle
& $ADB push <steered-bundle>\. /sdcard/Android/data/com.dragonnest.geniexsteeringlab/files/qwen-bundle/
# 3. Launch
& $ADB shell am start -n com.dragonnest.geniexsteeringlab/.MainActivity
```

The bundle must contain `aux_inputs.json` (`{"aux_inputs": ["alpha",
"steering_vector"]}`) alongside the context binaries — see the implementation
doc for the full compiled-model contract.

## CLI fallback (no APK needed)

`/data/local/tmp/geniexsteeringlab/aux_steering_probe` runs the same forked
runtime against any bundle:

```
./aux_steering_probe --bundle <dir> --ids 1,2,3,4 \
    --vector vec.bin --alpha 0 --alpha 5 --alpha -5 [--generate 32]
```

One JSON line per run: alpha, logits head/checksum or generated text, latency,
`reload_count: 0`. The tiny test bundle (`testmodel/bundle`) demonstrates the
mechanism numerically (see `artifacts/geniex_runtime_steering/s25_test_graph_proof.json`).

## Honest-claims checklist

- Steering claims rest on fixed prompts, greedy (temperature 0) decoding, a
  norm-matched random-vector control, and an alpha sweep — not single
  anecdotes.
- The mechanism proof (tensors live on HTP, per-phase binding, no reload) is
  numerical: device deltas match `alpha * (vec @ W)` within fp16 tolerance on
  the test graph.
- Semantic quality of ±10 on the W4A16 model is whatever the model produces —
  the demo shows the *mechanism* plus direction; calibrated behavior metrics
  (neutral NLL, refusal rates) remain future work.

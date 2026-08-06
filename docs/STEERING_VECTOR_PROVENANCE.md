# Steering Vector Provenance

## Demo-approved vector: `concise-vs-verbose-layer-7`

| Field | Recovered value | Status |
|---|---|---|
| Base model | `Qwen/Qwen3-0.6B` | verified locally without hardware |
| Model/tokenizer revision | `c1899de289a04d12100db370d81485cdf75e47ca` | verified locally without hardware |
| Hidden size | 1024 | verified locally without hardware |
| Extraction/injection layer | output of layer 7 / layer 7 | verified locally without hardware |
| Token policy | last non-padding token | verified locally without hardware |
| Concept | verbose (positive) minus concise (negative) | verified locally without hardware |
| Extraction | mean paired positive-minus-negative direction | verified locally without hardware |
| Normalization | raw mean difference; L2 normalize at injection | verified locally without hardware |
| Dtype | float32 | verified locally without hardware |
| Training pairs | 19; seed 1234 | verified locally without hardware |
| Holdout indices | 4, 6, 10, 17, 23 | verified locally without hardware |
| Dataset hash | `eaee40898b46b295d2197631d7863624a18624b71f000d9a8563d1df540fb3f4` | verified locally without hardware |
| Vector hash | `7d69ff39a248a6e7df11d5fe2b533addfc19f84da75c4869a088eef8a2c32b2c` | verified locally without hardware |
| Sweep | alpha -4, -2, 0, 2, 4 | verified locally without hardware |
| Selected baked profile | layer 7, alpha -4 | verified locally without hardware |

The vector and dataset were recovered from the adjacent `qcom_hackathon`
workspace. Both copies of the layer-7 vector have the same SHA-256. The source
manifest records a raw vector norm of 8.41692543 and a mean pair-difference norm
of 12.77664852.

At alpha -4, recovered response-length evaluation reports a mean paired
reduction of about 46 words, with a 95% bootstrap interval of about 29 to 65
fewer words. The baked ONNX rewrite passed graph/external-data checks and the
base and baked variants compiled for the S25 class through AI Hub.

The recovered summary does not contain sufficient per-example evidence for
neutral-text NLL/perplexity, KL divergence, refusal/pathology regressions,
layer sensitivity, or repeatability across prompt templates. Those fields are
marked **unverified** in `configs/steering-vectors.yaml`; they must not be
silently promoted to validated metrics.

## Realization policy

- `qwen3-0.6b-s25-concise` is `baked_profile`, profile
  `concise-l7-alpha-m4`. It is an activation intervention compiled into a
  separate artifact, not a runtime vector input.
- The base S25 and X Elite Genie artifacts advertise `none`.
- The QNN Part-B runtime-vector experiment at layer 21 is valid AI Hub evidence
  that QNN can preserve vector/alpha inputs. It is not the layer-7 demo artifact
  and is not exposed as a full-model DragonNest deployment.
- Stock Genie/GenieX full-model deployments advertise no `runtime_vector` mode.
- A prompt profile, if added later, must advertise `prompt_profile` and must
  never populate `steering_vector_ids` or `supports_steering`.

## Reproducible source procedure

From the adjacent research repository with its original environment:

```powershell
$env:PYTHONPATH="$PWD\src"
.\.venv\Scripts\python.exe -m steering_poc.extract --config configs\qwen3_0_6b.yaml --data data\contrast_pairs.jsonl
.\.venv\Scripts\python.exe -m steering_poc.evaluate --config configs\qwen3_0_6b.yaml --layers 7 --alphas -4 -2 0 2 4 --prompts data\eval_prompts.jsonl --out-prefix artifacts\eval_layer7
Get-FileHash -Algorithm SHA256 artifacts\vector_layer_7.pt
```

Before using a regenerated vector, verify the exact revision/tokenizer, dataset
hash, layer, hidden width, dtype, alpha sweep, held-out output, and checksum.
Reject it if any of these cannot be reconstructed.

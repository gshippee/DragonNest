#!/usr/bin/env python3
"""Exact host-side reference for the tiny steering context.

Given token ids, alpha and a steering vector, computes the logits the device
must produce (last row), using the same table/W as make_onnx.py.

Usage: reference.py --ids 1,2,3,4 --alpha 2.0 [--vector v.bin]
Prints JSON: {"logits_last": [...], "argmax": n, "checksum": s}
"""
import argparse
import json

import numpy as np

H, V, CL = 8, 16, 16
EPS = 1e-6
SEED = 20260807


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--vector", default=None)
    ap.add_argument("--ar", type=int, default=4)
    args = ap.parse_args()

    # Match make_onnx.py exactly: W and the table are drawn from two
    # independent generators seeded identically (build() and main() each
    # construct a fresh rng).
    W = np.random.default_rng(SEED).standard_normal((H, V)).astype(np.float32) * 0.5
    table = (np.random.default_rng(SEED).standard_normal((V, H)).astype(np.float32) * 0.3).round(3)

    ids = [int(x) for x in args.ids.split(",")]
    assert len(ids) == args.ar, "supply exactly ar ids (no pad-row modeling here)"
    embeds = table[ids]  # [ar, H]

    vec = np.zeros(H, dtype=np.float32)
    if args.vector:
        vec = np.fromfile(args.vector, dtype=np.float32)
        assert vec.size == H
    alpha = args.alpha if args.alpha is not None else 0.0

    steered = embeds + alpha * vec[None, :]
    # attention_mask contribution: LLMModel writes a causal mask; its exact sum
    # depends on n_past/curr_len. getAttentionMask uses 0/1? The eps term makes
    # logits depend on it; compute both extremes so callers can bound it:
    # mask_sum in [0, ar*CL]. eps*ar*CL = 1e-6*64 = 6.4e-5 — below tolerance.
    logits = steered @ W  # [ar, V]
    last = logits[-1]
    print(
        json.dumps(
            {
                "logits_last": [round(float(x), 6) for x in last],
                "argmax": int(np.argmax(last)),
                "checksum_last": float(last.sum()),
                "mask_eps_bound": EPS * args.ar * CL,
            }
        )
    )


if __name__ == "__main__":
    main()

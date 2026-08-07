#!/usr/bin/env python3
"""Builds the tiny steering-capable LLM-shaped ONNX graph pair.

Mirrors the tensor set of geniex-qairt's proven-loadable LLM test fixture
(tests/testing/llm_fixture.hpp) plus the two runtime steering inputs:

  prefill_ar4_cl16_1_of_1 : ar=4
  token_ar1_cl16_1_of_1   : ar=1

  inputs : steering_vector f32 [1,1,H]   (runtime aux)
           alpha           f32 [1]       (runtime aux)
           input_embeds    f32 [ar,H]
           attention_mask  f32 [ar,CL]
           past_key_0_in   f32 [1,1,D,KV]
           past_value_0_in f32 [1,1,KV,D]
  outputs: logits          f32 [1,ar,V]
           past_key_0_out  f32 [1,1,D,ar]
           past_value_0_out f32 [1,1,ar,D]

  math   : steered = input_embeds + alpha * steering_vector
           logits  = steered @ W + eps * ReduceSum(attention_mask)
           KV outs = slice of KV ins (keeps them live through conversion)

Also writes embedding_weights.raw (V x H float32) and reference.py inputs so
the host can compute exact expected logits for any (ids, alpha, vector).
"""
import json
import struct
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

H, V, CL, D, KV = 8, 16, 16, 2, 15  # hidden, vocab, context, head_dim, kv_capacity
EPS = 1e-6
SEED = 20260807


def build(ar: int, name: str) -> onnx.ModelProto:
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((H, V)).astype(np.float32) * 0.5

    inputs = [
        helper.make_tensor_value_info("steering_vector", TensorProto.FLOAT, [1, 1, H]),
        helper.make_tensor_value_info("alpha", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("input_embeds", TensorProto.FLOAT, [ar, H]),
        helper.make_tensor_value_info("attention_mask", TensorProto.FLOAT, [ar, CL]),
        helper.make_tensor_value_info("past_key_0_in", TensorProto.FLOAT, [1, 1, D, KV]),
        helper.make_tensor_value_info("past_value_0_in", TensorProto.FLOAT, [1, 1, KV, D]),
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, ar, V]),
        helper.make_tensor_value_info("past_key_0_out", TensorProto.FLOAT, [1, 1, D, ar]),
        helper.make_tensor_value_info("past_value_0_out", TensorProto.FLOAT, [1, 1, ar, D]),
    ]

    W_init = helper.make_tensor("W", TensorProto.FLOAT, [H, V], W.flatten().tolist())
    eps_init = helper.make_tensor("eps", TensorProto.FLOAT, [1], [EPS])
    k_starts = helper.make_tensor("k_starts", TensorProto.INT64, [1], [KV - ar])
    k_ends = helper.make_tensor("k_ends", TensorProto.INT64, [1], [KV])
    ax3 = helper.make_tensor("ax3", TensorProto.INT64, [1], [3])
    ax2 = helper.make_tensor("ax2", TensorProto.INT64, [1], [2])

    nodes = [
        helper.make_node("Mul", ["steering_vector", "alpha"], ["scaled_vec"]),
        helper.make_node("Add", ["input_embeds", "scaled_vec"], ["steered"]),  # -> [1,ar,H]
        helper.make_node("MatMul", ["steered", "W"], ["proj"]),  # -> [1,ar,V]
        helper.make_node("ReduceSum", ["attention_mask"], ["mask_sum"], keepdims=0),  # scalar
        helper.make_node("Mul", ["mask_sum", "eps"], ["mask_eps"]),
        helper.make_node("Add", ["proj", "mask_eps"], ["logits"]),
        helper.make_node("Slice", ["past_key_0_in", "k_starts", "k_ends", "ax3"], ["past_key_0_out"]),
        helper.make_node("Slice", ["past_value_0_in", "k_starts", "k_ends", "ax2"], ["past_value_0_out"]),
    ]

    graph = helper.make_graph(
        nodes, name, inputs, outputs, initializer=[W_init, eps_init, k_starts, k_ends, ax3, ax2]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main(outdir: str) -> None:
    rng = np.random.default_rng(SEED)
    table = (rng.standard_normal((V, H)).astype(np.float32) * 0.3).round(3)

    for ar, name in [(4, "prefill_ar4_cl16_1_of_1"), (1, "token_ar1_cl16_1_of_1")]:
        m = build(ar, name)
        onnx.save(m, f"{outdir}/{name}.onnx")
        print("wrote", name, ".onnx")

    with open(f"{outdir}/embedding_weights.raw", "wb") as f:
        f.write(table.tobytes())
    np.save(f"{outdir}/embedding_table.npy", table)

    with open(f"{outdir}/aux_inputs.json", "w") as f:
        json.dump({"aux_inputs": ["alpha", "steering_vector"]}, f)

    genie_cfg = {
        "dialog": {
            "type": "basic",
            "context": {"size": CL, "n-vocab": V, "bos-token": -1, "eos-token": []},
            "engine": {
                "backend": {"type": "QnnHtp"},
                "model": {"type": "binary", "binary": {"ctx-bins": ["prefill.bin", "decode.bin"]}},
            },
        }
    }
    with open(f"{outdir}/genie_config.json", "w") as f:
        json.dump(genie_cfg, f, indent=1)
    print("wrote sidecars")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

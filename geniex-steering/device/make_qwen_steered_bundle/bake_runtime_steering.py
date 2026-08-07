#!/usr/bin/env python3
"""Rewrites the Qwen3-0.6B W4A16 ONNX checkpoint so activation steering is a
RUNTIME input pair instead of a baked constant.

Derived from PersonaCare's bake_static_steering.py (which inserted
Add(residual, alpha*unit(v)) as a constant). This variant adds two graph
INPUTS and the ops:

    alpha_3d = Reshape(alpha, [1,1,1])
    scaled   = Mul(steering_vector, alpha_3d)
    steered  = Add(layer7_residual, scaled)

so one compiled bundle serves every alpha in [-ALPHA_MAX, ALPHA_MAX] and any
1024-d vector without recompilation. The steering vector itself is NOT baked
anywhere; callers supply it per request (e.g. unit(vector_layer_7.pt)).

Encodings: the checkpoint is an AIMET-quantized export and the AI Hub compile
uses --quantize_io, so the new inputs/intermediates need activation encodings.
alpha is ranged +/-ALPHA_MAX_ENC, the vector +/-VEC_MAX_ENC (unit vectors have
max |component| well under that), and the steered tensor's cloned encoding is
widened by the maximum steering delta so extreme alphas do not clip.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

MODEL_ID = "Qwen/Qwen3-0.6B"
TARGET_TENSOR = "/model/model/layers.7/Add_1_output_0"
STEERED_TENSOR = "/steeringlab/runtime_layer7/Add_output_0"
SCALED_TENSOR = "/steeringlab/runtime_layer7/Mul_output_0"
ALPHA3D_TENSOR = "/steeringlab/runtime_layer7/alpha_3d"
SHAPE_INIT = "steeringlab.alpha_shape_111"
VEC_INPUT = "steering_vector"
ALPHA_INPUT = "alpha"
HIDDEN = 1024

ALPHA_MAX_ENC = 12.0  # slider is -10..+10; margin for headroom
VEC_MAX_ENC = 0.5     # unit 1024-d vectors: max |component| typically < 0.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sym16(name: str, max_abs: float) -> dict:
    return {
        "bw": 16,
        "dtype": "INT",
        "enc_type": "PER_TENSOR",
        "is_sym": True,
        "name": name,
        "offset": [-32768.0],
        "scale": [max_abs / 32767.0],
    }


def add_encodings(encodings: dict) -> None:
    activation = encodings["activation_encodings"]

    source = [x for x in activation if x.get("name") == TARGET_TENSOR]
    if len(source) != 1:
        raise ValueError(f"expected one activation encoding for {TARGET_TENSOR}, found {len(source)}")
    for name in (STEERED_TENSOR, SCALED_TENSOR, ALPHA3D_TENSOR, VEC_INPUT, ALPHA_INPUT):
        if any(x.get("name") == name for x in activation):
            raise ValueError(f"activation encoding for {name} already exists")

    # Steered residual: clone the source range, widened by the max steering
    # delta ALPHA_MAX_ENC * VEC_MAX_ENC per channel.
    steered = copy.deepcopy(source[0])
    steered["name"] = STEERED_TENSOR
    delta = ALPHA_MAX_ENC * VEC_MAX_ENC
    scales = steered.get("scale")
    if isinstance(scales, list) and scales:
        steered["scale"] = [float(s) + delta / 32767.0 for s in scales]
    activation.append(steered)

    activation.append(sym16(VEC_INPUT, VEC_MAX_ENC))
    activation.append(sym16(ALPHA_INPUT, ALPHA_MAX_ENC))
    activation.append(sym16(ALPHA3D_TENSOR, ALPHA_MAX_ENC))
    activation.append(sym16(SCALED_TENSOR, ALPHA_MAX_ENC * VEC_MAX_ENC))


def rewrite_graph(model: onnx.ModelProto) -> int:
    graph = model.graph
    for node in graph.node:
        if node.name.startswith("/steeringlab/"):
            raise ValueError("runtime steering nodes already exist")
    existing_inputs = {i.name for i in graph.input}
    if VEC_INPUT in existing_inputs or ALPHA_INPUT in existing_inputs:
        raise ValueError("steering inputs already exist")

    producers = [(i, n) for i, n in enumerate(graph.node) if TARGET_TENSOR in n.output]
    if len(producers) != 1:
        raise ValueError(f"expected one producer for {TARGET_TENSOR}, found {len(producers)}")
    producer_index, _ = producers[0]

    replaced = 0
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == TARGET_TENSOR:
                node.input[index] = STEERED_TENSOR
                replaced += 1
    for output in graph.output:
        if output.name == TARGET_TENSOR:
            output.name = STEERED_TENSOR
            replaced += 1
    if replaced == 0:
        raise ValueError(f"{TARGET_TENSOR} has no consumers")

    graph.input.extend(
        [
            helper.make_tensor_value_info(VEC_INPUT, TensorProto.FLOAT, [1, 1, HIDDEN]),
            helper.make_tensor_value_info(ALPHA_INPUT, TensorProto.FLOAT, [1]),
        ]
    )
    graph.initializer.append(
        helper.make_tensor(SHAPE_INIT, TensorProto.INT64, [3], [1, 1, 1])
    )
    nodes = [
        helper.make_node(
            "Reshape", [ALPHA_INPUT, SHAPE_INIT], [ALPHA3D_TENSOR], name="/steeringlab/runtime_layer7/Reshape"
        ),
        helper.make_node(
            "Mul", [VEC_INPUT, ALPHA3D_TENSOR], [SCALED_TENSOR], name="/steeringlab/runtime_layer7/Mul"
        ),
        helper.make_node(
            "Add", [TARGET_TENSOR, SCALED_TENSOR], [STEERED_TENSOR], name="/steeringlab/runtime_layer7/Add"
        ),
    ]
    for offset, node in enumerate(nodes):
        graph.node.insert(producer_index + 1 + offset, node)
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_onnx = args.checkpoint / "model_dynamic.onnx"
    source_encodings = args.checkpoint / "model.encodings"
    source_data = args.checkpoint / "model.data"
    for required in (source_onnx, source_encodings, source_data):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    model = onnx.load(source_onnx, load_external_data=False)
    consumer_count = rewrite_graph(model)
    onnx.save_model(model, args.output / "model_dynamic.onnx")

    with source_encodings.open("r", encoding="utf-8") as handle:
        encodings = json.load(handle)
    add_encodings(encodings)
    with (args.output / "model.encodings").open("w", encoding="utf-8") as handle:
        json.dump(encodings, handle, indent=2, sort_keys=True)

    # ONNX rejects hard-linked external data, so this must be a real copy.
    shutil.copy2(source_data, args.output / "model.data")
    for filename in (
        "args.json",
        "chat_template.jinja",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        source = args.checkpoint / filename
        if source.is_file():
            shutil.copy2(source, args.output / filename)

    onnx.checker.check_model(str(args.output / "model_dynamic.onnx"))

    manifest = {
        "format": "steeringlab-runtime-steering-v1",
        "model_id": MODEL_ID,
        "source_checkpoint": args.checkpoint.name,
        "source_onnx_sha256": sha256(source_onnx),
        "capture_point": TARGET_TENSOR,
        "steered_tensor": STEERED_TENSOR,
        "runtime_inputs": {
            ALPHA_INPUT: {"shape": [1], "dtype": "float32", "encoded_range": ALPHA_MAX_ENC},
            VEC_INPUT: {"shape": [1, 1, HIDDEN], "dtype": "float32", "encoded_range": VEC_MAX_ENC},
        },
        "rewritten_consumers": consumer_count,
        "note": "no vector or alpha value is baked; supply both per request",
    }
    with (args.output / "runtime_steering_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

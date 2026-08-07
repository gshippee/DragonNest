#!/usr/bin/env python3
"""Reproduce PersonaCare's static layer-7 Qwen3-0.6B profile bake.

This inserts an ordinary ONNX Add of ``alpha * unit(vector)`` after layer 7.
It does not add a runtime steering input or hook. The source and vector hashes
are pinned to the physically recovered demo provenance.
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
import torch
from onnx import helper, numpy_helper


MODEL_ID = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SOURCE_ONNX_SHA256 = "fc390cb9f1a26a36d6e079ce5ceb83c51ba0379cd2a6a2713b0245f8bf7439e2"
VECTOR_SHA256 = "7d69ff39a248a6e7df11d5fe2b533addfc19f84da75c4869a088eef8a2c32b2c"
TARGET_TENSOR = "/model/model/layers.7/Add_1_output_0"
STEERED_TENSOR = "/personacare/static_verbosity_layer7/Add_output_0"
CONSTANT_NAME = "personacare.static_verbosity_layer7.alpha_vector"
NODE_NAME = "/personacare/static_verbosity_layer7/Add"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vector(path: Path) -> tuple[np.ndarray, dict]:
    if sha256(path) != VECTOR_SHA256:
        raise ValueError("steering vector checksum does not match recovered provenance")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = dict(payload.get("metadata", {}))
    vector = payload["vector"].detach().float().cpu().numpy()
    if metadata.get("model_id") != MODEL_ID or metadata.get("layer") != 7:
        raise ValueError("vector model/layer metadata is not Qwen3-0.6B layer 7")
    if vector.shape != (1024,):
        raise ValueError(f"vector shape is {vector.shape}, expected (1024,)")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"invalid vector norm {norm}")
    return vector / norm, metadata


def rewrite_graph(model: onnx.ModelProto, constant: np.ndarray) -> int:
    producers = [(i, node) for i, node in enumerate(model.graph.node) if TARGET_TENSOR in node.output]
    if len(producers) != 1 or any(node.name == NODE_NAME for node in model.graph.node):
        raise ValueError("checkpoint does not contain exactly one untouched layer-7 target")
    producer_index, _ = producers[0]
    replaced = 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == TARGET_TENSOR:
                node.input[index] = STEERED_TENSOR
                replaced += 1
    if not replaced:
        raise ValueError("layer-7 tensor has no consumers")
    model.graph.initializer.append(
        numpy_helper.from_array(constant.reshape(1, 1, 1024).astype(np.float32), name=CONSTANT_NAME)
    )
    model.graph.node.insert(
        producer_index + 1,
        helper.make_node("Add", [TARGET_TENSOR, CONSTANT_NAME], [STEERED_TENSOR], name=NODE_NAME),
    )
    return replaced


def add_encoding(encodings: dict, constant: np.ndarray) -> None:
    activation = encodings["activation_encodings"]
    params = encodings["param_encodings"]
    source = [item for item in activation if item.get("name") == TARGET_TENSOR]
    if len(source) != 1:
        raise ValueError("expected one source activation encoding")
    output = copy.deepcopy(source[0])
    output["name"] = STEERED_TENSOR
    activation.append(output)
    max_abs = max(float(np.max(np.abs(constant))), np.finfo(np.float32).eps)
    params.append({"bw": 16, "dtype": "INT", "enc_type": "PER_TENSOR", "is_sym": True, "name": CONSTANT_NAME, "offset": [-32768.0], "scale": [max_abs / 32767.0]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, choices=(-4.0, 4.0), required=True)
    args = parser.parse_args()
    source_onnx = args.checkpoint / "model_dynamic.onnx"
    if sha256(source_onnx) != SOURCE_ONNX_SHA256:
        raise ValueError("source ONNX checksum does not match the recovered base")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    unit_vector, vector_metadata = load_vector(args.vector)
    constant = (args.alpha * unit_vector).astype(np.float32)
    model = onnx.load(source_onnx, load_external_data=False)
    consumers = rewrite_graph(model, constant)
    onnx.save_model(model, args.output / "model_dynamic.onnx")
    encodings = json.loads((args.checkpoint / "model.encodings").read_text(encoding="utf-8"))
    add_encoding(encodings, constant)
    (args.output / "model.encodings").write_text(json.dumps(encodings, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copy2(args.checkpoint / "model.data", args.output / "model.data")
    for filename in ("args.json", "chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json"):
        source = args.checkpoint / filename
        if source.is_file():
            shutil.copy2(source, args.output / filename)
    onnx.checker.check_model(str(args.output / "model_dynamic.onnx"))
    manifest = {
        "format": "personacare-static-steering-v1",
        "model_id": MODEL_ID,
        "base_revision": BASE_REVISION,
        "source_onnx_sha256": sha256(source_onnx),
        "baked_onnx_sha256": sha256(args.output / "model_dynamic.onnx"),
        "vector_sha256": sha256(args.vector),
        "vector_metadata": vector_metadata,
        "layer": 7,
        "alpha": args.alpha,
        "normalization": "l2_at_injection",
        "capture_point": TARGET_TENSOR,
        "steered_tensor": STEERED_TENSOR,
        "rewritten_consumers": consumers,
        "constant_min": float(constant.min()),
        "constant_max": float(constant.max()),
        "constant_l2_norm": float(np.linalg.norm(constant)),
    }
    (args.output / "static_steering_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


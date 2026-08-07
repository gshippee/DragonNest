"""QNN calls and tensor preparation proven by the X Elite bring-up harness.

This production module preserves the physically recovered graph names,
per-tensor quantization, real transformers RoPE/mask primitives, absolute
layer indices, prompt/decode lengths, and HTP-only execution. It deliberately
keeps qnn-net-run's per-call context reload because that is the proven path.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from . import qnn_runner

CONTEXT_LENGTH = 512
SEQUENCE_LENGTH = 128
PAST_LEN_PROMPT = CONTEXT_LENGTH - SEQUENCE_LENGTH
PAST_LEN_DECODE = CONTEXT_LENGTH - 1
BASE_MODEL_ID = os.environ.get(
    "DRAGONNEST_QWEN17_TOKENIZER", "Qwen/Qwen3-1.7B"
)


def ensure_binary_info(bin_path: str | Path, cache_json_path: Path) -> dict:
    if not cache_json_path.exists():
        cache_json_path.parent.mkdir(parents=True, exist_ok=True)
        utility = qnn_runner.BIN_DIR / "qnn-context-binary-utility.exe"
        if not utility.exists():
            raise FileNotFoundError(
                f"qnn-context-binary-utility.exe not found at {utility}; set QAIRT_ROOT"
            )
        result = subprocess.run(
            [
                str(utility),
                "--context_binary",
                str(bin_path),
                "--json_file",
                str(cache_json_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or not cache_json_path.exists():
            raise RuntimeError(
                f"qnn-context-binary-utility.exe failed for {bin_path} "
                f"(exit {result.returncode})\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
    return json.loads(cache_json_path.read_text(encoding="utf-8"))


def graph_io(meta: dict, graph_name: str) -> dict:
    for graph in meta["info"]["graphs"]:
        if graph["info"]["graphName"] == graph_name:
            return graph["info"]
    raise KeyError(
        f"graph {graph_name!r} not found; available: "
        f"{[graph['info']['graphName'] for graph in meta['info']['graphs']]}"
    )


def find_graph_index(meta: dict, graph_name: str) -> int:
    for index, graph in enumerate(meta["info"]["graphs"]):
        if graph["info"]["graphName"] == graph_name:
            return index
    raise KeyError(graph_name)


def scale_offset(tensors: list[dict], name: str) -> tuple[float, int]:
    for tensor in tensors:
        info = tensor["info"]
        if info["name"] == name:
            params = info["quantizeParams"]["scaleOffset"]
            return params["scale"], params["offset"]
    raise KeyError(name)


def quantize_u16(real: np.ndarray, scale: float, offset: int) -> np.ndarray:
    raw = np.round(real / scale - offset)
    return np.clip(raw, 0, 65535).astype(np.uint16)


def dequantize(raw: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return (raw.astype(np.float32) + offset) * scale


_rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None


def rope_cos_sin_full() -> tuple[torch.Tensor, torch.Tensor]:
    global _rope_cache
    if _rope_cache is None:
        config = AutoConfig.from_pretrained(BASE_MODEL_ID)
        rope = LlamaRotaryEmbedding(config=config)
        position_ids = torch.arange(CONTEXT_LENGTH).view(1, -1)
        cos_full, sin_full = rope.forward(torch.tensor([1.0]), position_ids)
        embedding_size = cos_full.size(-1) // 2
        cos_full = cos_full[:, :, :embedding_size].unsqueeze(0)
        sin_full = sin_full[:, :, :embedding_size].unsqueeze(0)
        _rope_cache = (cos_full, sin_full)
    return _rope_cache


def position_cos_sin(position_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    cos_full, sin_full = rope_cos_sin_full()
    cos = cos_full[0, 0, :, :][position_ids].unsqueeze(1)
    sin = sin_full[0, 0, :, :][position_ids].unsqueeze(1)
    return cos.numpy(), sin.numpy()


def causal_attention_mask(num_real_tokens: int, query_length: int) -> np.ndarray:
    attention_mask = torch.zeros((1, CONTEXT_LENGTH))
    attention_mask[:, -num_real_tokens:] = 1.0
    mask = AttentionMaskConverter(True).to_4d(
        attention_mask,
        query_length=query_length,
        key_value_length=CONTEXT_LENGTH,
        dtype=torch.float32,
    )
    return mask.clip(-100, 0).numpy()


def run_s0_prompt(
    bin_path: str | Path, meta: dict, input_ids: np.ndarray
) -> tuple[np.ndarray, float]:
    graph_name = "prompt_ar128_cl512_1_of_4"
    start = time.time()
    outputs = qnn_runner.run_context_binary(
        bin_path=bin_path,
        inputs={"input_ids": input_ids},
        output_names=["embedding"],
        output_shapes={"embedding": (1, SEQUENCE_LENGTH, 2048)},
        output_dtypes={"embedding": np.uint16},
        backend="htp",
        timeout_sec=120,
        graph_index=find_graph_index(meta, graph_name),
        num_graphs=2,
    )
    return outputs["embedding"], time.time() - start


def run_s0_decode(
    bin_path: str | Path, meta: dict, token_id: int
) -> tuple[np.ndarray, float]:
    graph_name = "token_ar1_cl512_1_of_4"
    start = time.time()
    outputs = qnn_runner.run_context_binary(
        bin_path=bin_path,
        inputs={"input_ids": np.asarray([[token_id]], dtype=np.int32)},
        output_names=["embedding"],
        output_shapes={"embedding": (1, 1, 2048)},
        output_dtypes={"embedding": np.uint16},
        backend="htp",
        timeout_sec=120,
        graph_index=find_graph_index(meta, graph_name),
        num_graphs=2,
    )
    return outputs["embedding"], time.time() - start


def run_stage_prompt(
    bin_path: str | Path,
    meta: dict,
    prompt_graph_name: str,
    boundary_input_name: str,
    boundary_input_raw_u16: np.ndarray,
    boundary_output_name: str,
    boundary_output_shape: tuple[int, ...],
    layer_indices: list[int],
    num_real_tokens: int,
    has_steering: bool = False,
) -> dict:
    graph_index = find_graph_index(meta, prompt_graph_name)
    io = graph_io(meta, prompt_graph_name)
    inputs: dict[str, np.ndarray] = {
        boundary_input_name: boundary_input_raw_u16
    }

    padding_size = SEQUENCE_LENGTH - num_real_tokens
    positions = [0] * padding_size + list(range(SEQUENCE_LENGTH - padding_size))
    position_ids = torch.tensor(positions, dtype=torch.long).reshape(
        1, SEQUENCE_LENGTH
    )
    cos, sin = position_cos_sin(position_ids)
    cos_scale, cos_offset = scale_offset(io["graphInputs"], "position_ids_cos")
    sin_scale, sin_offset = scale_offset(io["graphInputs"], "position_ids_sin")
    inputs["position_ids_cos"] = quantize_u16(cos, cos_scale, cos_offset)
    inputs["position_ids_sin"] = quantize_u16(sin, sin_scale, sin_offset)
    mask = causal_attention_mask(num_real_tokens, SEQUENCE_LENGTH)
    mask_scale, mask_offset = scale_offset(io["graphInputs"], "attention_mask")
    inputs["attention_mask"] = quantize_u16(mask, mask_scale, mask_offset)

    past_key_in: dict[int, np.ndarray] = {}
    past_value_in: dict[int, np.ndarray] = {}
    for layer in layer_indices:
        _, key_offset = scale_offset(io["graphInputs"], f"past_key_{layer}_in")
        _, value_offset = scale_offset(
            io["graphInputs"], f"past_value_{layer}_in"
        )
        if key_offset != -128 or value_offset != -128:
            raise RuntimeError("unexpected Qwen3 KV zero point")
        past_key_in[layer] = np.full(
            (8, 1, 128, PAST_LEN_PROMPT), 128, dtype=np.uint8
        )
        past_value_in[layer] = np.full(
            (8, 1, PAST_LEN_PROMPT, 128), 128, dtype=np.uint8
        )
        inputs[f"past_key_{layer}_in"] = past_key_in[layer]
        inputs[f"past_value_{layer}_in"] = past_value_in[layer]

    if has_steering:
        alpha_scale, alpha_offset = scale_offset(io["graphInputs"], "alpha")
        vector_scale, vector_offset = scale_offset(
            io["graphInputs"], "steering_vector"
        )
        inputs["alpha"] = quantize_u16(
            np.zeros((1,), dtype=np.float32), alpha_scale, alpha_offset
        )
        inputs["steering_vector"] = quantize_u16(
            np.zeros((1, 1, 2048), dtype=np.float32),
            vector_scale,
            vector_offset,
        )

    output_names = [boundary_output_name] + [
        name
        for layer in layer_indices
        for name in (f"past_key_{layer}_out", f"past_value_{layer}_out")
    ]
    output_shapes = {boundary_output_name: boundary_output_shape}
    output_dtypes = {boundary_output_name: np.uint16}
    for layer in layer_indices:
        output_shapes[f"past_key_{layer}_out"] = (8, 1, 128, 128)
        output_shapes[f"past_value_{layer}_out"] = (8, 1, 128, 128)
        output_dtypes[f"past_key_{layer}_out"] = np.uint8
        output_dtypes[f"past_value_{layer}_out"] = np.uint8

    start = time.time()
    outputs = qnn_runner.run_context_binary(
        bin_path=bin_path,
        inputs=inputs,
        output_names=output_names,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        backend="htp",
        timeout_sec=120,
        graph_index=graph_index,
        num_graphs=2,
    )
    output_scale, output_offset = scale_offset(
        io["graphOutputs"], boundary_output_name
    )
    return {
        "output_u16": outputs[boundary_output_name],
        "output_scale": output_scale,
        "output_offset": output_offset,
        "raw_outputs": outputs,
        "past_key_in": past_key_in,
        "past_value_in": past_value_in,
        "elapsed_sec": time.time() - start,
    }


def run_stage_decode(
    bin_path: str | Path,
    meta: dict,
    decode_graph_name: str,
    boundary_input_name: str,
    boundary_input_raw_u16: np.ndarray,
    boundary_output_name: str,
    boundary_output_shape: tuple[int, ...],
    kv_buffer,
    layer_indices: list[int],
    rope_position: int,
    num_real_tokens: int,
    has_steering: bool = False,
) -> dict:
    graph_index = find_graph_index(meta, decode_graph_name)
    io = graph_io(meta, decode_graph_name)
    inputs: dict[str, np.ndarray] = {
        boundary_input_name: boundary_input_raw_u16
    }
    position_ids = torch.tensor([[rope_position]], dtype=torch.long)
    cos, sin = position_cos_sin(position_ids)
    cos_scale, cos_offset = scale_offset(io["graphInputs"], "position_ids_cos")
    sin_scale, sin_offset = scale_offset(io["graphInputs"], "position_ids_sin")
    inputs["position_ids_cos"] = quantize_u16(cos, cos_scale, cos_offset)
    inputs["position_ids_sin"] = quantize_u16(sin, sin_scale, sin_offset)
    mask = causal_attention_mask(num_real_tokens, 1)
    mask_scale, mask_offset = scale_offset(io["graphInputs"], "attention_mask")
    inputs["attention_mask"] = quantize_u16(mask, mask_scale, mask_offset)

    for layer in layer_indices:
        key, value = kv_buffer.get_past_in(layer, PAST_LEN_DECODE)
        inputs[f"past_key_{layer}_in"] = key
        inputs[f"past_value_{layer}_in"] = value
    if has_steering:
        alpha_scale, alpha_offset = scale_offset(io["graphInputs"], "alpha")
        vector_scale, vector_offset = scale_offset(
            io["graphInputs"], "steering_vector"
        )
        inputs["alpha"] = quantize_u16(
            np.zeros((1,), dtype=np.float32), alpha_scale, alpha_offset
        )
        inputs["steering_vector"] = quantize_u16(
            np.zeros((1, 1, 2048), dtype=np.float32),
            vector_scale,
            vector_offset,
        )

    output_names = [boundary_output_name] + [
        name
        for layer in layer_indices
        for name in (f"past_key_{layer}_out", f"past_value_{layer}_out")
    ]
    output_shapes = {boundary_output_name: boundary_output_shape}
    output_dtypes = {boundary_output_name: np.uint16}
    for layer in layer_indices:
        output_shapes[f"past_key_{layer}_out"] = (8, 1, 128, 1)
        output_shapes[f"past_value_{layer}_out"] = (8, 1, 1, 128)
        output_dtypes[f"past_key_{layer}_out"] = np.uint8
        output_dtypes[f"past_value_{layer}_out"] = np.uint8

    start = time.time()
    outputs = qnn_runner.run_context_binary(
        bin_path=bin_path,
        inputs=inputs,
        output_names=output_names,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        backend="htp",
        timeout_sec=120,
        graph_index=graph_index,
        num_graphs=2,
    )
    for layer in layer_indices:
        kv_buffer.update(
            layer,
            outputs[f"past_key_{layer}_out"],
            outputs[f"past_value_{layer}_out"],
        )
    output_scale, output_offset = scale_offset(
        io["graphOutputs"], boundary_output_name
    )
    return {
        "output_u16": outputs[boundary_output_name],
        "output_scale": output_scale,
        "output_offset": output_offset,
        "elapsed_sec": time.time() - start,
    }

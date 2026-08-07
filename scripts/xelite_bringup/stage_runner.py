"""Physical QNN execution helpers for the Qwen3-1.7B X Elite split pipeline.

Shared by every stage (S0-S3): per-tensor quantization lookup from a
context binary's own metadata (never assumed/hardcoded -- confirmed this
differs even between adjacent stages, e.g. KV tensor naming), RoPE cos/sin
via the real `transformers.LlamaRotaryEmbedding` class (Qualcomm's own
`qai_hub_models` RopeEmbedding wraps this same class; importing that package
directly pulls in an unpublished private dependency, so this reuses the
underlying third-party primitive instead of hand-deriving RoPE), and the
real `AttentionMaskConverter` for causal mask construction.

See docs/results/qwen3_1_7b_xelite_physical_bringup.md for the full
derivation and physical evidence behind every formula here.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from dragon_nest.runtime import qnn_runner

CONTEXT_LENGTH = 512
SEQUENCE_LENGTH = 128
PAST_LEN_PROMPT = CONTEXT_LENGTH - SEQUENCE_LENGTH  # 384
PAST_LEN_DECODE = CONTEXT_LENGTH - 1  # 511
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# Context-binary metadata
# ---------------------------------------------------------------------------

def ensure_binary_info(bin_path: str, cache_json_path: Path) -> dict:
    """Return the context binary's `qnn-context-binary-utility.exe --json_file`
    dump, generating it into `cache_json_path` if not already cached there.
    This is the metadata source of truth for graph names/order, tensor
    names, shapes, dtypes, and per-tensor quantization -- never assumed from
    another stage or from the pipeline manifest's declared schema, both of
    which have been physically confirmed wrong in places (see the bring-up
    doc: stage 0/1's graph order is reversed relative to stage 1/2/3's, and
    the manifest originally declared float32 for tensors that are actually
    quantized fixed-point)."""
    if not cache_json_path.exists():
        cache_json_path.parent.mkdir(parents=True, exist_ok=True)
        utility = qnn_runner.BIN_DIR / "qnn-context-binary-utility.exe"
        if not utility.exists():
            raise FileNotFoundError(
                f"qnn-context-binary-utility.exe not found at {utility}; set QAIRT_ROOT"
            )
        result = subprocess.run(
            [str(utility), "--context_binary", str(bin_path), "--json_file", str(cache_json_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0 or not cache_json_path.exists():
            raise RuntimeError(
                f"qnn-context-binary-utility.exe failed for {bin_path} "
                f"(exit {result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    return json.loads(cache_json_path.read_text())


def graph_io(meta: dict, graph_name: str) -> dict:
    for g in meta["info"]["graphs"]:
        if g["info"]["graphName"] == graph_name:
            return g["info"]
    raise KeyError(f"graph {graph_name!r} not found; available: "
                    f"{[g['info']['graphName'] for g in meta['info']['graphs']]}")


def find_graph_index(meta: dict, graph_name: str) -> int:
    for i, g in enumerate(meta["info"]["graphs"]):
        if g["info"]["graphName"] == graph_name:
            return i
    raise KeyError(graph_name)


def scale_offset(tensor_info_list: list[dict], name: str) -> tuple[float, int]:
    for t in tensor_info_list:
        ti = t["info"]
        if ti["name"] == name:
            so = ti["quantizeParams"]["scaleOffset"]
            return so["scale"], so["offset"]
    raise KeyError(name)


def quantize_u16(real: np.ndarray, scale: float, offset: int) -> np.ndarray:
    raw = np.round(real / scale - offset)
    return np.clip(raw, 0, 65535).astype(np.uint16)


def dequantize(raw: np.ndarray, scale: float, offset: int) -> np.ndarray:
    return (raw.astype(np.float32) + offset) * scale


# ---------------------------------------------------------------------------
# RoPE / attention mask (real transformers primitives, not hand-derived)
# ---------------------------------------------------------------------------

_rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None


def rope_cos_sin_full() -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for every position 0..CONTEXT_LENGTH-1 once, using
    the base model's real rope_theta/head_dim fetched from its public
    Hugging Face config (network access required, no proprietary data)."""
    global _rope_cache
    if _rope_cache is None:
        config = AutoConfig.from_pretrained(BASE_MODEL_ID)
        rope = LlamaRotaryEmbedding(config=config)
        dummy_x = torch.tensor([1.0])
        position_ids = torch.arange(CONTEXT_LENGTH).view(1, -1)
        cos_full, sin_full = rope.forward(dummy_x, position_ids)
        emb_size = cos_full.size(-1) // 2  # head_dim // 2
        cos_full = cos_full[:, :, :emb_size].unsqueeze(0)  # [1,1,CONTEXT_LENGTH,emb_size]
        sin_full = sin_full[:, :, :emb_size].unsqueeze(0)
        _rope_cache = (cos_full, sin_full)
    return _rope_cache


def position_cos_sin(position_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    cos_full, sin_full = rope_cos_sin_full()
    cos = cos_full[0, 0, :, :][position_ids].unsqueeze(1)
    sin = sin_full[0, 0, :, :][position_ids].unsqueeze(1)
    return cos.numpy(), sin.numpy()


def causal_attention_mask(num_real_tokens: int, query_length: int) -> np.ndarray:
    """`num_real_tokens` is the count of real (non-padding) tokens in the
    512-position context frame -- for a prompt call this is the prompt's own
    real-token count; for a decode call it's that count plus every token
    generated so far, INCLUDING the one being processed in this call."""
    attention_mask = torch.zeros((1, CONTEXT_LENGTH))
    attention_mask[:, -num_real_tokens:] = 1.0
    converter = AttentionMaskConverter(True)
    mask4d = converter.to_4d(
        attention_mask, query_length=query_length, key_value_length=CONTEXT_LENGTH, dtype=torch.float32,
    )
    # -100 clip: confirmed (not the generic sample-input helper's -50) from
    # this pipeline's actual attention_mask quantization, which is an exact
    # linear map of [-100, 0] onto the uint16 range. See the bring-up doc.
    return mask4d.clip(-100, 0).numpy()


# ---------------------------------------------------------------------------
# S0 (embedding stage): no RoPE/mask/KV dependency
# ---------------------------------------------------------------------------

def run_s0_prompt(bin_path: str, meta: dict, input_ids: np.ndarray) -> tuple[np.ndarray, float]:
    graph_name = "prompt_ar128_cl512_1_of_4"
    idx = find_graph_index(meta, graph_name)
    t0 = time.time()
    outputs = qnn_runner.run_context_binary(
        bin_path=bin_path, inputs={"input_ids": input_ids},
        output_names=["embedding"], output_shapes={"embedding": (1, SEQUENCE_LENGTH, 2048)},
        output_dtypes={"embedding": np.uint16}, backend="htp", timeout_sec=120,
        graph_index=idx, num_graphs=2,
    )
    return outputs["embedding"], time.time() - t0


def run_s0_decode(bin_path: str, meta: dict, token_id: int) -> tuple[np.ndarray, float]:
    graph_name = "token_ar1_cl512_1_of_4"
    idx = find_graph_index(meta, graph_name)
    input_ids = np.array([[token_id]], dtype=np.int32)
    t0 = time.time()
    outputs = qnn_runner.run_context_binary(
        bin_path=bin_path, inputs={"input_ids": input_ids},
        output_names=["embedding"], output_shapes={"embedding": (1, 1, 2048)},
        output_dtypes={"embedding": np.uint16}, backend="htp", timeout_sec=120,
        graph_index=idx, num_graphs=2,
    )
    return outputs["embedding"], time.time() - t0


# ---------------------------------------------------------------------------
# S1/S2/S3 (transformer-layer stages)
# ---------------------------------------------------------------------------

def run_stage_prompt(
    bin_path: str,
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
    idx = find_graph_index(meta, prompt_graph_name)
    io = graph_io(meta, prompt_graph_name)

    inputs: dict[str, np.ndarray] = {boundary_input_name: boundary_input_raw_u16}

    padding_size = SEQUENCE_LENGTH - num_real_tokens
    position_ids_list = [0] * padding_size + list(range(SEQUENCE_LENGTH - padding_size))
    position_ids = torch.tensor(position_ids_list, dtype=torch.long).reshape(1, SEQUENCE_LENGTH)
    cos, sin = position_cos_sin(position_ids)
    cos_scale, cos_offset = scale_offset(io["graphInputs"], "position_ids_cos")
    sin_scale, sin_offset = scale_offset(io["graphInputs"], "position_ids_sin")
    inputs["position_ids_cos"] = quantize_u16(cos, cos_scale, cos_offset)
    inputs["position_ids_sin"] = quantize_u16(sin, sin_scale, sin_offset)

    mask = causal_attention_mask(num_real_tokens, query_length=SEQUENCE_LENGTH)
    mask_scale, mask_offset = scale_offset(io["graphInputs"], "attention_mask")
    inputs["attention_mask"] = quantize_u16(mask, mask_scale, mask_offset)

    past_key_in, past_value_in = {}, {}
    for layer in layer_indices:
        k_scale, k_offset = scale_offset(io["graphInputs"], f"past_key_{layer}_in")
        v_scale, v_offset = scale_offset(io["graphInputs"], f"past_value_{layer}_in")
        assert k_offset == -128 and v_offset == -128, "unexpected KV zero-point"
        past_key_in[layer] = np.full((8, 1, 128, PAST_LEN_PROMPT), 128, dtype=np.uint8)
        past_value_in[layer] = np.full((8, 1, PAST_LEN_PROMPT, 128), 128, dtype=np.uint8)
        inputs[f"past_key_{layer}_in"] = past_key_in[layer]
        inputs[f"past_value_{layer}_in"] = past_value_in[layer]

    if has_steering:
        alpha_scale, alpha_offset = scale_offset(io["graphInputs"], "alpha")
        sv_scale, sv_offset = scale_offset(io["graphInputs"], "steering_vector")
        inputs["alpha"] = quantize_u16(np.zeros((1,), dtype=np.float32), alpha_scale, alpha_offset)
        inputs["steering_vector"] = quantize_u16(np.zeros((1, 1, 2048), dtype=np.float32), sv_scale, sv_offset)

    output_names = [boundary_output_name] + [
        n for layer in layer_indices for n in (f"past_key_{layer}_out", f"past_value_{layer}_out")
    ]
    output_shapes = {boundary_output_name: boundary_output_shape}
    output_dtypes = {boundary_output_name: np.uint16}
    for layer in layer_indices:
        output_shapes[f"past_key_{layer}_out"] = (8, 1, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        output_shapes[f"past_value_{layer}_out"] = (8, 1, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        output_dtypes[f"past_key_{layer}_out"] = np.uint8
        output_dtypes[f"past_value_{layer}_out"] = np.uint8

    t0 = time.time()
    outputs = qnn_runner.run_context_binary(
        bin_path=bin_path, inputs=inputs, output_names=output_names,
        output_shapes=output_shapes, output_dtypes=output_dtypes,
        backend="htp", timeout_sec=120, graph_index=idx, num_graphs=2,
    )
    elapsed = time.time() - t0

    out_u16 = outputs[boundary_output_name]
    out_scale, out_offset = scale_offset(io["graphOutputs"], boundary_output_name)

    return {
        "output_u16": out_u16,
        "output_scale": out_scale,
        "output_offset": out_offset,
        "raw_outputs": outputs,
        "past_key_in": past_key_in,
        "past_value_in": past_value_in,
        "elapsed_sec": elapsed,
    }


def run_stage_decode(
    bin_path: str,
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
    idx = find_graph_index(meta, decode_graph_name)
    io = graph_io(meta, decode_graph_name)

    inputs: dict[str, np.ndarray] = {boundary_input_name: boundary_input_raw_u16}

    position_ids = torch.tensor([[rope_position]], dtype=torch.long)
    cos, sin = position_cos_sin(position_ids)
    cos_scale, cos_offset = scale_offset(io["graphInputs"], "position_ids_cos")
    sin_scale, sin_offset = scale_offset(io["graphInputs"], "position_ids_sin")
    inputs["position_ids_cos"] = quantize_u16(cos, cos_scale, cos_offset)
    inputs["position_ids_sin"] = quantize_u16(sin, sin_scale, sin_offset)

    mask = causal_attention_mask(num_real_tokens, query_length=1)
    mask_scale, mask_offset = scale_offset(io["graphInputs"], "attention_mask")
    inputs["attention_mask"] = quantize_u16(mask, mask_scale, mask_offset)

    for layer in layer_indices:
        past_key_in, past_value_in = kv_buffer.get_past_in(layer, past_len=PAST_LEN_DECODE)
        inputs[f"past_key_{layer}_in"] = past_key_in
        inputs[f"past_value_{layer}_in"] = past_value_in

    if has_steering:
        alpha_scale, alpha_offset = scale_offset(io["graphInputs"], "alpha")
        sv_scale, sv_offset = scale_offset(io["graphInputs"], "steering_vector")
        inputs["alpha"] = quantize_u16(np.zeros((1,), dtype=np.float32), alpha_scale, alpha_offset)
        inputs["steering_vector"] = quantize_u16(np.zeros((1, 1, 2048), dtype=np.float32), sv_scale, sv_offset)

    output_names = [boundary_output_name] + [
        n for layer in layer_indices for n in (f"past_key_{layer}_out", f"past_value_{layer}_out")
    ]
    output_shapes = {boundary_output_name: boundary_output_shape}
    output_dtypes = {boundary_output_name: np.uint16}
    for layer in layer_indices:
        output_shapes[f"past_key_{layer}_out"] = (8, 1, 128, 1)
        output_shapes[f"past_value_{layer}_out"] = (8, 1, 1, 128)
        output_dtypes[f"past_key_{layer}_out"] = np.uint8
        output_dtypes[f"past_value_{layer}_out"] = np.uint8

    t0 = time.time()
    outputs = qnn_runner.run_context_binary(
        bin_path=bin_path, inputs=inputs, output_names=output_names,
        output_shapes=output_shapes, output_dtypes=output_dtypes,
        backend="htp", timeout_sec=120, graph_index=idx, num_graphs=2,
    )
    elapsed = time.time() - t0

    for layer in layer_indices:
        kv_buffer.update(layer, outputs[f"past_key_{layer}_out"], outputs[f"past_value_{layer}_out"])

    out_u16 = outputs[boundary_output_name]
    out_scale, out_offset = scale_offset(io["graphOutputs"], boundary_output_name)

    return {"output_u16": out_u16, "output_scale": out_scale, "output_offset": out_offset, "elapsed_sec": elapsed}

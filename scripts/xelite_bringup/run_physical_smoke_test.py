"""Physical Qwen3-1.7B X Elite smoke test: one real prefill (S0->S1->S2->S3)
followed by a real multi-token autoregressive decode loop (S0->S1->S2->S3
decode, persistent per-stage KV state, advancing RoPE position and growing
attention mask every step), entirely on Hexagon HTP via qnn-net-run.exe.
No mock execution, no CPU fallback, no placeholder boundary tensors.

This is the exact reconstruction of the physical proof recorded in
docs/results/qwen3_1_7b_xelite_physical_bringup.md -- run it to reproduce
that result on another X Elite host with the same external artifacts.

Required environment variables (see this directory's README.md):
    QAIRT_ROOT
    QWEN3_1_7B_S0_XELITE_QNN
    QWEN3_1_7B_S1_XELITE_QNN
    QWEN3_1_7B_S2_XELITE_QNN
    QWEN3_1_7B_S3_XELITE_QNN

Optional:
    DRAGONNEST_XELITE_BRINGUP_SCRATCH  (defaults to a temp subdirectory;
        caches each binary's qnn-context-binary-utility.exe metadata dump
        so repeat runs don't regenerate it)
    DRAGONNEST_XELITE_DECODE_STEPS  (defaults to 8)
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import psutil
from transformers import AutoTokenizer

import stage_runner as sr
from kv_buffer import StageKVBuffer

REQUIRED_ENV = [
    "QAIRT_ROOT",
    "QWEN3_1_7B_S0_XELITE_QNN",
    "QWEN3_1_7B_S1_XELITE_QNN",
    "QWEN3_1_7B_S2_XELITE_QNN",
    "QWEN3_1_7B_S3_XELITE_QNN",
]

NUM_DECODE_STEPS = int(os.environ.get("DRAGONNEST_XELITE_DECODE_STEPS", "8"))
EOS_IDS = {151645, 151643}  # <|im_end|>, <|endoftext|>

DEFAULT_PROMPT_SYSTEM = "You are a helpful AI assistant."
DEFAULT_PROMPT_USER = "What is gravity? Keep the answer under ten words."

latencies: list[dict] = []


def log_latency(stage: str, step: str, seconds: float) -> None:
    latencies.append({"stage": stage, "step": step, "seconds": seconds})
    print(f"  [latency] {stage} {step}: {seconds:.3f}s")


def scratch_dir() -> Path:
    d = Path(os.environ.get(
        "DRAGONNEST_XELITE_BRINGUP_SCRATCH",
        str(Path(tempfile.gettempdir()) / "dragon_nest" / "xelite_bringup"),
    ))
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_prompt_input_ids(tok) -> tuple[np.ndarray, int]:
    messages = [
        {"role": "system", "content": DEFAULT_PROMPT_SYSTEM},
        {"role": "user", "content": DEFAULT_PROMPT_USER},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="np", padding="max_length", max_length=sr.CONTEXT_LENGTH)
    input_ids_full = enc["input_ids"].astype(np.int32)
    num_tokens = int(min(enc["attention_mask"].sum(), sr.SEQUENCE_LENGTH))
    input_ids = input_ids_full[:, -sr.SEQUENCE_LENGTH:]
    return input_ids, num_tokens


def main() -> None:
    missing = [v for v in REQUIRED_ENV if v not in os.environ]
    if missing:
        raise SystemExit(f"Missing required environment variable(s): {missing}. See README.md.")

    scratch = scratch_dir()
    s0_bin = os.environ["QWEN3_1_7B_S0_XELITE_QNN"]
    s1_bin = os.environ["QWEN3_1_7B_S1_XELITE_QNN"]
    s2_bin = os.environ["QWEN3_1_7B_S2_XELITE_QNN"]
    s3_bin = os.environ["QWEN3_1_7B_S3_XELITE_QNN"]

    print("Fetching context-binary metadata (qnn-context-binary-utility.exe; cached under scratch dir)...")
    s0_meta = sr.ensure_binary_info(s0_bin, scratch / "s0_binary_info.json")
    s1_meta = sr.ensure_binary_info(s1_bin, scratch / "s1_binary_info.json")
    s2_meta = sr.ensure_binary_info(s2_bin, scratch / "s2_binary_info.json")
    s3_meta = sr.ensure_binary_info(s3_bin, scratch / "s3_binary_info.json")

    print(f"Loading tokenizer for {sr.BASE_MODEL_ID} (public Hugging Face repo, network access required)...")
    tok = AutoTokenizer.from_pretrained(sr.BASE_MODEL_ID, is_fast=False)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.truncation_side = "left"

    input_ids, num_tokens = build_prompt_input_ids(tok)
    print(f"Prompt tokenized: {num_tokens} real tokens (of {sr.SEQUENCE_LENGTH}-token window)")

    print("\n=== PREFILL ===")
    s0_embedding, t = sr.run_s0_prompt(s0_bin, s0_meta, input_ids)
    log_latency("S0", "prompt", t)

    s1_result = sr.run_stage_prompt(
        s1_bin, s1_meta, "prompt_ar128_cl512_2_of_4",
        boundary_input_name="embedding", boundary_input_raw_u16=s0_embedding,
        boundary_output_name="add_21844", boundary_output_shape=(1, 128, 2048),
        layer_indices=list(range(0, 10)), num_real_tokens=num_tokens, has_steering=True,
    )
    log_latency("S1", "prompt", s1_result["elapsed_sec"])

    s2_result = sr.run_stage_prompt(
        s2_bin, s2_meta, "prompt_ar128_cl512_3_of_4",
        boundary_input_name="add_21844", boundary_input_raw_u16=s1_result["output_u16"],
        boundary_output_name="add_42314", boundary_output_shape=(1, 128, 2048),
        layer_indices=list(range(10, 20)), num_real_tokens=num_tokens, has_steering=False,
    )
    log_latency("S2", "prompt", s2_result["elapsed_sec"])

    s3_result = sr.run_stage_prompt(
        s3_bin, s3_meta, "prompt_ar128_cl512_4_of_4",
        boundary_input_name="add_42314", boundary_input_raw_u16=s2_result["output_u16"],
        boundary_output_name="logits", boundary_output_shape=(1, 128, 151936),
        layer_indices=list(range(20, 28)), num_real_tokens=num_tokens, has_steering=False,
    )
    log_latency("S3", "prompt", s3_result["elapsed_sec"])

    logits = sr.dequantize(s3_result["output_u16"][0, -1, :], s3_result["output_scale"], s3_result["output_offset"])
    first_token = int(np.argmax(logits))
    print(f"\nPrefill top-1 token: id={first_token}, decoded={tok.decode([first_token])!r}")

    kv_s1 = StageKVBuffer(list(range(0, 10)))
    kv_s2 = StageKVBuffer(list(range(10, 20)))
    kv_s3 = StageKVBuffer(list(range(20, 28)))
    for layer, (result, kv) in [(l, (s1_result, kv_s1)) for l in range(0, 10)] + \
                                 [(l, (s2_result, kv_s2)) for l in range(10, 20)] + \
                                 [(l, (s3_result, kv_s3)) for l in range(20, 28)]:
        kv.seed_from_prompt(layer, result["past_key_in"][layer], result["raw_outputs"][f"past_key_{layer}_out"], True)
        kv.seed_from_prompt(layer, result["past_value_in"][layer], result["raw_outputs"][f"past_value_{layer}_out"], False)
    print("Seeded S1/S2/S3 persistent KV buffers from real prefill state.")

    generated_ids: list[int] = [first_token]
    current_token = first_token

    print(f"\n=== DECODE LOOP (up to {NUM_DECODE_STEPS} tokens) ===")
    for step in range(NUM_DECODE_STEPS):
        rope_position = num_tokens + step
        num_real_tokens_so_far = num_tokens + step + 1
        print(f"\n-- decode step {step + 1}: token_id={current_token} "
              f"({tok.decode([current_token])!r}), rope_position={rope_position} --")

        s0_dec_embedding, t = sr.run_s0_decode(s0_bin, s0_meta, current_token)
        log_latency("S0", f"decode[{step + 1}]", t)

        s1_dec = sr.run_stage_decode(
            s1_bin, s1_meta, "token_ar1_cl512_2_of_4",
            boundary_input_name="embedding", boundary_input_raw_u16=s0_dec_embedding,
            boundary_output_name="add_21844", boundary_output_shape=(1, 1, 2048),
            kv_buffer=kv_s1, layer_indices=list(range(0, 10)),
            rope_position=rope_position, num_real_tokens=num_real_tokens_so_far, has_steering=True,
        )
        log_latency("S1", f"decode[{step + 1}]", s1_dec["elapsed_sec"])

        s2_dec = sr.run_stage_decode(
            s2_bin, s2_meta, "token_ar1_cl512_3_of_4",
            boundary_input_name="add_21844", boundary_input_raw_u16=s1_dec["output_u16"],
            boundary_output_name="add_42314", boundary_output_shape=(1, 1, 2048),
            kv_buffer=kv_s2, layer_indices=list(range(10, 20)),
            rope_position=rope_position, num_real_tokens=num_real_tokens_so_far, has_steering=False,
        )
        log_latency("S2", f"decode[{step + 1}]", s2_dec["elapsed_sec"])

        s3_dec = sr.run_stage_decode(
            s3_bin, s3_meta, "token_ar1_cl512_4_of_4",
            boundary_input_name="add_42314", boundary_input_raw_u16=s2_dec["output_u16"],
            boundary_output_name="logits", boundary_output_shape=(1, 1, 151936),
            kv_buffer=kv_s3, layer_indices=list(range(20, 28)),
            rope_position=rope_position, num_real_tokens=num_real_tokens_so_far, has_steering=False,
        )
        log_latency("S3", f"decode[{step + 1}]", s3_dec["elapsed_sec"])

        step_logits = sr.dequantize(s3_dec["output_u16"].reshape(-1), s3_dec["output_scale"], s3_dec["output_offset"])
        next_token = int(np.argmax(step_logits))
        print(f"  -> next token id={next_token}, decoded={tok.decode([next_token])!r}")

        generated_ids.append(next_token)
        current_token = next_token
        if next_token in EOS_IDS:
            print(f"  EOS token {next_token} generated -- stopping early.")
            break

    print("\n=== RESULTS ===")
    print(f"Generated token IDs: {generated_ids}")
    decoded_text = tok.decode(generated_ids)
    print(f"Decoded text: {decoded_text!r}")
    print("\nIncremental decode:")
    for i in range(1, len(generated_ids) + 1):
        print(f"  after {i} token(s): {tok.decode(generated_ids[:i])!r}")

    print("\n=== LATENCY SUMMARY ===")
    for entry in latencies:
        print(f"  {entry['stage']:>3} {entry['step']:>14}: {entry['seconds']:.3f}s")
    total = sum(e["seconds"] for e in latencies)
    print(f"  TOTAL: {total:.3f}s across {len(latencies)} physical HTP calls")

    print("\n=== CLEANUP ===")
    kv_s1.release()
    kv_s2.release()
    kv_s3.release()
    print("Released all stage-local KV buffer state.")
    qnn_procs = [p for p in psutil.process_iter(["name"]) if p.info["name"] and "qnn" in p.info["name"].lower()]
    print(f"Orphaned qnn-net-run.exe processes still running: {len(qnn_procs)}")


if __name__ == "__main__":
    main()

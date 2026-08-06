"""Run Whisper-base's encoder.onnx + decoder.onnx locally via ONNX Runtime's
QNN execution provider (through onnxrt_runner.py's ARM64 worker subprocess),
reusing qai_hub_models' HfWhisperApp for audio feature extraction (HF
WhisperFeatureExtractor) and token->text decoding (HF WhisperTokenizer).

This is a different model from whisper_pipeline.py (whisper-tiny, 4 decoder
blocks, hybrid NPU-encoder/CPU-decoder architecture) -- this is whisper-base
(6 decoder blocks) and, unlike whisper-tiny's decoder.bin (which fails to
load on any Hexagon protection domain), this decoder.onnx is an EPContext-
wrapped QNN context binary that loads and runs on-NPU successfully -- so both
encoder and decoder run on-NPU here, no CPU-decoder fallback needed.

encoder.onnx/decoder.onnx are thin EPContext wrappers around precompiled QNN
context binaries (encoder_qairt_context.bin / decoder_qairt_context.bin, which
must sit alongside the .onnx files -- ORT resolves them internally). All I/O
is float16 (except input_ids/position_ids, int32) per metadata.json -- casts
happen only at the run_onnx() boundary, matching whisper_pipeline.py's
existing convention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from onnxrt_runner import run_onnx  # noqa: E402

from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp  # noqa: E402

WHISPER_VERSION = "openai/whisper-base"

WHISPER_ONNX_DIR = Path(
    r"C:\Users\harisury\Downloads\ONNX"
    r"\whisper_base-precompiled_qnn_onnx-float-qualcomm_snapdragon_x_elite"
    r"\whisper_base-precompiled_qnn_onnx-float-qualcomm_snapdragon_x_elite"
)
ENCODER_ONNX = WHISPER_ONNX_DIR / "encoder.onnx"
DECODER_ONNX = WHISPER_ONNX_DIR / "decoder.onnx"
METADATA = json.loads((WHISPER_ONNX_DIR / "metadata.json").read_text())

NUM_DECODER_BLOCKS = sum(
    1
    for name in METADATA["model_files"]["decoder.onnx"]["inputs"]
    if name.startswith("k_cache_self_") and name.endswith("_in")
)

_NP_DTYPES = {"int32": np.int32, "float32": np.float32, "float16": np.float16}


def _tensor_spec(file_name: str, direction: str, tensor_name: str) -> dict:
    return METADATA["model_files"][file_name][direction][tensor_name]


def _dtype(file_name: str, direction: str, tensor_name: str) -> np.dtype:
    return _NP_DTYPES[_tensor_spec(file_name, direction, tensor_name)["dtype"]]


class EncoderModule:
    """Callable matching HfWhisperEncoder.forward, backed by encoder.onnx.

    Returns a flat tuple (k_cross_0, v_cross_0, ..., k_cross_5, v_cross_5) --
    HfWhisperApp accepts this shape directly (see app.py's kv_cache_cross
    wrapping logic).
    """

    FILE = "encoder.onnx"
    OUTPUT_NAMES = [
        f"{prefix}_cache_cross_{i}"
        for i in range(NUM_DECODER_BLOCKS)
        for prefix in ("k", "v")
    ]

    def __init__(self, backend: str = "htp") -> None:
        self.backend = backend

    def __call__(self, input_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        inputs = {
            "input_features": input_features.detach().numpy().astype(
                _dtype(self.FILE, "inputs", "input_features")
            )
        }
        outputs = run_onnx(
            ENCODER_ONNX,
            inputs=inputs,
            output_names=self.OUTPUT_NAMES,
            backend=self.backend,
        )
        return tuple(
            torch.from_numpy(outputs[name].astype(np.float32))
            for name in self.OUTPUT_NAMES
        )


class DecoderModule:
    """Callable matching HfWhisperApp's flattened decoder call convention:

        decoder(input_ids, attention_mask,
                *flattened_kv_cache_self, *flattened_kv_cache_cross,
                position_ids)
        -> (logits, k_self_0, v_self_0, ..., k_self_5, v_self_5)

    backed by decoder.onnx. Unlike whisper_pipeline.py's CPU decoder
    (HfWhisperDecoder), this always runs on the NPU via run_onnx() -- no
    torch.nn.Module weights loaded.
    """

    FILE = "decoder.onnx"
    OUTPUT_NAMES = [
        "logits",
        *[
            f"{prefix}_cache_self_{i}_out"
            for i in range(NUM_DECODER_BLOCKS)
            for prefix in ("k", "v")
        ],
    ]

    def __init__(self, backend: str = "htp") -> None:
        self.backend = backend

    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        input_ids, attention_mask = args[0], args[1]
        kv_self_flat = args[2 : 2 + 2 * NUM_DECODER_BLOCKS]
        kv_cross_flat = args[
            2 + 2 * NUM_DECODER_BLOCKS : 2 + 4 * NUM_DECODER_BLOCKS
        ]
        position_ids = args[2 + 4 * NUM_DECODER_BLOCKS]

        inputs: dict[str, np.ndarray] = {
            "input_ids": input_ids.detach().numpy().astype(
                _dtype(self.FILE, "inputs", "input_ids")
            ),
            "attention_mask": attention_mask.detach().numpy().astype(
                _dtype(self.FILE, "inputs", "attention_mask")
            ),
            "position_ids": position_ids.detach().numpy().astype(
                _dtype(self.FILE, "inputs", "position_ids")
            ),
        }
        for i in range(NUM_DECODER_BLOCKS):
            k_name, v_name = f"k_cache_self_{i}_in", f"v_cache_self_{i}_in"
            inputs[k_name] = kv_self_flat[2 * i].detach().numpy().astype(
                _dtype(self.FILE, "inputs", k_name)
            )
            inputs[v_name] = kv_self_flat[2 * i + 1].detach().numpy().astype(
                _dtype(self.FILE, "inputs", v_name)
            )
            kc_name, vc_name = f"k_cache_cross_{i}", f"v_cache_cross_{i}"
            inputs[kc_name] = kv_cross_flat[2 * i].detach().numpy().astype(
                _dtype(self.FILE, "inputs", kc_name)
            )
            inputs[vc_name] = kv_cross_flat[2 * i + 1].detach().numpy().astype(
                _dtype(self.FILE, "inputs", vc_name)
            )

        outputs = run_onnx(
            DECODER_ONNX,
            inputs=inputs,
            output_names=self.OUTPUT_NAMES,
            backend=self.backend,
        )
        return tuple(
            torch.from_numpy(outputs[name].astype(np.float32))
            for name in self.OUTPUT_NAMES
        )


def build_app(backend: str = "htp") -> HfWhisperApp:
    return HfWhisperApp(
        encoder=EncoderModule(backend=backend),
        decoder=DecoderModule(backend=backend),
        hf_model_id=WHISPER_VERSION,
    )


def run(audio_path: str, backend: str = "htp") -> str:
    app = build_app(backend=backend)
    audio, sample_rate = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    return app.transcribe(audio, sample_rate)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("--backend", choices=["cpu", "htp"], default="htp")
    args = parser.parse_args()

    transcription = run(args.audio_path, backend=args.backend)
    print(f"Transcription: {transcription}")

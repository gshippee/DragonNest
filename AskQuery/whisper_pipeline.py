"""Run Whisper-tiny's encoder.bin locally via qnn-net-run on the Snapdragon
NPU (or CPU), reusing qai_hub_models' HfWhisperApp for audio feature
extraction (HF WhisperFeatureExtractor), the autoregressive KV-cache decode
loop, and token->text decoding (HF WhisperTokenizer). Only the encoder
forward pass is swapped for an NPU-backed callable.

The decoder runs the real PyTorch weights on CPU instead of decoder.bin.
decoder.bin's ~97MB context binary (~120MB in-memory estimate) fails to load
on this device's HTP backend on every protection domain (PD 0-3), with
`Skel failed to process context binary` / err 5005 / "Failed to find
available PD ... with context size estimate 120426752" -- reproduced both
through the full pipeline and via a direct isolated qnn-net-run invocation,
with and without --use_mmap. QAIRT's HTP backend docs describe a per-PD
heap that (unlike VTCM size) has no documented CLI/JSON/registry override,
and encoder.bin (~20MB) plus every MeloTTS binary (<30MB) load fine on the
same device/backend -- this points to a hard per-PD memory ceiling on this
device that decoder.bin's size exceeds, not a bug in this pipeline.

Unlike MeloTTS's flow/decoder, encoder.bin's I/O is plain float16 (not
int-quantized) -- no scale/zero_point dequantization needed, just a
float16<->float32 cast at the NPU boundary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from qnn_runner import run_context_binary  # noqa: E402

from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp  # noqa: E402
from qai_hub_models.models._shared.hf_whisper.model import HfWhisperDecoder  # noqa: E402

WHISPER_VERSION = "openai/whisper-tiny"

WHISPER_DIR = Path(
    r"C:\Users\harisury\Downloads\whisper_tiny-qnn_context_binary-float-qualcomm_snapdragon_x_elite"
    r"\whisper_tiny-qnn_context_binary-float-qualcomm_snapdragon_x_elite"
)
ENCODER_BIN = WHISPER_DIR / "encoder.bin"
METADATA = json.loads((WHISPER_DIR / "metadata.json").read_text())

NUM_DECODER_BLOCKS = 4

_NP_DTYPES = {"int32": np.int32, "float32": np.float32, "float16": np.float16}


def _tensor_spec(file_name: str, direction: str, tensor_name: str) -> dict:
    return METADATA["model_files"][file_name][direction][tensor_name]


def _shape(file_name: str, direction: str, tensor_name: str) -> tuple[int, ...]:
    return tuple(_tensor_spec(file_name, direction, tensor_name)["shape"])


def _dtype(file_name: str, direction: str, tensor_name: str) -> np.dtype:
    return _NP_DTYPES[_tensor_spec(file_name, direction, tensor_name)["dtype"]]


class EncoderModule:
    """Callable matching HfWhisperEncoder.forward, backed by encoder.bin.

    Returns a flat tuple (k_cross_0, v_cross_0, k_cross_1, v_cross_1, ...) --
    HfWhisperApp accepts this shape directly (see app.py's kv_cache_cross
    wrapping logic).
    """

    FILE = "encoder.bin"
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
        outputs = run_context_binary(
            ENCODER_BIN,
            inputs=inputs,
            output_names=self.OUTPUT_NAMES,
            output_shapes={
                name: _shape(self.FILE, "outputs", name) for name in self.OUTPUT_NAMES
            },
            output_dtypes={
                name: _dtype(self.FILE, "outputs", name) for name in self.OUTPUT_NAMES
            },
            backend=self.backend,
        )
        return tuple(
            torch.from_numpy(outputs[name].astype(np.float32))
            for name in self.OUTPUT_NAMES
        )


def build_app(backend: str = "htp") -> HfWhisperApp:
    return HfWhisperApp(
        encoder=EncoderModule(backend=backend),
        decoder=HfWhisperDecoder.from_pretrained(WHISPER_VERSION),
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

"""Run MeloTTS's encoder.bin + flow.bin + decoder.bin locally via qnn-net-run on
the Snapdragon NPU (or CPU), reusing qai_hub_models' MeloTTSApp for text
preprocessing (melo package: phones/tones/BERT features), generate_path(), and
chunked-decoder orchestration. The G2P/BERT context binaries (t5_encoder.bin,
t5_decoder.bin, bert_wrapper.bin) are unused here because MeloTTSApp's own
tts_to_file() never calls them -- text preprocessing is done by the melo
package on CPU instead."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import device_config  # noqa: E402
from qnn_runner import run_context_binary  # noqa: E402

from qai_hub_models.datasets.common_voice.voiceai_lang import TTSLanguage  # noqa: E402
from qai_hub_models.models._shared.melotts.app import MeloTTSApp  # noqa: E402
from qai_hub_models.models._shared.melotts.model import get_tts_object  # noqa: E402

MELOTTS_DIR = device_config.MELOTTS_DIR
ENCODER_BIN = MELOTTS_DIR / "encoder.bin"
FLOW_BIN = MELOTTS_DIR / "flow.bin"
DECODER_BIN = MELOTTS_DIR / "decoder.bin"
METADATA = json.loads((MELOTTS_DIR / "metadata.json").read_text())

_NP_DTYPES = {"int32": np.int32, "float32": np.float32, "uint16": np.uint16}


class TTSTextTooLongError(RuntimeError):
    """Raised when the encoder's predicted output length (y_lengths) exceeds
    UPSAMPLED_MAX_SEQ_LEN -- the model's fixed frame budget for the flow/
    decoder stages. Frames beyond that ceiling are never validly synthesized
    (the z tensor flow.bin produces is a fixed UPSAMPLED_MAX_SEQ_LEN-length
    buffer regardless of input), so trimming audio to y_lengths in that case
    plays back the decoder's zero-padded tail as constant static. Phone count
    alone doesn't reliably predict this -- content with heavy punctuation,
    numerals, or unusual words can make the model predict a much longer
    duration than phone count would suggest -- so this is checked directly
    against the encoder's real output rather than guessed in advance."""


def _tensor_spec(file_name: str, direction: str, tensor_name: str) -> dict:
    return METADATA["model_files"][file_name][direction][tensor_name]


def _shape(file_name: str, direction: str, tensor_name: str) -> tuple[int, ...]:
    return tuple(_tensor_spec(file_name, direction, tensor_name)["shape"])


def _dtype(file_name: str, direction: str, tensor_name: str) -> np.dtype:
    return _NP_DTYPES[_tensor_spec(file_name, direction, tensor_name)["dtype"]]


def _quant(file_name: str, direction: str, tensor_name: str) -> tuple[float, int]:
    q = _tensor_spec(file_name, direction, tensor_name)["quantization_parameters"]
    return q["scale"], q["zero_point"]


def _quantize(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(x / scale) + zero_point
    return np.clip(q, 0, 65535).astype(np.uint16)


def _dequantize(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale


class EncoderModule:
    """Callable matching qai_hub_models' Encoder.forward, backed by encoder.bin."""

    FILE = "encoder.bin"
    INPUT_NAMES = [
        "x", "x_lengths", "tone", "sid", "language",
        "bert", "ja_bert", "sdp_ratio", "length_scale", "noise_scale_w",
    ]
    OUTPUT_NAMES = ["y_lengths", "x_mask", "m_p", "logs_p", "g", "w_ceil"]

    def __init__(self, backend: str = "htp") -> None:
        self.backend = backend

    def __call__(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        tone: torch.Tensor,
        sid: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        ja_bert: torch.Tensor,
        sdp_ratio: torch.Tensor,
        length_scale: torch.Tensor,
        noise_scale_w: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        args = {
            "x": x, "x_lengths": x_lengths, "tone": tone, "sid": sid,
            "language": language, "bert": bert, "ja_bert": ja_bert,
            "sdp_ratio": sdp_ratio, "length_scale": length_scale,
            "noise_scale_w": noise_scale_w,
        }
        inputs = {
            name: args[name].detach().numpy().astype(
                _dtype(self.FILE, "inputs", name)
            )
            for name in self.INPUT_NAMES
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


class FlowModule:
    """Callable matching qai_hub_models' Flow.forward, backed by flow.bin.

    flow.bin's I/O tensors are all uint16-quantized (per metadata.json); this
    wrapper quantizes float32 inputs and dequantizes the float32 output.
    """

    FILE = "flow.bin"
    INPUT_NAMES = ["m_p", "logs_p", "y_mask", "g", "attn_squeezed", "noise_scale"]
    OUTPUT_NAME = "z"

    def __init__(self, backend: str = "htp") -> None:
        self.backend = backend

    def __call__(
        self,
        m_p: torch.Tensor,
        logs_p: torch.Tensor,
        y_mask: torch.Tensor,
        g: torch.Tensor,
        attn_squeezed: torch.Tensor,
        noise_scale: torch.Tensor,
    ) -> torch.Tensor:
        args = {
            "m_p": m_p, "logs_p": logs_p, "y_mask": y_mask, "g": g,
            "attn_squeezed": attn_squeezed, "noise_scale": noise_scale,
        }
        inputs = {}
        for name in self.INPUT_NAMES:
            scale, zero_point = _quant(self.FILE, "inputs", name)
            inputs[name] = _quantize(
                args[name].detach().numpy().astype(np.float32), scale, zero_point
            )
        outputs = run_context_binary(
            FLOW_BIN,
            inputs=inputs,
            output_names=[self.OUTPUT_NAME],
            output_shapes={self.OUTPUT_NAME: _shape(self.FILE, "outputs", self.OUTPUT_NAME)},
            output_dtypes={self.OUTPUT_NAME: _dtype(self.FILE, "outputs", self.OUTPUT_NAME)},
            backend=self.backend,
        )
        scale, zero_point = _quant(self.FILE, "outputs", self.OUTPUT_NAME)
        z = _dequantize(outputs[self.OUTPUT_NAME], scale, zero_point)
        return torch.from_numpy(z)


class DecoderModule:
    """Callable matching qai_hub_models' Decoder.forward, backed by decoder.bin.

    decoder.bin's I/O tensors are all uint16-quantized (per metadata.json).
    """

    FILE = "decoder.bin"
    INPUT_NAMES = ["z", "g"]
    OUTPUT_NAME = "audio"

    def __init__(self, backend: str = "htp") -> None:
        self.backend = backend

    def __call__(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        args = {"z": z, "g": g}
        inputs = {}
        for name in self.INPUT_NAMES:
            scale, zero_point = _quant(self.FILE, "inputs", name)
            inputs[name] = _quantize(
                args[name].detach().numpy().astype(np.float32), scale, zero_point
            )
        outputs = run_context_binary(
            DECODER_BIN,
            inputs=inputs,
            output_names=[self.OUTPUT_NAME],
            output_shapes={self.OUTPUT_NAME: _shape(self.FILE, "outputs", self.OUTPUT_NAME)},
            output_dtypes={self.OUTPUT_NAME: _dtype(self.FILE, "outputs", self.OUTPUT_NAME)},
            backend=self.backend,
        )
        scale, zero_point = _quant(self.FILE, "outputs", self.OUTPUT_NAME)
        audio = _dequantize(outputs[self.OUTPUT_NAME], scale, zero_point)
        return torch.from_numpy(audio)


def build_app(backend: str = "htp") -> MeloTTSApp:
    tts_object = get_tts_object(TTSLanguage.ENGLISH)
    return MeloTTSApp(
        encoder=EncoderModule(backend=backend),
        flow=FlowModule(backend=backend),
        decoder=DecoderModule(backend=backend),
        tts_object=tts_object,
        language=TTSLanguage.ENGLISH,
    )


def _tts_to_file_safe(app: MeloTTSApp, text: str, speaker_id: int, output_path: str) -> None:
    """Reimplements MeloTTSApp.tts_to_file's decoder chunking loop with one
    fix: qai_hub_models's own version (melotts/app.py's tts_to_file) slices
    the fixed-length z tensor as z[:, :, t-OVERLAP : t+MAX_DEC_SEQ_LEN+OVERLAP]
    on every iteration without bounds-checking against z's actual length
    (UPSAMPLED_MAX_SEQ_LEN=1536). Once t gets close to that ceiling -- which
    happens for any text whose synthesized length isn't tiny -- the slice
    comes back shorter than MAX_DEC_SEQ_LEN + 2*OVERLAP=64 frames (Python
    slicing just truncates past the end rather than erroring), and
    decoder.bin then rejects the undersized tensor with a batch-size/file-size
    mismatch. Fix: pad the same way the *first* chunk already is (via
    z_buf = zeros(...); z_buf[:len] = z[:len]) on every chunk, not just the
    first."""
    from qai_hub_models.models._shared.melotts.model import (
        DEC_SEQ_OVERLAP,
        MAX_DEC_SEQ_LEN,
        UPSAMPLE_FACTOR,
        UPSAMPLED_MAX_SEQ_LEN,
    )
    from qai_hub_models.models._shared.voiceai_tts.app_utils import generate_path

    phones, tones, lang_ids, bert, ja_bert, phone_len = app.preprocess_text(app.tts_object, text)
    x = phones.unsqueeze(0)
    x_lengths = torch.tensor([phone_len], dtype=torch.int64)
    sid = torch.tensor([speaker_id], dtype=torch.int64)
    tone = tones.unsqueeze(0)
    language_ids = lang_ids.unsqueeze(0)
    bert = bert.unsqueeze(0)
    ja_bert = ja_bert.unsqueeze(0)
    sdp_ratio_pt = torch.tensor([0.2], dtype=torch.float32)
    length_scale_pt = torch.tensor([1.0], dtype=torch.float32)
    noise_scale_w_pt = torch.tensor([0.8], dtype=torch.float32)

    y_lengths, x_mask, m_p, logs_p, g, w_ceil = app.encoder(
        x, x_lengths, tone, sid, language_ids, bert, ja_bert,
        sdp_ratio_pt, length_scale_pt, noise_scale_w_pt,
    )
    if int(y_lengths[0]) > UPSAMPLED_MAX_SEQ_LEN:
        raise TTSTextTooLongError(
            f"Encoder predicted y_lengths={int(y_lengths[0])} frames, exceeding "
            f"UPSAMPLED_MAX_SEQ_LEN={UPSAMPLED_MAX_SEQ_LEN} -- this text needs to "
            f"be split into smaller chunks (phone_len={phone_len} was under the "
            f"512 cap, but predicted duration overflowed anyway)."
        )
    y_mask = torch.unsqueeze(
        torch.arange(UPSAMPLED_MAX_SEQ_LEN) < y_lengths.unsqueeze(dim=-1), dim=1
    ).to(torch.float32)
    attn_mask = x_mask.unsqueeze(dim=2) * y_mask.unsqueeze(dim=-1)
    attn = generate_path(w_ceil, attn_mask)
    attn_squeezed = attn.squeeze(1).to(torch.float32)
    m_p = m_p.to(torch.float32)
    logs_p = logs_p.to(torch.float32)
    noise_scale_pt = torch.tensor([0.667], dtype=torch.float32)
    z = app.flow(m_p, logs_p, y_mask, g, attn_squeezed, noise_scale_pt)

    chunk_width = MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP
    z_len = z.shape[2]

    def _padded_slice(start: int, stop: int) -> torch.Tensor:
        buf = torch.zeros([z.shape[0], z.shape[1], chunk_width], dtype=torch.float32)
        clipped_start, clipped_stop = max(start, 0), min(stop, z_len)
        if clipped_stop > clipped_start:
            buf[:, :, clipped_start - start : clipped_stop - start] = z[:, :, clipped_start:clipped_stop]
        return buf

    z_buf = _padded_slice(0, MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)
    audio_chunk = app.decoder(z_buf, g)
    audio = audio_chunk.squeeze()[: MAX_DEC_SEQ_LEN * UPSAMPLE_FACTOR]

    total_dec_seq_len = MAX_DEC_SEQ_LEN
    while total_dec_seq_len < y_lengths:
        z_buf = _padded_slice(
            total_dec_seq_len - DEC_SEQ_OVERLAP,
            total_dec_seq_len + MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP,
        )
        audio_chunk = app.decoder(z_buf, g)
        audio_chunk = audio_chunk.squeeze()[
            DEC_SEQ_OVERLAP * UPSAMPLE_FACTOR : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP) * UPSAMPLE_FACTOR
        ]
        audio = torch.cat([audio, audio_chunk])
        total_dec_seq_len += MAX_DEC_SEQ_LEN

    length = int(y_lengths[0]) * UPSAMPLE_FACTOR
    audio = audio.squeeze()[:length]
    sf.write(output_path, audio.detach().cpu().numpy(), samplerate=app.tts_object.hps.data.sampling_rate)


def run(text: str, output_path: str, backend: str = "htp") -> str:
    app = build_app(backend=backend)
    _tts_to_file_safe(app, text, app.speaker_id, output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("output_wav")
    parser.add_argument("--backend", choices=["cpu", "htp"], default="htp")
    args = parser.parse_args()

    path = run(args.text, args.output_wav, backend=args.backend)
    print(f"Audio saved to {path}")

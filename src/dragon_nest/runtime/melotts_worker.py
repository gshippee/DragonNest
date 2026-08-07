"""MeloTTS synthesis worker -- runs inside the ``qai_env`` interpreter, NOT
inside DragonNest's own venv.

``Image2Audio/melotts_pipeline.py`` needs ``qai_hub_models``, ``melo``, and
``soundfile``; DragonNest's ``.venv`` has none of them, and installing ``melo``
there would drag its own pinned torch/transformers/jieba/pyopenjtalk set into
the Brain's environment. So this file is spawned as a subprocess by
``melotts_runner.py`` the same way ``genie_runner.py`` spawns
``genie-t2t-run.exe`` -- the NPU is reached through ``qnn-net-run.exe``
(a native aarch64 binary) either way, so nothing is lost by staying
out-of-process.

Consequently this module must import ONLY the standard library plus
``Image2Audio``'s own modules. ``dragon_nest`` is not installed in ``qai_env``
and importing it here would fail.

Protocol: every result -- success or failure -- is one line on stdout prefixed
with ``RESULT_MARKER``. Everything else torch/melo/jieba print during model
load is ignored by the caller, which reads only the marked line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

RESULT_MARKER = "__DRAGONNEST_TTS__"

# Distinct exit codes so the caller can tell "this machine isn't set up for
# speech" (a 503 the user can act on) from "synthesis itself broke" (a 500).
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_UNAVAILABLE = 3


def _image2audio_dir() -> Path:
    """Locate the Image2Audio package: ``<repo>/Image2Audio``.

    This file lives at ``<repo>/src/dragon_nest/runtime/melotts_worker.py``,
    so the repo root is three parents up. ``DRAGONNEST_IMAGE2AUDIO_DIR``
    overrides for a non-standard checkout layout.
    """
    override = os.environ.get("DRAGONNEST_IMAGE2AUDIO_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "Image2Audio"


def _emit(payload: dict) -> None:
    sys.stdout.write(f"{RESULT_MARKER} {json.dumps(payload)}\n")
    sys.stdout.flush()


def _fail(kind: str, message: str, detail: str = "") -> int:
    _emit({"ok": False, "kind": kind, "error": message, "detail": detail})
    return EXIT_UNAVAILABLE if kind == "unavailable" else EXIT_FAILED


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="htp", choices=["cpu", "htp"])
    parser.add_argument("--max-phones", type=int, default=0)
    args = parser.parse_args()

    package_dir = _image2audio_dir()
    if not package_dir.is_dir():
        return _fail(
            "unavailable",
            f"Image2Audio package not found at {package_dir}",
            "Set DRAGONNEST_IMAGE2AUDIO_DIR if this checkout uses a different layout.",
        )
    sys.path.insert(0, str(package_dir))

    try:
        import numpy as np
        import soundfile as sf

        import chunking
        import device_config
        import melotts_pipeline
    except ImportError as exc:
        return _fail(
            "unavailable",
            f"speech dependencies are missing from {sys.executable}: {exc}",
            "This interpreter needs qai_hub_models, melo, soundfile, and torch "
            "(the qai_env environment). Point DRAGONNEST_TTS_PYTHON at it.",
        )

    # MeloTTS's .bin files are QNN *context binaries* compiled for HTP at
    # export time; QnnCpu.dll cannot deserialize them ("Context de-serialization
    # failed"). There is no CPU fallback for this model -- say so plainly rather
    # than letting the backend fail deep inside qnn-net-run.
    if args.backend != "htp":
        return _fail(
            "unavailable",
            f"MeloTTS supports backend='htp' only, got {args.backend!r}",
            "Its context binaries are precompiled for the HTP backend and "
            "cannot be loaded on CPU.",
        )

    melotts_dir = device_config.MELOTTS_DIR
    missing = [
        name
        for name in ("encoder.bin", "flow.bin", "decoder.bin", "metadata.json")
        if not (melotts_dir / name).is_file()
    ]
    if missing:
        return _fail(
            "unavailable",
            f"MeloTTS model files missing from {melotts_dir}: {', '.join(missing)}",
            "Download the melotts_en Snapdragon X Elite artifact and set "
            "MELOTTS_DIR to the directory holding the .bin files.",
        )

    text = args.text_file.read_text(encoding="utf-8").strip()
    if not text:
        return _fail("failed", "no text to speak")

    max_phones = args.max_phones or chunking.DEFAULT_MAX_TTS_PHONES
    try:
        # Chunking is not optional. MeloTTSApp.preprocess_text truncates past
        # MAX_SEQ_LEN=512 phones while still reporting the pre-truncation
        # count, so an over-budget chunk synthesizes short but gets trimmed
        # long -- the tail plays the decoder's zero-padded buffer as static.
        # chunk_text_for_tts measures phones through MeloTTS's own G2P rather
        # than estimating from character count, which badly undershoots.
        chunks = chunking.chunk_text_for_tts(text, max_phones=max_phones)
    except Exception as exc:
        return _fail("failed", f"could not chunk text for TTS: {exc}", traceback.format_exc())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.parent / f".{args.output.stem}.parts"
    scratch.mkdir(parents=True, exist_ok=True)

    try:
        # Build the app once and reuse it across chunks. melotts_pipeline.run()
        # rebuilds it per call, which re-pays get_tts_object's melo + BERT load
        # on every chunk. _tts_to_file_safe is the bounds-checked synthesis path
        # (qai_hub_models' own tts_to_file slices z out of range on the final
        # decoder chunk); it is used deliberately in place of the public run().
        app = melotts_pipeline.build_app(backend=args.backend)
        sample_rate = int(app.tts_object.hps.data.sampling_rate)

        segments = []
        for index, chunk in enumerate(chunks):
            part_path = scratch / f"part_{index:03d}.wav"
            melotts_pipeline._tts_to_file_safe(
                app, chunk, app.speaker_id, str(part_path)
            )
            audio, part_rate = sf.read(str(part_path), dtype="float32")
            if part_rate != sample_rate:
                return _fail(
                    "failed",
                    f"chunk {index} came back at {part_rate}Hz, expected {sample_rate}Hz",
                )
            segments.append(audio)

        if not segments:
            return _fail("failed", "synthesis produced no audio")

        combined = np.concatenate(segments) if len(segments) > 1 else segments[0]
        # PCM_16 rather than soundfile's default 32-bit float for a float input:
        # same audible quality for speech at half the bytes over the wire.
        sf.write(
            str(args.output),
            np.clip(combined, -1.0, 1.0),
            samplerate=sample_rate,
            subtype="PCM_16",
        )
    except melotts_pipeline.TTSTextTooLongError as exc:
        return _fail("failed", f"text too long for one TTS chunk: {exc}")
    except Exception as exc:
        return _fail("failed", f"MeloTTS synthesis failed: {exc}", traceback.format_exc())
    finally:
        for part in scratch.glob("part_*.wav"):
            part.unlink(missing_ok=True)
        try:
            scratch.rmdir()
        except OSError:
            pass

    _emit(
        {
            "ok": True,
            "output": str(args.output),
            "sample_rate": sample_rate,
            "chunks": len(chunks),
            "duration_seconds": round(len(combined) / sample_rate, 3),
        }
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

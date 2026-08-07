"""Thin subprocess wrapper around MeloTTS, running on the Snapdragon NPU.

Mirrors ``genie_runner.py``'s pattern: MeloTTS's three context binaries
(encoder/flow/decoder) are driven by ``Image2Audio/melotts_pipeline.py``, which
needs ``qai_hub_models``/``melo``/``soundfile`` -- packages deliberately kept
out of DragonNest's own venv. ``melotts_worker.py`` is therefore run under a
second interpreter (``qai_env`` by default, override with
``DRAGONNEST_TTS_PYTHON``) and this module speaks to it over one marked stdout
line.

Speech here is a capability of the machine hosting the Brain, not a routed
fabric task: the Brain and the dashboard share a process (``run_brain.py``), so
the .wav this produces is served straight to the browser over HTTP and never
crosses the gRPC transport.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from . import npu_lease
from .melotts_worker import RESULT_MARKER, EXIT_UNAVAILABLE

WORKER = Path(__file__).with_name("melotts_worker.py")

DEFAULT_TTS_PYTHON = Path.home() / "qai_env" / "Scripts" / "python.exe"

SCRATCH_ROOT = (
    Path(
        os.environ.get(
            "DRAGONNEST_SCRATCH_DIR",
            str(Path(tempfile.gettempdir()) / "dragon_nest"),
        )
    )
    / "tts"
)

# Every qnn-net-run call is a fresh process: one encoder, one flow, then one
# decoder call per 40 output frames (up to ~38 per chunk at the 1536-frame
# ceiling), on top of melo/BERT init in the worker. A single ~480-phone chunk
# lands around a minute; this ceiling covers a multi-chunk reply without
# hanging the request forever if the DSP session is wedged.
DEFAULT_TIMEOUT_SEC = 600

# How long speech waits for the pinned language model to release the NPU before
# giving up. The speaker button only lights up once a response has arrived --
# by which point that task's Genie call is already done -- so contention comes
# from a *concurrent* request, and a short wait covers the overlap without
# leaving the HTTP request hanging.
DEFAULT_NPU_WAIT_SEC = 30

# 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) is how qnn-net-run.exe fail-fasts
# when the Hexagon skel library can't be resolved -- it crashes *after* the
# graph appears to compile cleanly, so the log looks like success right up to
# the exit code. ADSP_LIBRARY_PATH pointing at lib\hexagon-v73\unsigned is the
# fix; surface it rather than making the next person rediscover it.
_FAIL_FAST_RETURNCODES = {0xC0000409, 0xC0000409 - (1 << 32)}  # unsigned, signed
_ADSP_HINT = (
    "qnn-net-run.exe fail-fasted (0xC0000409). This usually means "
    "ADSP_LIBRARY_PATH is unset or wrong -- it must point at "
    "<QAIRT_ROOT>\\lib\\hexagon-v73\\unsigned, where libQnnHtpV73Skel.so "
    "actually lives. The DspTransport.openSession warning in the log is the "
    "cause here, not cosmetic."
)


class TtsError(RuntimeError):
    """Synthesis was attempted and failed."""


class TtsUnavailableError(TtsError):
    """This machine is not provisioned for speech at all.

    Distinct from ``TtsError`` so the API can answer 503 (nothing to retry
    until the host is set up) instead of 500.
    """


def tts_python_executable() -> Path:
    override = os.environ.get("DRAGONNEST_TTS_PYTHON")
    return Path(override) if override else DEFAULT_TTS_PYTHON


def _parse_result(stdout: str | None) -> dict:
    """Read the worker's marked result line.

    torch/melo/jieba all write to stdout during model load, so the payload is
    found by marker rather than by assuming it is the only output. As in
    ``genie_runner``, ``subprocess.run``'s ``.stdout`` can come back None on
    Windows under DSP session contention even at returncode 0 -- that must not
    leak an AttributeError.
    """
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(RESULT_MARKER):
            payload = line[len(RESULT_MARKER) :].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError as exc:
                raise TtsError(f"unreadable TTS worker result: {payload}") from exc
    raise TtsError(
        "the TTS worker produced no result line; it likely died during model "
        f"load.\nstdout:\n{stdout or '(empty)'}"
    )


def synthesize(
    text: str,
    output_path: str | Path,
    backend: str = "htp",
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    max_phones: int = 0,
    python_executable: str | Path | None = None,
    npu_wait_sec: float = DEFAULT_NPU_WAIT_SEC,
) -> Path:
    """Speak ``text`` to a 16-bit PCM .wav at ``output_path``.

    Long text is split on sentence boundaries by the worker (measured against
    MeloTTS's real phone budget) and the chunks are concatenated, so callers
    can pass a whole model response.
    """
    if not text.strip():
        raise TtsError("no text to speak")

    interpreter = (
        Path(python_executable) if python_executable else tts_python_executable()
    )
    if not interpreter.is_file():
        raise TtsUnavailableError(
            f"speech interpreter not found: {interpreter}. Speech needs the "
            "qai_env environment (qai_hub_models, melo, soundfile, torch); set "
            "DRAGONNEST_TTS_PYTHON to its python.exe."
        )

    output = Path(output_path)
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    text_file = SCRATCH_ROOT / f"tts_text_{uuid.uuid4().hex[:12]}.txt"
    command = [
        str(interpreter),
        str(WORKER),
        "--text-file",
        str(text_file.resolve()),
        "--output",
        str(output.resolve()),
        "--backend",
        backend,
    ]
    if max_phones:
        command += ["--max-phones", str(max_phones)]

    try:
        # The prompt goes through a file for the same reason genie_runner uses
        # --prompt_file: a full model response can exceed the Windows
        # command-line length limit.
        text_file.write_text(text, encoding="utf-8")
        try:
            # Yield to the pinned Qwen3-4B bundle. Speech is the interruptible
            # workload here: rather than opening a second DSP session while the
            # language model holds the device, wait briefly and then decline.
            with npu_lease.lease(
                "melotts", hold_sec=timeout_sec, wait_sec=npu_wait_sec, required=True
            ):
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
        except subprocess.TimeoutExpired as exc:
            raise TtsError(
                f"MeloTTS synthesis timed out after {timeout_sec}s. This usually "
                "means another process (a stale qnn-net-run.exe or geniex.exe) is "
                "still holding the NPU/DSP session -- check with Get-Process and "
                "kill orphans -- rather than genuine slowness.\n"
                f"partial stdout:\n{exc.stdout}\npartial stderr:\n{exc.stderr}"
            ) from exc

        if result.returncode in _FAIL_FAST_RETURNCODES:
            raise TtsError(f"{_ADSP_HINT}\nstderr:\n{result.stderr}")

        payload = _parse_result(result.stdout)
        if not payload.get("ok"):
            message = payload.get("error", "MeloTTS synthesis failed")
            detail = payload.get("detail", "")
            if payload.get("kind") == "unavailable" or result.returncode == EXIT_UNAVAILABLE:
                raise TtsUnavailableError(f"{message}\n{detail}".strip())
            raise TtsError(f"{message}\n{detail}".strip())
    finally:
        text_file.unlink(missing_ok=True)

    if not output.is_file():
        raise TtsError(f"the TTS worker reported success but wrote no file: {output}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speak text with MeloTTS on the NPU")
    parser.add_argument("text")
    parser.add_argument("output_wav")
    parser.add_argument("--backend", choices=["cpu", "htp"], default="htp")
    args = parser.parse_args()

    path = synthesize(args.text, args.output_wav, backend=args.backend)
    print(f"Audio saved to {path}")

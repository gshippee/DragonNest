"""Thin subprocess wrapper around Qualcomm Genie's genie-t2t-run.exe, running
Qwen3-4B (w4a16-quantized) on the Snapdragon NPU. Mirrors qnn_runner.py's
pattern for qnn-net-run.exe -- a CLI tool with no persistent-process/API mode,
so every call here pays a full ~3.1GB context-binary load.

genie-t2t-run.exe's own stdout wraps the reply as:
    [PROMPT]: <the prompt we sent>

    [BEGIN]: <think>...</think>

    <answer text>[END]
Qwen3 always emits a <think>...</think> reasoning block first (this model was
built with reasoning mode on); only the text after it is the actual answer.
_parse_response() extracts the text strictly between [BEGIN]: and [END] and
strips the leading <think> block.

Long prompts (e.g. full OCR text of a dense note) go through --prompt_file
instead of -p, since -p places the prompt on the Windows command line, which
has a length limit; a temp file has no such limit.

Ported from ``PersonaCare/genie_runner.py``, where it was validated with the
Qwen3-4B Snapdragon X Elite bundle. DragonNest additionally permits each call
to select its bundle, executable, and config from an artifact manifest.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

from . import npu_lease

GENIE_DIR = Path(
    os.environ.get(
        "GENIE_DIR",
        str(
            Path.home()
            / "Downloads"
            / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x_elite"
            / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x_elite"
        ),
    )
)
GENIE_EXE = GENIE_DIR / "genie-t2t-run.exe"
GENIE_CONFIG = "genie_config.json"  # relative to GENIE_DIR, per genie_config.json's own relative model/tokenizer paths

_SCRATCH_ROOT = (
    Path(
        os.environ.get(
            "DRAGONNEST_SCRATCH_DIR",
            str(Path(tempfile.gettempdir()) / "dragon_nest"),
        )
    )
    / "genie"
)

# Genie pays a full ~3.1GB context-binary load plus token-by-token generation
# on every call; observed cold-start+short-reply calls finish in well under two
# minutes. A run past this is symptomatic of NPU/DSP session contention (e.g. a
# stale geniex.exe still holding the HTP device) rather than genuine slowness --
# fail fast with a clear error instead of hanging silently.
DEFAULT_TIMEOUT_SEC = 180

_RESPONSE_RE = re.compile(r"\[BEGIN\]:\s*(.*?)\s*\[END\]", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_chatml_prompt(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _parse_response(stdout: str | None) -> str:
    # subprocess.run's .stdout can come back None (observed on Windows under
    # NPU/DSP session contention) even when returncode == 0, outside the
    # already-handled TimeoutExpired path. Fall through to the existing
    # "no [BEGIN]:...[END] response" error instead of a raw regex TypeError.
    stdout = stdout or ""
    match = _RESPONSE_RE.search(stdout)
    if not match:
        raise RuntimeError(
            f"Could not find a [BEGIN]:...[END] response in genie-t2t-run.exe output:\n{stdout}"
        )
    reply = _THINK_RE.sub("", match.group(1)).strip()
    return reply


def run_genie(
    prompt: str,
    backend: str = "htp",
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    genie_dir: str | Path | None = None,
    genie_executable: str | Path | None = None,
    genie_config: str | Path | None = None,
) -> str:
    """Run one text-completion query against Qwen3-4B via Genie, return the
    final answer text (with any <think> reasoning block stripped)."""
    if backend != "htp":
        raise ValueError(
            f"genie-t2t-run.exe in this bundle only supports backend='htp', got {backend!r}"
        )

    bundle_dir = Path(genie_dir) if genie_dir is not None else GENIE_DIR
    executable = (
        Path(genie_executable)
        if genie_executable is not None
        else bundle_dir / GENIE_EXE.name
    )
    config = str(genie_config) if genie_config is not None else GENIE_CONFIG
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Genie bundle directory not found: {bundle_dir}")
    if not executable.is_file():
        raise FileNotFoundError(f"Genie executable not found: {executable}")
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = bundle_dir / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Genie config not found: {config_path}")

    _SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    prompt_file = _SCRATCH_ROOT / f"genie_prompt_{uuid.uuid4().hex[:12]}.txt"
    try:
        prompt_file.write_text(prompt, encoding="utf-8")
        try:
            # Announce that this host's NPU is in use so speech synthesis
            # stands down rather than opening a competing DSP session. This is
            # deliberately fail-open (required=False): the pinned language
            # model is the priority workload and must never be blocked, or
            # made to fail, by the lease.
            with npu_lease.lease(
                "qwen3-4b-genie",
                hold_sec=timeout_sec,
                wait_sec=0.0,
                required=False,
            ):
                result = subprocess.run(
                    [
                        str(executable),
                        "-c",
                        str(config_path),
                        "--prompt_file",
                        str(prompt_file.resolve()),
                    ],
                    cwd=bundle_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"genie-t2t-run.exe timed out after {timeout_sec}s. This usually means "
                f"another process (a stale geniex.exe or qnn-net-run.exe) is still holding "
                f"the NPU/DSP session open -- check with Get-Process and kill orphaned "
                f"processes -- rather than genuine slowness.\n"
                f"partial stdout:\n{exc.stdout}\npartial stderr:\n{exc.stderr}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"genie-t2t-run.exe failed (exit {result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return _parse_response(result.stdout)
    finally:
        prompt_file.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("user_prompt")
    parser.add_argument("--system", default="You are a helpful AI assistant.")
    args = parser.parse_args()

    reply = run_genie(build_chatml_prompt(args.system, args.user_prompt))
    print(reply)

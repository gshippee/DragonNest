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
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path

import device_config

GENIE_DIR = device_config.GENIE_DIR
GENIE_EXE = GENIE_DIR / "genie-t2t-run.exe"
GENIE_CONFIG = "genie_config.json"  # relative to GENIE_DIR, per genie_config.json's own relative model/tokenizer paths

_SCRATCH_ROOT = Path(__file__).parent / "scratch"

# Genie pays a full ~3.1GB context-binary load plus token-by-token generation
# on every call; observed cold-start+short-reply calls finish in well under two
# minutes. A run past this is symptomatic of NPU/DSP session contention (e.g. a
# stale geniex.exe still holding the HTP device) rather than genuine slowness --
# fail fast with a clear error instead of hanging silently.
DEFAULT_TIMEOUT_SEC = 180

_RESPONSE_RE = re.compile(r"\[BEGIN\]:\s*(.*?)\s*\[END\]", re.DOTALL)
_PARTIAL_RESPONSE_RE = re.compile(r"\[BEGIN\]:\s*(.*)", re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_EOS_TOKEN_RE = re.compile(r"</s>", re.IGNORECASE)


def build_chatml_prompt(system: str, user: str, no_think: bool = False) -> str:
    """Build a ChatML prompt for Qwen3. Qwen3 always defaults to reasoning
    mode (an open <think> block after the assistant marker) -- fine for
    genuinely open-ended questions, but on messy/confusing input (e.g. a
    noisy OCR transcript) its chain-of-thought can spiral into rambling
    self-argument that never converges within genie's call timeout. Passing
    no_think=True pre-fills an already-closed, empty <think></think> block,
    which per Qwen3's documented convention skips reasoning entirely and
    makes generation start directly on the answer -- appropriate for the
    mechanical filter/extract/merge tasks in this pipeline, which don't
    benefit from reasoning anyway."""
    assistant_prefix = "<think>\n\n</think>\n\n" if no_think else ""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )


def _parse_response(stdout: str, allow_partial: bool = False) -> str:
    match = _RESPONSE_RE.search(stdout)
    if match:
        reply = match.group(1)
    else:
        if not allow_partial:
            raise RuntimeError(
                f"Could not find a [BEGIN]:...[END] response in genie-t2t-run.exe output:\n{stdout}"
            )
        # A call that timed out mid-generation never emitted [END] -- everything
        # after [BEGIN]: is still the model's actual (unfinished) answer, so
        # salvage it rather than discarding a mostly-complete response just
        # because generation didn't converge in time.
        partial_match = _PARTIAL_RESPONSE_RE.search(stdout)
        if not partial_match:
            raise RuntimeError(
                f"Could not find a [BEGIN]:... response in genie-t2t-run.exe output:\n{stdout}"
            )
        reply = partial_match.group(1)
    # Qwen3 always reasons before answering, but on confusing/rambling input it
    # sometimes emits malformed reasoning -- multiple </think> closes, a
    # missing opening <think>, or a stray </s> -- rather than one clean
    # <think>...</think> pair. Cutting at the LAST </think> (instead of
    # matching a <think>...</think> pair) reliably drops all reasoning text
    # regardless of how it was malformed, since genuine answer text never
    # contains that literal tag.
    closes = list(_THINK_CLOSE_RE.finditer(reply))
    if closes:
        reply = reply[closes[-1].end():]
    reply = _EOS_TOKEN_RE.sub("", reply)
    return reply.strip()



def run_genie(
    prompt: str,
    backend: str = "htp",
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    salvage_partial: bool = False,
    seed: int | None = None,
) -> str:
    """Run one text-completion query against Qwen3-4B via Genie, return the
    final answer text (with any <think> reasoning block stripped). Even with
    no_think and a wide repetition-penalty window, this 4B model sometimes
    never emits [END] on messy input -- it produces a complete, reasonable
    answer and then keeps rambling into a second, garbled one, burning the
    call's whole timeout. With salvage_partial=True, a timeout doesn't raise:
    whatever was generated before the kill (usually the first, good answer)
    is parsed and returned instead, since for this pipeline's mechanical
    tasks a slightly-early-cut answer beats a hard failure.

    genie_config.json pins a fixed sampler seed (42), so by default two calls
    with the same prompt deterministically produce the same reply -- a retry
    of a bad/unparseable reply is pointless unless the seed changes too. Pass
    seed to override it: a per-call config JSON (a copy of genie_config.json
    with dialog.sampler.seed replaced) is written to scratch and passed via
    -c instead of the default config name."""
    if backend != "htp":
        raise ValueError(f"genie-t2t-run.exe in this bundle only supports backend='htp', got {backend!r}")

    _SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    prompt_file = _SCRATCH_ROOT / f"genie_prompt_{uuid.uuid4().hex[:12]}.txt"
    config_file = None
    config_arg = GENIE_CONFIG
    try:
        prompt_file.write_text(prompt, encoding="utf-8")
        if seed is not None:
            config = json.loads((GENIE_DIR / GENIE_CONFIG).read_text(encoding="utf-8"))
            config["dialog"]["sampler"]["seed"] = seed
            config_file = _SCRATCH_ROOT / f"genie_config_{uuid.uuid4().hex[:12]}.json"
            config_file.write_text(json.dumps(config), encoding="utf-8")
            config_arg = str(config_file.resolve())
        try:
            result = subprocess.run(
                [
                    str(GENIE_EXE),
                    "-c", config_arg,
                    "--prompt_file", str(prompt_file.resolve()),
                ],
                cwd=GENIE_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            if salvage_partial and exc.stdout:
                print(
                    f"  (warning: genie-t2t-run.exe timed out after {timeout_sec}s; "
                    f"salvaging partial output)"
                )
                return _parse_response(exc.stdout, allow_partial=True)
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
        if config_file is not None:
            config_file.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("user_prompt")
    parser.add_argument("--system", default="You are a helpful AI assistant.")
    args = parser.parse_args()

    reply = run_genie(build_chatml_prompt(args.system, args.user_prompt))
    print(reply)

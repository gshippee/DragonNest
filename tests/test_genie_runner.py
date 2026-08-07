from __future__ import annotations

import pytest

from dragon_nest.runtime.genie_runner import _parse_response


def test_parse_response_none_stdout_raises_clean_runtime_error():
    """subprocess.run's .stdout can come back None (observed on Windows
    under NPU/DSP session contention) even when returncode == 0, outside
    the already-handled TimeoutExpired path. This must not leak a raw
    regex TypeError ("expected string or bytes-like object, got
    'NoneType'") to the end user -- it must fall through to the existing
    "no [BEGIN]:...[END] response" error."""
    with pytest.raises(RuntimeError, match=r"\[BEGIN\]:\.\.\.\[END\]"):
        _parse_response(None)


def test_parse_response_empty_stdout_raises_clean_runtime_error():
    with pytest.raises(RuntimeError, match=r"\[BEGIN\]:\.\.\.\[END\]"):
        _parse_response("")


def test_parse_response_extracts_answer_and_strips_think_block():
    stdout = (
        "[PROMPT]: hello\n\n"
        "[BEGIN]: <think>reasoning</think>\n\nTokyo is the capital.[END]"
    )
    assert _parse_response(stdout) == "Tokyo is the capital."

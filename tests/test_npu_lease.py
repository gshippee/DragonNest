from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from dragon_nest.runtime import melotts_runner, npu_lease
from dragon_nest.runtime.melotts_worker import RESULT_MARKER


@pytest.fixture
def lease_path(tmp_path, monkeypatch):
    path = tmp_path / "npu.lease"
    monkeypatch.setattr(npu_lease, "LEASE_PATH", path)
    return path


def _hold(path: Path, holder: str, expires_in: float) -> None:
    """Simulate another OS process (a Device Agent, the Brain) holding the NPU."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"holder": holder, "pid": -1, "expires_at": time.time() + expires_in}),
        encoding="utf-8",
    )


def test_lease_is_exclusive_while_held(lease_path):
    with npu_lease.lease("qwen3-4b-genie", hold_sec=60):
        assert npu_lease.current_holder() == "qwen3-4b-genie"
        with pytest.raises(npu_lease.NpuBusyError, match="qwen3-4b-genie"):
            with npu_lease.lease("melotts", hold_sec=60, wait_sec=0):
                pass


def test_lease_is_released_after_the_block(lease_path):
    with npu_lease.lease("melotts", hold_sec=60):
        pass
    assert npu_lease.current_holder() == ""
    assert not lease_path.exists()


def test_lease_is_released_even_when_the_block_raises(lease_path):
    with pytest.raises(ValueError):
        with npu_lease.lease("melotts", hold_sec=60):
            raise ValueError("synthesis blew up")
    assert npu_lease.current_holder() == ""


def test_expired_lease_is_stolen_rather_than_blocking_the_device(lease_path):
    """Staleness is time-based: a crashed holder's lease expires when its
    operation could no longer plausibly be running, so a dead process cannot
    strand the NPU until someone deletes a file by hand."""
    _hold(lease_path, "qwen3-4b-genie", expires_in=-1)
    assert npu_lease.current_holder() == ""
    with npu_lease.lease("melotts", hold_sec=60, wait_sec=0):
        assert npu_lease.current_holder() == "melotts"


def test_corrupt_lease_file_does_not_deadlock_the_device(lease_path):
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text("half-written garbage", encoding="utf-8")
    with npu_lease.lease("melotts", hold_sec=60, wait_sec=0):
        assert npu_lease.current_holder() == "melotts"


def test_genie_is_never_blocked_by_a_lease_it_cannot_take(lease_path):
    """The pinned 4B is the priority workload: it announces that the NPU is in
    use so speech stands down, but must never be blocked or failed by the
    lease itself."""
    _hold(lease_path, "melotts", expires_in=600)
    with npu_lease.lease(
        "qwen3-4b-genie", hold_sec=180, wait_sec=0, required=False
    ) as acquired:
        assert acquired is False  # ran anyway


def test_releasing_does_not_steal_a_lease_taken_over_by_someone_else(lease_path):
    with npu_lease.lease("melotts", hold_sec=60):
        _hold(lease_path, "qwen3-4b-genie", expires_in=600)
    # The block exited, but the lease now belongs to the language model.
    assert npu_lease.current_holder() == "qwen3-4b-genie"


def test_speech_declines_rather_than_competing_with_the_pinned_model(
    tmp_path, monkeypatch, lease_path
):
    """End to end: with the 4B holding the NPU, synthesize must not spawn a
    competing qnn-net-run process."""
    _hold(lease_path, "qwen3-4b-genie", expires_in=600)
    monkeypatch.setattr(melotts_runner, "tts_python_executable", lambda: Path(__file__))

    def fail_if_called(command, **kwargs):
        raise AssertionError("speech started a competing DSP session")

    monkeypatch.setattr(melotts_runner.subprocess, "run", fail_if_called)

    with pytest.raises(npu_lease.NpuBusyError, match="qwen3-4b-genie"):
        melotts_runner.synthesize("Hello.", tmp_path / "out.wav", npu_wait_sec=0)


def test_speech_runs_once_the_pinned_model_releases_the_npu(
    tmp_path, monkeypatch, lease_path
):
    monkeypatch.setattr(melotts_runner, "tts_python_executable", lambda: Path(__file__))

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"RIFFfake")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=f"{RESULT_MARKER} " + json.dumps({"ok": True, "output": str(output)}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(melotts_runner.subprocess, "run", fake_run)
    _hold(lease_path, "qwen3-4b-genie", expires_in=-1)  # just finished

    output = tmp_path / "out.wav"
    assert melotts_runner.synthesize("Hello.", output, npu_wait_sec=0) == output
    # And the NPU is handed back for the next language-model call.
    assert npu_lease.current_holder() == ""

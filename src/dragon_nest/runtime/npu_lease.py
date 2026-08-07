"""A cross-process lease on this host's NPU/HTP device.

The Snapdragon X Elite laptop runs the pinned Qwen3-4B Genie bundle *and*
MeloTTS, and both reach the Hexagon DSP the same way -- a short-lived native
process (``genie-t2t-run.exe`` / ``qnn-net-run.exe``) that opens a DSP session
for the duration of its run. Both runners already document what happens when
those overlap: the second process contends for the session, and the symptom is
a timeout or a wedged session that outlives the run
(``DspTransport.openSession qnn_open failed``).

The Brain, a Device Agent, and the TTS worker are separate OS processes, so an
in-process lock cannot coordinate them. This lease is a file, which can.

Priority is deliberately asymmetric. The 4B is the pinned workload:

* ``run_genie`` takes the lease **fail-open** -- it announces that the NPU is
  busy so speech stands down, but it is never blocked or made to fail by the
  lease itself. The validated LLM path keeps working even if leasing breaks.
* Speech takes the lease **required** -- if the language model holds it, speech
  waits briefly and then gives up with :class:`NpuBusyError` rather than
  starting a competing DSP session.

Staleness is time-based rather than PID-based: each holder records when its own
operation could no longer plausibly still be running (its own timeout), so a
crashed holder's lease expires on its own instead of blocking the device until
someone deletes a file. Checking PID liveness would be the obvious alternative,
but ``os.kill(pid, 0)`` is not a safe liveness probe on Windows.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LEASE_PATH = (
    Path(
        os.environ.get(
            "DRAGONNEST_SCRATCH_DIR",
            str(Path(tempfile.gettempdir()) / "dragon_nest"),
        )
    )
    / "npu.lease"
)

_POLL_INTERVAL_SEC = 0.25


class NpuBusyError(RuntimeError):
    """Another workload holds this host's NPU and the caller declined to wait."""


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Unreadable or half-written: treat as absent rather than deadlocking
        # the device on a corrupt file.
        return None


def current_holder() -> str:
    """Name of the live lease holder, or "" if the NPU is free."""
    record = _read(LEASE_PATH)
    if not record or record.get("expires_at", 0) <= time.time():
        return ""
    return str(record.get("holder", "unknown"))


def _try_acquire(holder: str, expires_at: float) -> bool:
    LEASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"holder": holder, "pid": os.getpid(), "expires_at": expires_at}
    )
    try:
        # O_CREAT|O_EXCL is the atomic "claim it only if nobody else has"
        # primitive; two processes racing here cannot both win.
        handle = os.open(LEASE_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = _read(LEASE_PATH)
        if existing is not None and existing.get("expires_at", 0) > time.time():
            return False
        # Expired or unreadable: the previous holder cannot still be running.
        try:
            LEASE_PATH.unlink()
        except OSError:
            return False
        return _try_acquire(holder, expires_at)
    try:
        os.write(handle, payload.encode("utf-8"))
    finally:
        os.close(handle)
    return True


def _release(holder: str) -> None:
    record = _read(LEASE_PATH)
    # Only drop the lease if it is still ours: if it expired and someone else
    # took it, deleting the file would release *their* claim.
    if record is not None and record.get("pid") == os.getpid() and record.get("holder") == holder:
        try:
            LEASE_PATH.unlink()
        except OSError:
            pass


@contextmanager
def lease(
    holder: str,
    hold_sec: float,
    wait_sec: float = 0.0,
    required: bool = True,
) -> Iterator[bool]:
    """Hold this host's NPU for the duration of the block.

    ``hold_sec`` is how long the caller's operation could plausibly run; the
    lease self-expires after that so a crash cannot strand the device.
    ``wait_sec`` is how long to wait for a current holder to finish. With
    ``required=False`` the block runs even if the lease could not be taken
    (used by the pinned language model, which must never be blocked); the
    yielded bool says whether it was actually acquired.
    """
    deadline = time.time() + max(0.0, wait_sec)
    acquired = False
    while True:
        acquired = _try_acquire(holder, time.time() + hold_sec)
        if acquired or time.time() >= deadline:
            break
        time.sleep(_POLL_INTERVAL_SEC)

    if not acquired and required:
        blocker = current_holder() or "another workload"
        raise NpuBusyError(
            f"the NPU is busy with {blocker}; {holder} did not start a competing "
            "DSP session. Try again once that finishes."
        )
    try:
        yield acquired
    finally:
        if acquired:
            _release(holder)

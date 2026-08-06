"""qai_env-side client for onnxrt_worker.py.

qai_env's Python is x64 (emulated on this ARM64 machine) and can never load
ONNX Runtime's QNN HTP backend directly -- only a native ARM64 process can.
This module launches onnxrt_worker.py under qai_env_arm64's native ARM64
interpreter as a persistent subprocess, and drives it over a small
newline-delimited-JSON protocol (see onnxrt_worker.py's docstring). The
worker caches InferenceSessions per (model_path, backend), so repeated calls
against the same model (e.g. an autoregressive decode loop) only pay the
compile/load cost once, for the life of this process.

API mirrors qnn_runner.py's shape (run_dlc/run_context_binary -> here just
run_onnx), but there's no raw-file round trip: inputs/outputs are numpy
arrays in memory, base64-framed over the pipe.
"""

from __future__ import annotations

import atexit
import base64
import json
import subprocess
import threading
from pathlib import Path

import numpy as np

ARM64_PYTHON = Path(r"C:\Users\harisury\qai_env_arm64\Scripts\python.exe")
WORKER_SCRIPT = Path(__file__).with_name("onnxrt_worker.py")

_proc: subprocess.Popen | None = None
_lock = threading.Lock()
_next_id = 0


def _ensure_worker() -> subprocess.Popen:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc

    if not ARM64_PYTHON.exists():
        raise RuntimeError(f"Native ARM64 interpreter not found at {ARM64_PYTHON}")

    _proc = subprocess.Popen(
        [str(ARM64_PYTHON), str(WORKER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    ready_line = _proc.stdout.readline()
    if not ready_line:
        stderr_tail = ""
        raise RuntimeError(
            "onnxrt_worker.py exited before signaling ready. "
            f"{stderr_tail}"
        )
    ready = json.loads(ready_line)
    if not ready.get("ready"):
        raise RuntimeError(f"onnxrt_worker.py sent unexpected startup line: {ready_line!r}")

    atexit.register(_shutdown)
    return _proc


def _shutdown() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        try:
            _proc.stdin.close()
        except OSError:
            pass
        _proc.terminate()
    _proc = None


def shutdown_worker() -> None:
    """Terminate the persistent ARM64 worker (if running), releasing its
    cached InferenceSessions and, with them, their exclusive lock on the
    NPU/DSP device. The worker keeps sessions open by design so repeated
    run_onnx() calls against the same model are fast -- but that open session
    blocks any other exclusive-access HTP consumer (qnn-net-run.exe,
    genie-t2t-run.exe) from acquiring the device in the same process until
    it's released. Callers that hand the device off to one of those between
    run_onnx() calls should call this first. A no-op if no worker is running;
    the worker is lazily respawned on the next run_onnx() call."""
    with _lock:
        _shutdown()


def _encode_tensor(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _decode_tensor(spec: dict) -> np.ndarray:
    raw = base64.b64decode(spec["data_b64"])
    arr = np.frombuffer(raw, dtype=np.dtype(spec["dtype"]))
    return arr.reshape(spec["shape"]).copy()


def run_onnx(
    model_path: str | Path,
    inputs: dict[str, np.ndarray],
    output_names: list[str],
    backend: str = "htp",
) -> dict[str, np.ndarray]:
    """Run one ONNX model via ORT + QNN EP in the ARM64 worker process,
    return outputs by name. Model path must be absolute; the worker loads
    EPContext-wrapped .onnx files (e.g. Whisper) exactly like QDQ .onnx
    files (e.g. EasyOCR) -- ORT resolves the sibling *_qairt_context.bin
    internally in the EPContext case, no special-casing needed here."""
    global _next_id

    with _lock:
        proc = _ensure_worker()
        _next_id += 1
        request_id = _next_id
        request = {
            "id": request_id,
            "model_path": str(Path(model_path).resolve()),
            "backend": backend,
            "inputs": {name: _encode_tensor(arr) for name, arr in inputs.items()},
            "output_names": output_names,
        }
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()

        line = proc.stdout.readline()
        if not line:
            _shutdown()
            raise RuntimeError("onnxrt_worker.py died mid-request (no response on stdout)")
        response = json.loads(line)

    if response["id"] != request_id:
        raise RuntimeError(f"onnxrt_worker.py response id mismatch: expected {request_id}, got {response['id']}")
    if not response["ok"]:
        raise RuntimeError(f"onnxrt_worker.py error: {response['error']}")

    return {name: _decode_tensor(spec) for name, spec in response["outputs"].items()}

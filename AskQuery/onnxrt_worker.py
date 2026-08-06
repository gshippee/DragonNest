"""Runs inside qai_env_arm64 (native ARM64 Python) as a persistent subprocess.

This process is the only thing on this machine that can actually reach the
Hexagon NPU via ONNX Runtime's QNN execution provider -- qai_env is x64
(emulated on this ARM64 machine) and can never load the HTP backend. This
worker exists so qai_env's orchestration code (torch, qai_hub_models,
transformers -- none of which have ARM64 wheels) can still drive real NPU
inference: it launches this script once, keeps it running, and exchanges
requests/responses with it over stdin/stdout while ORT's own diagnostic
chatter (which also lands on stdout) is redirected out of the way first.

Protocol (newline-delimited JSON on the real stdout, one request/response
per line):
  request:  {"id": int, "model_path": str, "backend": "htp"|"cpu",
             "inputs": {name: {"shape": [...], "dtype": str, "data_b64": str}},
             "output_names": [str, ...]}
  response: {"id": int, "ok": true,
             "outputs": {name: {"shape": [...], "dtype": str, "data_b64": str}}}
          | {"id": int, "ok": false, "error": str}

Sessions are cached per (model_path, backend) for the life of the worker, so
repeated calls against the same model (e.g. an autoregressive decode loop)
pay the compile/load cost only once.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import traceback

# QNN's graph-compile progress bar and stage timings print straight to the OS
# stdout file descriptor (bypassing Python's buffering), so it can't share the
# pipe the parent reads as our IPC channel -- interleaved raw bytes broke
# JSON framing. Save a duplicate of the real fd 1 (still connected to the
# parent's pipe) *before* importing onnxruntime, then splice fd 1 itself over
# to devnull so any native C-level writes to "stdout" land there instead.
# IPC then goes exclusively through the saved duplicate.
_real_stdout_fd = os.dup(1)
_ipc_out = os.fdopen(_real_stdout_fd, "w", buffering=1)

_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 1)
os.close(_devnull_fd)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import onnxruntime_qnn  # noqa: E402

_LIB = onnxruntime_qnn.get_library_path()
ort.register_execution_provider_library("QNNExecutionProvider", _LIB)

_BACKEND_PATHS = {
    "htp": onnxruntime_qnn.get_qnn_htp_path(),
}

_SESSIONS: dict[tuple[str, str], ort.InferenceSession] = {}


def _npu_devices():
    return [
        d
        for d in ort.get_ep_devices()
        if d.ep_name == "QNNExecutionProvider"
        and d.device.type == ort.OrtHardwareDeviceType.NPU
    ]


def _get_session(model_path: str, backend: str) -> ort.InferenceSession:
    key = (model_path, backend)
    sess = _SESSIONS.get(key)
    if sess is not None:
        return sess

    devices = _npu_devices()
    if not devices:
        raise RuntimeError("No QNNExecutionProvider NPU device found")

    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {"backend_path": _BACKEND_PATHS[backend]})
    sess = ort.InferenceSession(model_path, sess_options=so, providers=[])
    _SESSIONS[key] = sess
    return sess


def _decode_tensor(spec: dict) -> np.ndarray:
    raw = base64.b64decode(spec["data_b64"])
    arr = np.frombuffer(raw, dtype=np.dtype(spec["dtype"]))
    return arr.reshape(spec["shape"])


def _encode_tensor(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _handle(request: dict) -> dict:
    sess = _get_session(request["model_path"], request["backend"])
    inputs = {name: _decode_tensor(spec) for name, spec in request["inputs"].items()}
    outputs = sess.run(request["output_names"], inputs)
    return {
        "id": request["id"],
        "ok": True,
        "outputs": {
            name: _encode_tensor(arr)
            for name, arr in zip(request["output_names"], outputs)
        },
    }


def main() -> None:
    _ipc_out.write(json.dumps({"ready": True}) + "\n")
    _ipc_out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = _handle(request)
        except Exception as exc:  # noqa: BLE001
            response = {
                "id": request.get("id") if isinstance(request, dict) else None,
                "ok": False,
                "error": f"{exc}\n{traceback.format_exc()}",
            }
        _ipc_out.write(json.dumps(response) + "\n")
        _ipc_out.flush()


if __name__ == "__main__":
    main()

"""One-time setup: precompile EasyOCR's detector.dlc/recognizer.dlc into HTP
context binaries via qnn-context-binary-generator.exe, so easyocr_pipeline.py
can load them with run_context_binary_batch() instead of paying qnn-net-run's
full graph-recompile cost (recognizer.dlc: ~185-200s) on every single call.

Run this once (or whenever the source .dlc files change):
    python prepare_context_binaries.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import device_config
from qnn_runner import BIN_DIR, HEXAGON_DIR, LIB_DIR

EASYOCR_DIR = device_config.EASYOCR_DLC_DIR
DETECTOR_DLC = EASYOCR_DIR / "detector.dlc"
RECOGNIZER_DLC = EASYOCR_DIR / "recognizer.dlc"

CONTEXT_BIN_DIR = Path(__file__).parent / "context_cache"
DETECTOR_CTX_BIN = CONTEXT_BIN_DIR / "detector_htp.bin"
RECOGNIZER_CTX_BIN = CONTEXT_BIN_DIR / "recognizer_htp.bin"

QNN_CONTEXT_BINARY_GENERATOR = BIN_DIR / "qnn-context-binary-generator.exe"


def _env():
    import os

    env = os.environ.copy()
    env["PATH"] = f"{LIB_DIR};{HEXAGON_DIR};" + env.get("PATH", "")
    return env


def generate_context_binary(dlc_path: Path, binary_file: Path, backend: str = "htp") -> None:
    """Run qnn-context-binary-generator.exe once to compile dlc_path into an
    HTP context binary saved at binary_file. This pays the graph-optimization
    /VTCM-allocation/graph-sequencing cost that qnn-net-run.exe would otherwise
    redo on every call -- a one-time cost (observed ~185-200s for recognizer.dlc)
    instead of a per-call one."""
    binary_file.parent.mkdir(parents=True, exist_ok=True)
    backend_dll = LIB_DIR / ("QnnHtp.dll" if backend == "htp" else "QnnCpu.dll")
    cmd = [
        str(QNN_CONTEXT_BINARY_GENERATOR),
        "--dlc_path", str(dlc_path),
        "--backend", str(backend_dll),
        "--binary_file", binary_file.name,
        "--output_dir", str(binary_file.parent),
    ]
    print(f"Generating context binary for {dlc_path.name} -> {binary_file} ...")
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=binary_file.parent, env=_env(),
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"qnn-context-binary-generator failed (exit {result.returncode}) for {dlc_path}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    # qnn-context-binary-generator appends ".bin" to --binary_file itself.
    produced = binary_file.parent / f"{binary_file.name}.bin"
    if not produced.exists():
        raise FileNotFoundError(
            f"Expected {produced} to exist after generation.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    produced.replace(binary_file)
    print(f"Done in {elapsed:.1f}s -> {binary_file}")


def ensure_context_binaries(backend: str = "htp") -> tuple[Path, Path]:
    """Generate the detector/recognizer context binaries if missing. Returns
    (detector_ctx_bin, recognizer_ctx_bin)."""
    if not DETECTOR_CTX_BIN.exists():
        generate_context_binary(DETECTOR_DLC, DETECTOR_CTX_BIN, backend=backend)
    if not RECOGNIZER_CTX_BIN.exists():
        generate_context_binary(RECOGNIZER_DLC, RECOGNIZER_CTX_BIN, backend=backend)
    return DETECTOR_CTX_BIN, RECOGNIZER_CTX_BIN


if __name__ == "__main__":
    ensure_context_binaries()

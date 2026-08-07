"""Central home for every machine-specific path this pipeline needs (QAIRT
SDK install, the native-ARM64 Python env, and each model's download
directory). Every value defaults to a Path.home()-relative location matching
this repo's own install layout, and can be overridden per-machine with an
environment variable of the same name -- so a fresh checkout on a different
user account/machine only needs env vars set (if its layout differs), never
a source edit."""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()


def _path(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else default


QAIRT_ROOT = _path(
    "QAIRT_ROOT", HOME / "Downloads" / "v2.48.40.260702" / "qairt" / "2.48.40.260702"
)

QAI_ENV_ARM64_PYTHON = _path(
    "QAI_ENV_ARM64_PYTHON", HOME / "qai_env_arm64" / "Scripts" / "python.exe"
)

EASYOCR_DLC_DIR = _path(
    "EASYOCR_DLC_DIR",
    HOME / "Downloads" / "easyocr-qnn_dlc-float" / "easyocr-qnn_dlc-float",
)

EASYOCR_ONNX_DIR = _path(
    "EASYOCR_ONNX_DIR",
    HOME / "Downloads" / "ONNX" / "easyocr-onnx-w8a8" / "easyocr-onnx-w8a8",
)

MELOTTS_DIR = _path(
    "MELOTTS_DIR",
    HOME
    / "Downloads"
    / "melotts_en-voice_ai-mixed_with_float-qualcomm_snapdragon_x_elite"
    / "melotts_en-voice_ai-mixed_with_float-qualcomm_snapdragon_x_elite",
)

WHISPER_TINY_DIR = _path(
    "WHISPER_TINY_DIR",
    HOME
    / "Downloads"
    / "whisper_tiny-qnn_context_binary-float-qualcomm_snapdragon_x_elite"
    / "whisper_tiny-qnn_context_binary-float-qualcomm_snapdragon_x_elite",
)

WHISPER_BASE_ONNX_DIR = _path(
    "WHISPER_BASE_ONNX_DIR",
    HOME
    / "Downloads"
    / "ONNX"
    / "whisper_base-precompiled_qnn_onnx-float-qualcomm_snapdragon_x_elite"
    / "whisper_base-precompiled_qnn_onnx-float-qualcomm_snapdragon_x_elite",
)

GENIE_DIR = _path(
    "GENIE_DIR",
    HOME
    / "Downloads"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x_elite"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x_elite",
)

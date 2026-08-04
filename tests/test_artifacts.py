from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dragon_nest.artifacts import (
    ArtifactChecksumError,
    ArtifactNotFoundError,
    ArtifactRegistry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path, artifact_path: str, checksum: str) -> Path:
    path.write_text(
        f"""
models:
  - model_id: test-qnn
    model_version: v1
    runtime: qnn
    artifact_path: {artifact_path}
    checksum: {checksum}
    tokenizer_id: test-tokenizer
    precision: fp16
    supported_accelerators: [htp]
    min_memory_mb: 1
    max_context_tokens: 16
    supports_steering: false
    supports_data_parallel: false
    supports_layer_pipeline: false
    runtime_options:
      artifact_kind: context_binary
      outputs:
        - name: output
          shape: [1]
          dtype: float32
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_registry_validates_artifact_checksum(tmp_path: Path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"validated model")
    manifest = _manifest(
        tmp_path / "manifest.yaml",
        artifact.name,
        f"sha256:{_sha256(artifact)}",
    )

    registry = ArtifactRegistry.from_yaml(manifest)

    assert registry.validate("test-qnn") == artifact
    assert registry.is_available("test-qnn")


def test_registry_rejects_checksum_mismatch(tmp_path: Path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"wrong bytes")
    manifest = _manifest(
        tmp_path / "manifest.yaml",
        artifact.name,
        f"sha256:{'0' * 64}",
    )

    registry = ArtifactRegistry.from_yaml(manifest)

    with pytest.raises(ArtifactChecksumError, match="checksum mismatch"):
        registry.validate("test-qnn")


def test_registry_rejects_unresolved_artifact_environment(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MISSING_DRAGONNEST_MODEL", raising=False)
    manifest = _manifest(
        tmp_path / "manifest.yaml",
        "${MISSING_DRAGONNEST_MODEL}",
        f"sha256:{'0' * 64}",
    )

    registry = ArtifactRegistry.from_yaml(manifest)

    with pytest.raises(ArtifactNotFoundError, match="unresolved environment"):
        registry.validate("test-qnn")

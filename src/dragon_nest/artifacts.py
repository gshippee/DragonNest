from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import RuntimeName, SteeringMode


class ArtifactError(ValueError):
    """Base error for invalid or unavailable model artifacts."""


class ArtifactNotFoundError(ArtifactError):
    pass


class ArtifactChecksumError(ArtifactError):
    pass


@dataclass(frozen=True)
class SplitBoundary:
    pipeline_id: str
    start_layer: int | None = None
    end_layer: int | None = None
    total_layers: int = 0
    input_tensor: str = ""
    output_tensor: str = ""
    includes_embedding: bool = False
    includes_lm_head: bool = False
    boundary_format: str = "raw"
    stage_index: int = -1
    stage_count: int = 0
    transformer_start_layer: int | None = None
    transformer_end_layer: int | None = None

    def __post_init__(self) -> None:
        start = self.transformer_start_layer
        end = self.transformer_end_layer
        if start is None and self.start_layer is not None:
            start = self.start_layer
        if end is None and self.end_layer is not None:
            end = self.end_layer
        object.__setattr__(self, "transformer_start_layer", start)
        object.__setattr__(self, "transformer_end_layer", end)
        object.__setattr__(self, "start_layer", start)
        object.__setattr__(self, "end_layer", end)


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    model_version: str
    runtime: RuntimeName
    artifact_path: str
    checksum: str
    tokenizer_id: str
    precision: str
    supported_accelerators: tuple[str, ...]
    min_memory_mb: int
    max_context_tokens: int
    supports_steering: bool
    supports_data_parallel: bool
    supports_layer_pipeline: bool
    split_boundary: SplitBoundary | None = None
    runtime_options: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str = ""
    base_model: str = ""
    base_model_revision: str = ""
    tokenizer_fingerprint: str = ""
    steering_mode: SteeringMode = SteeringMode.NONE
    behavior_profile_id: str = ""
    vector_id: str = ""
    vector_calibration: Mapping[str, Any] = field(default_factory=dict)
    target_compatibility_class: str = ""
    artifact_format: str = ""
    quantization: str = ""
    context_profile: Mapping[str, Any] = field(default_factory=dict)
    input_tensor_schema: tuple[Mapping[str, Any], ...] = ()
    output_tensor_schema: tuple[Mapping[str, Any], ...] = ()
    size_bytes: int | None = None
    build_provenance: Mapping[str, Any] = field(default_factory=dict)
    ai_hub: Mapping[str, Any] = field(default_factory=dict)
    measured_hardware_results: tuple[Mapping[str, Any], ...] = ()
    verification_status: str = "unverified"


_UNEXPANDED_ENV = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class ArtifactRegistry:
    def __init__(self, artifacts: Mapping[str, ModelArtifact], base_dir: str | Path):
        self._artifacts = dict(artifacts)
        self.base_dir = Path(base_dir).resolve()
        self._validated: dict[tuple[str, bool], Path] = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ArtifactRegistry":
        manifest_path = Path(path).resolve()
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        entries = raw.get("models", [])
        if not isinstance(entries, list):
            raise ArtifactError("manifest models must be a list")
        artifacts: dict[str, ModelArtifact] = {}
        for index, item in enumerate(entries):
            if not isinstance(item, Mapping):
                raise ArtifactError(
                    f"invalid model manifest entry {index}: expected mapping"
                )
            try:
                split_raw = item.get("split_boundary")
                split = SplitBoundary(**split_raw) if split_raw else None
                supports_steering = bool(item["supports_steering"])
                steering_mode = SteeringMode(
                    item.get(
                        "steering_mode",
                        SteeringMode.RUNTIME_VECTOR.value
                        if supports_steering
                        else SteeringMode.NONE.value,
                    )
                )
                artifact = ModelArtifact(
                    model_id=str(item["model_id"]),
                    model_version=str(item["model_version"]),
                    runtime=RuntimeName(item["runtime"]),
                    artifact_path=str(item["artifact_path"]),
                    checksum=str(item["checksum"]),
                    tokenizer_id=str(item["tokenizer_id"]),
                    precision=str(item["precision"]),
                    supported_accelerators=tuple(item["supported_accelerators"]),
                    min_memory_mb=int(item["min_memory_mb"]),
                    max_context_tokens=int(item["max_context_tokens"]),
                    supports_steering=supports_steering,
                    supports_data_parallel=bool(item["supports_data_parallel"]),
                    supports_layer_pipeline=bool(item["supports_layer_pipeline"]),
                    split_boundary=split,
                    runtime_options=dict(item.get("runtime_options", {})),
                    artifact_id=str(item.get("artifact_id", item["model_id"])),
                    base_model=str(item.get("base_model", item["tokenizer_id"])),
                    base_model_revision=str(
                        item.get("base_model_revision", item["model_version"])
                    ),
                    tokenizer_fingerprint=str(item.get("tokenizer_fingerprint", "")),
                    steering_mode=steering_mode,
                    behavior_profile_id=str(item.get("behavior_profile_id", "")),
                    vector_id=str(item.get("vector_id", "")),
                    vector_calibration=dict(item.get("vector_calibration", {})),
                    target_compatibility_class=str(
                        item.get("target_compatibility_class", "")
                    ),
                    artifact_format=str(item.get("artifact_format", "")),
                    quantization=str(item.get("quantization", item["precision"])),
                    context_profile=dict(item.get("context_profile", {})),
                    input_tensor_schema=tuple(item.get("input_tensor_schema", ())),
                    output_tensor_schema=tuple(item.get("output_tensor_schema", ())),
                    size_bytes=(
                        int(item["size_bytes"])
                        if item.get("size_bytes") is not None
                        else None
                    ),
                    build_provenance=dict(item.get("build_provenance", {})),
                    ai_hub=dict(item.get("ai_hub", {})),
                    measured_hardware_results=tuple(
                        item.get("measured_hardware_results", ())
                    ),
                    verification_status=str(
                        item.get("verification_status", "unverified")
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ArtifactError(
                    f"invalid model manifest entry {index}: {exc}"
                ) from exc
            if artifact.model_id in artifacts:
                raise ArtifactError(f"duplicate model_id {artifact.model_id}")
            if any(
                existing.artifact_id == artifact.artifact_id
                for existing in artifacts.values()
            ):
                raise ArtifactError(f"duplicate artifact_id {artifact.artifact_id}")
            cls._validate_metadata(artifact)
            artifacts[artifact.model_id] = artifact
        return cls(artifacts, manifest_path.parent)

    @staticmethod
    def _validate_metadata(artifact: ModelArtifact) -> None:
        if not artifact.model_id or not artifact.model_version:
            raise ArtifactError("model_id and model_version must be non-empty")
        if not artifact.tokenizer_id:
            raise ArtifactError(f"{artifact.model_id}: tokenizer_id must be non-empty")
        if artifact.min_memory_mb < 0 or artifact.max_context_tokens <= 0:
            raise ArtifactError(f"{artifact.model_id}: invalid memory/context limits")
        if not artifact.supported_accelerators:
            raise ArtifactError(
                f"{artifact.model_id}: supported_accelerators cannot be empty"
            )
        if not artifact.artifact_id:
            raise ArtifactError(f"{artifact.model_id}: artifact_id must be non-empty")
        if artifact.size_bytes is not None and artifact.size_bytes < 0:
            raise ArtifactError(f"{artifact.model_id}: size_bytes cannot be negative")
        if (
            artifact.steering_mode == SteeringMode.RUNTIME_VECTOR
            and not artifact.supports_steering
        ):
            raise ArtifactError(
                f"{artifact.model_id}: runtime_vector requires supports_steering=true"
            )
        if (
            artifact.steering_mode != SteeringMode.RUNTIME_VECTOR
            and artifact.supports_steering
        ):
            raise ArtifactError(
                f"{artifact.model_id}: supports_steering is reserved for runtime_vector"
            )
        if (
            artifact.steering_mode in {
                SteeringMode.BAKED_PROFILE,
                SteeringMode.PROMPT_PROFILE,
            }
            and not artifact.behavior_profile_id
        ):
            raise ArtifactError(
                f"{artifact.model_id}: {artifact.steering_mode.value} requires behavior_profile_id"
            )
        split = artifact.split_boundary
        if split:
            if not artifact.supports_layer_pipeline:
                raise ArtifactError(
                    f"{artifact.model_id}: split_boundary requires supports_layer_pipeline"
                )
            if split.stage_count:
                if not (0 <= split.stage_index < split.stage_count):
                    raise ArtifactError(f"{artifact.model_id}: invalid pipeline stage index")
            elif split.stage_index >= 0:
                raise ArtifactError(
                    f"{artifact.model_id}: stage_count is required with stage_index"
                )
            start = split.transformer_start_layer
            end = split.transformer_end_layer
            if (start is None) != (end is None):
                raise ArtifactError(
                    f"{artifact.model_id}: transformer layer range must be fully specified"
                )
            if start is not None:
                ordered = start <= end if split.stage_count else start < end
                if not (
                    0 <= start
                    and ordered
                    and (split.total_layers <= 0 or end <= split.total_layers)
                ):
                    raise ArtifactError(f"{artifact.model_id}: invalid split layer range")
            if start is None and not split.includes_embedding:
                raise ArtifactError(
                    f"{artifact.model_id}: a layerless stage must own embeddings"
                )
            if not split.input_tensor or not split.output_tensor:
                raise ArtifactError(f"{artifact.model_id}: split tensors must be named")

    def get(self, model_id: str) -> ModelArtifact:
        try:
            return self._artifacts[model_id]
        except KeyError as exc:
            raise ArtifactNotFoundError(f"unknown model artifact {model_id}") from exc

    def all(self) -> tuple[ModelArtifact, ...]:
        return tuple(self._artifacts.values())

    def resolve_path(self, artifact: ModelArtifact) -> Path:
        expanded = os.path.expandvars(os.path.expanduser(artifact.artifact_path))
        if _UNEXPANDED_ENV.search(expanded):
            raise ArtifactNotFoundError(
                f"{artifact.model_id}: unresolved environment variable in artifact_path"
            )
        path = Path(expanded)
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve()

    def validate(self, model_id: str, verify_checksum: bool = True) -> Path:
        cache_key = (model_id, verify_checksum)
        if cache_key in self._validated:
            return self._validated[cache_key]
        artifact = self.get(model_id)
        path = self.resolve_path(artifact)
        if not path.exists():
            raise ArtifactNotFoundError(f"{model_id}: artifact does not exist: {path}")
        if verify_checksum:
            self._verify_checksum(artifact, path)
        self._validated[cache_key] = path
        return path

    def is_available(self, model_id: str, verify_checksum: bool = True) -> bool:
        try:
            self.validate(model_id, verify_checksum=verify_checksum)
        except ArtifactError:
            return False
        return True

    @staticmethod
    def _verify_checksum(artifact: ModelArtifact, path: Path) -> None:
        checksum = os.path.expandvars(artifact.checksum)
        if _UNEXPANDED_ENV.search(checksum):
            raise ArtifactChecksumError(
                f"{artifact.model_id}: unresolved environment variable in checksum"
            )
        try:
            algorithm, expected = checksum.split(":", 1)
        except ValueError as exc:
            raise ArtifactChecksumError(
                f"{artifact.model_id}: checksum must use sha256:<hex> or sha256-tree:<hex>"
            ) from exc
        if not _SHA256.fullmatch(expected):
            raise ArtifactChecksumError(
                f"{artifact.model_id}: invalid SHA-256 checksum"
            )
        if algorithm == "sha256" and path.is_file():
            actual = _hash_file(path)
        elif algorithm == "sha256-tree" and path.is_dir():
            actual = _hash_tree(path)
        else:
            raise ArtifactChecksumError(
                f"{artifact.model_id}: checksum type {algorithm!r} does not match {path}"
            )
        if actual.lower() != expected.lower():
            raise ArtifactChecksumError(
                f"{artifact.model_id}: checksum mismatch; expected {expected}, got {actual}"
            )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def calculate_checksum(path: str | Path) -> str:
    artifact_path = Path(path)
    if artifact_path.is_file():
        return f"sha256:{_hash_file(artifact_path)}"
    if artifact_path.is_dir():
        return f"sha256-tree:{_hash_tree(artifact_path)}"
    raise ArtifactNotFoundError(f"artifact does not exist: {artifact_path}")

"""Executable artifacts and their per-device deployment states.

The scheduler routes executable deployments — one device paired with one
concrete artifact — not abstract models and not raw free memory. ArtifactSpec
is the immutable description of an executable artifact (or pipeline stage);
DeploymentState is the live relationship between one device and one artifact;
DeploymentIndex derives those states from device advertisements, heartbeat
warm lists, and simulation overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from .models import Device
from .registry import DeviceRecord


class ArtifactState(StrEnum):
    ABSENT = "absent"
    AVAILABLE_REMOTE = "available_remote"
    DOWNLOADING = "downloading"
    INSTALLED = "installed"
    LOADING = "loading"
    WARM = "warm"
    FAILED = "failed"
    QUARANTINED = "quarantined"


# Deployment states from which an artifact can serve a request right now.
EXECUTABLE_STATES = frozenset({ArtifactState.WARM, ArtifactState.INSTALLED})


@dataclass(frozen=True)
class ArtifactSpec:
    """Immutable executable model artifact or pipeline stage."""

    artifact_id: str
    base_model_id: str
    base_model_family: str
    model_version: str
    behavior_profile_id: str = ""  # non-empty => statically baked behavior
    steering_realization: str = "none"  # how behavior is realized IN the artifact
    compatibility_classes: tuple[str, ...] = ("mock",)
    runtime: str = "mock"
    runtime_version: str = ""
    quantization: str = "none"
    max_context_tokens: int = 4096
    topology: str = "full_model"  # full_model | pipeline_stage
    start_layer: int = -1
    end_layer: int = -1
    boundary_schema: str = ""
    checksum: str = ""
    artifact_size_mb: int = 0
    estimated_memory_mb: int = 0
    measured_memory_mb: int = 0
    measured_load_time_ms: int = 0
    prefill_tokens_per_s: float = 0.0
    decode_tokens_per_s: float = 0.0
    kv_cache_bytes_per_token: int = 0
    build_provenance: str = ""
    readiness: str = "ready"  # ready | unvalidated

    def memory_mb(self) -> tuple[int, bool]:
        """Resident memory in MB and whether the value is an estimate."""
        if self.measured_memory_mb > 0:
            return self.measured_memory_mb, False
        return self.estimated_memory_mb, True

    def supports_context(self, tokens: int) -> bool:
        return tokens <= self.max_context_tokens


@dataclass(frozen=True)
class DeploymentState:
    device_id: str
    artifact_id: str
    state: ArtifactState
    resident_bytes: int = 0
    measured_prefill_tokens_per_s: float = 0.0
    measured_decode_tokens_per_s: float = 0.0


class ArtifactCatalog:
    def __init__(self, artifacts: dict[str, ArtifactSpec]):
        self._artifacts = artifacts

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ArtifactCatalog":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        artifacts: dict[str, ArtifactSpec] = {}
        for item in raw.get("artifacts", []):
            spec = ArtifactSpec(
                artifact_id=str(item["artifact_id"]),
                base_model_id=str(item.get("base_model_id", item["artifact_id"])),
                base_model_family=str(item["base_model_family"]),
                model_version=str(item.get("model_version", "")),
                behavior_profile_id=str(item.get("behavior_profile_id", "")),
                steering_realization=str(item.get("steering_realization", "none")),
                compatibility_classes=tuple(
                    item.get("compatibility_classes", ["mock"])
                ),
                runtime=str(item.get("runtime", "mock")),
                runtime_version=str(item.get("runtime_version", "")),
                quantization=str(item.get("quantization", "none")),
                max_context_tokens=int(item.get("max_context_tokens", 4096)),
                topology=str(item.get("topology", "full_model")),
                start_layer=int(item.get("start_layer", -1)),
                end_layer=int(item.get("end_layer", -1)),
                boundary_schema=str(item.get("boundary_schema", "")),
                checksum=str(item.get("checksum", "")),
                artifact_size_mb=int(item.get("artifact_size_mb", 0)),
                estimated_memory_mb=int(item.get("estimated_memory_mb", 0)),
                measured_memory_mb=int(item.get("measured_memory_mb", 0)),
                measured_load_time_ms=int(item.get("measured_load_time_ms", 0)),
                prefill_tokens_per_s=float(item.get("prefill_tokens_per_s", 0.0)),
                decode_tokens_per_s=float(item.get("decode_tokens_per_s", 0.0)),
                kv_cache_bytes_per_token=int(
                    item.get("kv_cache_bytes_per_token", 0)
                ),
                build_provenance=str(item.get("build_provenance", "")),
                readiness=str(item.get("readiness", "ready")),
            )
            artifacts[spec.artifact_id] = spec
        return cls(artifacts)

    def get(self, artifact_id: str) -> ArtifactSpec:
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"unknown artifact {artifact_id}") from exc

    def maybe_get(self, artifact_id: str) -> ArtifactSpec | None:
        return self._artifacts.get(artifact_id)

    def all(self) -> tuple[ArtifactSpec, ...]:
        return tuple(self._artifacts[key] for key in sorted(self._artifacts))

    def baked_for(
        self, profile_id: str, base_model_family: str
    ) -> tuple[ArtifactSpec, ...]:
        return tuple(
            artifact
            for artifact in self.all()
            if artifact.behavior_profile_id == profile_id
            and artifact.base_model_family == base_model_family
        )

    def full_models(self, base_model_family: str) -> tuple[ArtifactSpec, ...]:
        return tuple(
            artifact
            for artifact in self.all()
            if artifact.topology == "full_model"
            and artifact.base_model_family == base_model_family
        )


_SOC_CLASS_MARKERS: tuple[tuple[str, str], ...] = (
    ("x elite", "snapdragon-x-elite"),
    ("x1e", "snapdragon-x-elite"),
    ("8 elite", "snapdragon-8-elite"),
    ("sm8750", "snapdragon-8-elite"),
)


def device_compatibility_classes(device: Device) -> tuple[str, ...]:
    """Static compatibility classes a device can execute.

    Portable mock artifacts run everywhere; Snapdragon-specific classes are
    derived from the agent-reported SoC identity, never guessed from brand
    names elsewhere in the inventory.
    """
    classes = ["mock"]
    soc = device.hardware.soc_model.lower()
    for marker, compatibility_class in _SOC_CLASS_MARKERS:
        if marker in soc and compatibility_class not in classes:
            classes.append(compatibility_class)
    return tuple(classes)


class DeploymentIndex:
    """Per-(device, artifact) deployment states for the scheduler."""

    def __init__(self, states: dict[tuple[str, str], DeploymentState]):
        self._states = states

    @classmethod
    def build(
        cls,
        records: Iterable[DeviceRecord],
        catalog: ArtifactCatalog,
        overrides: Mapping[tuple[str, str], ArtifactState],
    ) -> "DeploymentIndex":
        states: dict[tuple[str, str], DeploymentState] = {}
        for record in records:
            device = record.device
            advertised = {model.model_id: model for model in device.models}
            warm_ids = set(record.warm_model_ids) | {
                model.model_id for model in device.models if model.warm
            }
            for artifact in catalog.all():
                key = (device.device_id, artifact.artifact_id)
                if artifact.artifact_id in advertised:
                    state = (
                        ArtifactState.WARM
                        if artifact.artifact_id in warm_ids
                        else ArtifactState.INSTALLED
                    )
                else:
                    state = ArtifactState.ABSENT
                override = overrides.get(key)
                if override is not None:
                    state = override
                memory_mb, _ = artifact.memory_mb()
                states[key] = DeploymentState(
                    device_id=device.device_id,
                    artifact_id=artifact.artifact_id,
                    state=state,
                    resident_bytes=(
                        memory_mb * 1024 * 1024
                        if state == ArtifactState.WARM
                        else 0
                    ),
                )
        return cls(states)

    def state(self, device_id: str, artifact_id: str) -> DeploymentState:
        return self._states.get(
            (device_id, artifact_id),
            DeploymentState(device_id, artifact_id, ArtifactState.ABSENT),
        )

    def for_device(self, device_id: str) -> tuple[DeploymentState, ...]:
        return tuple(
            state
            for (dev, _), state in sorted(self._states.items())
            if dev == device_id
        )

    def warm_artifacts(self, device_id: str) -> tuple[str, ...]:
        return tuple(
            state.artifact_id
            for state in self.for_device(device_id)
            if state.state == ArtifactState.WARM
        )

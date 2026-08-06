from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..artifacts import (
    ArtifactChecksumError,
    ArtifactError,
    ArtifactRegistry,
    ModelArtifact,
    calculate_checksum,
)
from ..executors import ExecutorDispatcher
from ..models import ExecutionPlan, SteeringMode, TaskResult
from ..telemetry import PlatformTelemetry, TelemetrySnapshot


class RuntimeSteeringUnavailableError(ArtifactError):
    """Raised instead of silently treating a prompt/baked profile as a vector."""


@dataclass(frozen=True)
class ArtifactLoadResult:
    artifact_id: str
    load_time_ms: int
    warm: bool
    detail: str


@dataclass(frozen=True)
class RuntimeCapabilities:
    device_compatibility_key: str
    runtime_name: str
    runtime_version: str
    accelerator_available: bool
    supported_steering_modes: tuple[str, ...]
    installed_artifact_ids: tuple[str, ...]
    warm_artifact_ids: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeHealth:
    telemetry: TelemetrySnapshot
    installed_artifact_ids: tuple[str, ...]
    warm_artifact_ids: tuple[str, ...]
    artifact_load_time_ms: dict[str, int]


class HardwareRuntimeAdapter:
    """Lifecycle boundary around DragonNest's existing runtime executors.

    The adapter deliberately consumes ``ExecutionPlan`` and delegates to
    ``ExecutorDispatcher`` so it does not create a second control protocol.
    A runtime is called warm only when its manifest declares persistent-load
    support and a real load hook succeeds.  The current Genie CLI and Android
    dialog-per-request paths therefore remain installed-but-cold.
    """

    def __init__(
        self,
        artifacts: ArtifactRegistry,
        *,
        compatibility_key: str,
        runtime_name: str,
        runtime_version: str,
        accelerator_available: bool,
        telemetry: PlatformTelemetry,
        artifact_store: str | Path,
        dispatcher: ExecutorDispatcher | None = None,
        load_hook: Callable[[ModelArtifact, Path], None] | None = None,
        unload_hook: Callable[[ModelArtifact], None] | None = None,
    ):
        self.artifacts = artifacts
        self.compatibility_key = compatibility_key
        self.runtime_name = runtime_name
        self.runtime_version = runtime_version
        self.accelerator_available = accelerator_available
        self.telemetry_source = telemetry
        self.artifact_store = Path(artifact_store).resolve()
        self.dispatcher = dispatcher or ExecutorDispatcher(artifacts)
        self.qnn = self.dispatcher.qnn
        self.load_hook = load_hook
        self.unload_hook = unload_hook
        self._installed: set[str] = set()
        self._warm: set[str] = set()
        self._load_times_ms: dict[str, int] = {}
        self._attempts: dict[str, asyncio.Task[TaskResult]] = {}
        self._discover_installed()

    def _discover_installed(self) -> None:
        for artifact in self.artifacts.all():
            if self._compatible(artifact) and self.artifacts.is_available(
                artifact.model_id
            ):
                self._installed.add(artifact.artifact_id)

    def _compatible(self, artifact: ModelArtifact) -> bool:
        return not artifact.target_compatibility_class or (
            artifact.target_compatibility_class == self.compatibility_key
        )

    def validate_artifact(self, manifest: ModelArtifact | str) -> Path:
        artifact = (
            self.artifacts.get(manifest) if isinstance(manifest, str) else manifest
        )
        if not self._compatible(artifact):
            raise ArtifactError(
                f"{artifact.artifact_id}: target {artifact.target_compatibility_class!r} "
                f"does not match device {self.compatibility_key!r}"
            )
        path = self.artifacts.validate(artifact.model_id)
        if artifact.size_bytes is not None:
            actual_size = _artifact_size(path)
            if actual_size != artifact.size_bytes:
                raise ArtifactError(
                    f"{artifact.artifact_id}: size mismatch; expected "
                    f"{artifact.size_bytes}, got {actual_size}"
                )
        return path

    def install_artifact(self, source: str | Path, checksum: str) -> Path:
        """Copy a checksummed source into the local store without overwriting.

        The manifest is still the authority for whether the resulting bytes are
        schedulable.  Installation is content-addressed and does not download
        from AI Hub or interpret a remote URL.
        """

        source_path = Path(source).resolve()
        actual = calculate_checksum(source_path)
        if actual.lower() != checksum.lower():
            raise ArtifactChecksumError(
                f"install source checksum mismatch; expected {checksum}, got {actual}"
            )
        algorithm, digest = actual.split(":", 1)
        destination = self.artifact_store / digest
        if destination.exists():
            if calculate_checksum(destination).lower() != actual.lower():
                raise ArtifactChecksumError(
                    f"existing content-addressed artifact is corrupt: {destination}"
                )
            return destination

        self.artifact_store.mkdir(parents=True, exist_ok=True)
        temporary = self.artifact_store / f".{digest}.{uuid.uuid4().hex}.tmp"
        try:
            if algorithm == "sha256-tree":
                shutil.copytree(source_path, temporary)
            else:
                shutil.copy2(source_path, temporary)
            if calculate_checksum(temporary).lower() != actual.lower():
                raise ArtifactChecksumError("installed copy failed checksum verification")
            os.replace(temporary, destination)
        finally:
            if temporary.is_dir():
                shutil.rmtree(temporary, ignore_errors=True)
            elif temporary.exists():
                temporary.unlink(missing_ok=True)
        return destination

    def load_artifact(self, artifact_id: str) -> ArtifactLoadResult:
        artifact = self._artifact_by_id(artifact_id)
        start = time.perf_counter()
        path = self.validate_artifact(artifact)
        self._installed.add(artifact.artifact_id)
        persistent = bool(
            artifact.runtime_options.get("persistent_load_supported", False)
        )
        if persistent and self.load_hook is None:
            raise ArtifactError(
                f"{artifact.artifact_id}: persistent load declared without load hook"
            )
        if persistent:
            assert self.load_hook is not None
            self.load_hook(artifact, path)
            self._warm.add(artifact.artifact_id)
        elapsed = max(1, round((time.perf_counter() - start) * 1000))
        self._load_times_ms[artifact.artifact_id] = elapsed
        detail = (
            "runtime context loaded and retained"
            if persistent
            else "artifact validated; runtime has no persistent-load API"
        )
        return ArtifactLoadResult(artifact.artifact_id, elapsed, persistent, detail)

    def unload_artifact(self, artifact_id: str) -> None:
        artifact = self._artifact_by_id(artifact_id)
        if artifact.artifact_id in self._warm and self.unload_hook is not None:
            self.unload_hook(artifact)
        self._warm.discard(artifact.artifact_id)

    async def execute(
        self, request: ExecutionPlan, attempt_id: str | None = None
    ) -> TaskResult:
        key = attempt_id or request.task_id
        task = asyncio.create_task(self.dispatcher.execute(request))
        self._attempts[key] = task
        try:
            return await task
        finally:
            self._attempts.pop(key, None)

    async def execute_runtime_steered(
        self, request: ExecutionPlan, vector: object, attempt_id: str | None = None
    ) -> TaskResult:
        del vector  # The existing ExecutionPlan carries only a registered vector ID.
        artifact = self._artifact_for_plan(request)
        if artifact.steering_mode != SteeringMode.RUNTIME_VECTOR:
            raise RuntimeSteeringUnavailableError(
                f"{artifact.artifact_id}: steering mode is "
                f"{artifact.steering_mode.value}, not runtime_vector"
            )
        if not request.steering.enabled:
            raise RuntimeSteeringUnavailableError(
                "runtime-steered execution requires an enabled SteeringSpec"
            )
        return await self.execute(request, attempt_id=attempt_id)

    def cancel(self, attempt_id: str) -> bool:
        task = self._attempts.get(attempt_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def health(self) -> RuntimeHealth:
        return RuntimeHealth(
            telemetry=self.telemetry_source.sample(),
            installed_artifact_ids=tuple(sorted(self._installed)),
            warm_artifact_ids=tuple(sorted(self._warm)),
            artifact_load_time_ms=dict(self._load_times_ms),
        )

    def capabilities(self) -> RuntimeCapabilities:
        modes = {SteeringMode.NONE.value}
        for artifact in self.artifacts.all():
            if artifact.artifact_id in self._installed:
                modes.add(artifact.steering_mode.value)
        return RuntimeCapabilities(
            device_compatibility_key=self.compatibility_key,
            runtime_name=self.runtime_name,
            runtime_version=self.runtime_version,
            accelerator_available=self.accelerator_available,
            supported_steering_modes=tuple(sorted(modes)),
            installed_artifact_ids=tuple(sorted(self._installed)),
            warm_artifact_ids=tuple(sorted(self._warm)),
        )

    def _artifact_by_id(self, artifact_id: str) -> ModelArtifact:
        for artifact in self.artifacts.all():
            if artifact.artifact_id == artifact_id or artifact.model_id == artifact_id:
                return artifact
        raise ArtifactError(f"unknown artifact {artifact_id}")

    def _artifact_for_plan(self, plan: ExecutionPlan) -> ModelArtifact:
        if plan.tasks:
            return self.artifacts.get(plan.tasks[0].selected_model_id)
        if plan.stages:
            return self.artifacts.get(plan.stages[0].selected_model_id)
        raise ArtifactError("execution plan does not select an artifact")


class AdapterTelemetry:
    """Expose adapter warm state through DragonNest's existing heartbeat."""

    def __init__(self, adapter: HardwareRuntimeAdapter):
        self.adapter = adapter

    def sample(self) -> TelemetrySnapshot:
        snapshot = self.adapter.telemetry_source.sample()
        return TelemetrySnapshot(
            health=snapshot.health,
            active_task_ids=snapshot.active_task_ids,
            warm_model_ids=self.adapter.capabilities().warm_artifact_ids,
            simulated_constraint=snapshot.simulated_constraint,
        )


def _artifact_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())

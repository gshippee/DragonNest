"""Artifact/profile provisioning state machine.

Tracks building, validating, and deploying a behavior-profile artifact for a
target device. The control-plane contract is real; the compile step sits
behind a ProvisioningAdapter so a real Qualcomm AI Hub integration can
replace MockAiHubAdapter without touching the state machine. The mock
adapter prefixes every detail string with "[mock]" so no UI surface can
claim a real compilation happened unless a real adapter performed it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Protocol

from .deployments import ArtifactState


class ProvisioningState(StrEnum):
    MISSING = "missing"
    BUILD_QUEUED = "build_queued"
    COMPILING = "compiling"
    VALIDATING = "validating"
    READY_REMOTE = "ready_remote"
    DOWNLOADING = "downloading"
    INSTALLED = "installed"
    WARM = "warm"
    FAILED = "failed"


_CHAIN: dict[ProvisioningState, ProvisioningState] = {
    ProvisioningState.MISSING: ProvisioningState.BUILD_QUEUED,
    ProvisioningState.BUILD_QUEUED: ProvisioningState.COMPILING,
    ProvisioningState.COMPILING: ProvisioningState.VALIDATING,
    ProvisioningState.VALIDATING: ProvisioningState.READY_REMOTE,
    ProvisioningState.READY_REMOTE: ProvisioningState.DOWNLOADING,
    ProvisioningState.DOWNLOADING: ProvisioningState.INSTALLED,
    ProvisioningState.INSTALLED: ProvisioningState.WARM,
}

_TERMINAL = frozenset({ProvisioningState.WARM, ProvisioningState.FAILED})


@dataclass
class ProvisioningJob:
    job_id: str
    profile_id: str
    target_device_id: str
    artifact_id: str
    state: ProvisioningState
    adapter_name: str
    detail: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: list[tuple[str, str]] = field(default_factory=list)

    def record(self, state: ProvisioningState, detail: str) -> None:
        self.state = state
        self.detail = detail
        self.updated_at = time.time()
        self.history.append((state.value, detail))


class ProvisioningAdapter(Protocol):
    name: str

    def advance(self, job: ProvisioningJob) -> str:
        """Perform the work for the job's next state; returns a detail line."""
        ...


class MockAiHubAdapter:
    """Deterministic stand-in for a real Qualcomm AI Hub build pipeline.

    Every detail line is prefixed "[mock]" — nothing downstream may present
    these steps as a real compilation.
    """

    name = "mock-aihub"

    _DETAILS: dict[ProvisioningState, str] = {
        ProvisioningState.BUILD_QUEUED: "[mock] build request queued",
        ProvisioningState.COMPILING: "[mock] simulating AI Hub compile job",
        ProvisioningState.VALIDATING: "[mock] simulating numerics validation",
        ProvisioningState.READY_REMOTE: "[mock] artifact available remotely",
        ProvisioningState.DOWNLOADING: "[mock] simulating device download",
        ProvisioningState.INSTALLED: "[mock] artifact installed on device",
        ProvisioningState.WARM: "[mock] artifact loaded and warm",
    }

    def advance(self, job: ProvisioningJob) -> str:
        next_state = _CHAIN[job.state]
        return self._DETAILS[next_state]


OnDeployed = Callable[[str, str, ArtifactState], None]


class ProvisioningManager:
    """Demo-scoped, in-memory job store; jobs do not survive Brain restarts."""

    def __init__(
        self,
        adapter: ProvisioningAdapter,
        on_deployed: OnDeployed | None = None,
    ):
        self.adapter = adapter
        self.on_deployed = on_deployed
        self._jobs: dict[str, ProvisioningJob] = {}
        self._order: list[str] = []

    def start(
        self, profile_id: str, target_device_id: str, artifact_id: str
    ) -> ProvisioningJob:
        job = ProvisioningJob(
            job_id=f"prov-{uuid.uuid4().hex[:8]}",
            profile_id=profile_id,
            target_device_id=target_device_id,
            artifact_id=artifact_id,
            state=ProvisioningState.MISSING,
            adapter_name=self.adapter.name,
        )
        job.history.append((job.state.value, "profile has no deployable artifact"))
        self._jobs[job.job_id] = job
        self._order.append(job.job_id)
        return job

    def advance(self, job_id: str) -> ProvisioningJob:
        job = self.get(job_id)
        if job.state in _TERMINAL:
            raise ValueError(f"job {job_id} is terminal ({job.state.value})")
        next_state = _CHAIN[job.state]
        detail = self.adapter.advance(job)
        job.record(next_state, detail)
        if next_state == ProvisioningState.INSTALLED and self.on_deployed:
            self.on_deployed(
                job.target_device_id, job.artifact_id, ArtifactState.INSTALLED
            )
        if next_state == ProvisioningState.WARM and self.on_deployed:
            self.on_deployed(
                job.target_device_id, job.artifact_id, ArtifactState.WARM
            )
        return job

    def fail(self, job_id: str, detail: str) -> ProvisioningJob:
        job = self.get(job_id)
        if job.state in _TERMINAL:
            raise ValueError(f"job {job_id} is terminal ({job.state.value})")
        job.record(ProvisioningState.FAILED, detail)
        return job

    def tick_all(self) -> tuple[ProvisioningJob, ...]:
        advanced = []
        for job_id in self._order:
            if self._jobs[job_id].state not in _TERMINAL:
                advanced.append(self.advance(job_id))
        return tuple(advanced)

    def get(self, job_id: str) -> ProvisioningJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"unknown provisioning job {job_id}") from exc

    def jobs(self) -> tuple[ProvisioningJob, ...]:
        return tuple(self._jobs[job_id] for job_id in self._order)

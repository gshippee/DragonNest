from __future__ import annotations

import pytest

from dragon_nest.deployments import ArtifactState
from dragon_nest.provisioning import (
    MockAiHubAdapter,
    ProvisioningManager,
    ProvisioningState,
)


def _manager(deployed: list | None = None) -> ProvisioningManager:
    def on_deployed(device_id: str, artifact_id: str, state: ArtifactState) -> None:
        if deployed is not None:
            deployed.append((device_id, artifact_id, state))

    return ProvisioningManager(MockAiHubAdapter(), on_deployed=on_deployed)


def test_job_walks_the_full_legal_chain_to_warm():
    deployed: list = []
    manager = _manager(deployed)
    job = manager.start("family-assistant", "x-elite-01", "family-assistant-v0-baked")
    assert job.state == ProvisioningState.MISSING

    seen = [job.state]
    for _ in range(20):
        job = manager.advance(job.job_id)
        seen.append(job.state)
        if job.state == ProvisioningState.WARM:
            break

    assert seen == [
        ProvisioningState.MISSING,
        ProvisioningState.BUILD_QUEUED,
        ProvisioningState.COMPILING,
        ProvisioningState.VALIDATING,
        ProvisioningState.READY_REMOTE,
        ProvisioningState.DOWNLOADING,
        ProvisioningState.INSTALLED,
        ProvisioningState.WARM,
    ]
    assert (
        "x-elite-01",
        "family-assistant-v0-baked",
        ArtifactState.INSTALLED,
    ) in deployed
    assert (
        "x-elite-01",
        "family-assistant-v0-baked",
        ArtifactState.WARM,
    ) in deployed


def test_advancing_a_terminal_job_is_rejected():
    manager = _manager()
    job = manager.start("p", "d", "a")
    while job.state != ProvisioningState.WARM:
        job = manager.advance(job.job_id)
    with pytest.raises(ValueError, match="terminal"):
        manager.advance(job.job_id)


def test_job_can_fail_from_an_active_state():
    manager = _manager()
    job = manager.start("p", "d", "a")
    manager.advance(job.job_id)  # build_queued
    failed = manager.fail(job.job_id, "mock compile error")
    assert failed.state == ProvisioningState.FAILED
    with pytest.raises(ValueError, match="terminal"):
        manager.advance(job.job_id)


def test_mock_adapter_never_claims_real_compilation():
    manager = _manager()
    job = manager.start("p", "d", "a")
    for _ in range(7):
        job = manager.advance(job.job_id)
    assert all(detail.startswith("[mock]") for _, detail in job.history[1:])
    assert job.adapter_name == "mock-aihub"


def test_jobs_listing_is_stable():
    manager = _manager()
    first = manager.start("p1", "d", "a1")
    second = manager.start("p2", "d", "a2")
    listed = manager.jobs()
    assert [job.job_id for job in listed] == [first.job_id, second.job_id]

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dragon_nest.behavior import BehaviorFallbackPolicy, BehaviorProfileRegistry
from dragon_nest.models import (
    ExecutionMode,
    ExecutionPlan,
    PlannedTask,
    SteeringMode,
    SteeringSpec,
)
from dragon_nest.transport.brain import BrainService


ROOT = Path(__file__).resolve().parents[1]
BEHAVIORS = ROOT / "configs/behavior-profiles.yaml"


def _plan(realization: str, behavior_profile_id: str = "detailed") -> ExecutionPlan:
    return ExecutionPlan(
        task_id="task-1",
        execution_mode=ExecutionMode.SINGLE,
        request_text="Say hello.",
        tasks=(PlannedTask(shard_id="shard-0", request_text="Say hello."),),
        steering=SteeringSpec(
            enabled=True,
            vector_id="concise-vs-verbose-layer-7",
            target_layer=7,
            alpha=8.0,
            mode=SteeringMode.RUNTIME_VECTOR.value,
        ),
        behavior_profile_id=behavior_profile_id,
        profile_realization=realization,
    )


@pytest.fixture
def service() -> BrainService:
    return BrainService(behavior_registry=BehaviorProfileRegistry.from_yaml(BEHAVIORS))


def test_steered_profiles_admit_an_unsteered_last_resort():
    """The demo control that simulates memory pressure is meant to show the
    scheduler moving work, not to make styled requests unanswerable."""
    registry = BehaviorProfileRegistry.from_yaml(BEHAVIORS)
    for profile_id in ("concise", "detailed"):
        profile = registry.get(profile_id)
        assert profile.fallback_policy == BehaviorFallbackPolicy.ALLOW_UNSTEERED
        assert SteeringMode.NONE.value in [
            mode.value for mode in profile.allowed_modes()
        ]


def test_degraded_plan_drops_steering_and_reports_none(service: BrainService):
    degraded = service._unsteered_fallback_plan(
        _plan(SteeringMode.RUNTIME_VECTOR.value), "detailed"
    )
    assert degraded is not None
    assert degraded.profile_realization == SteeringMode.NONE.value
    # Clearing the spec is what widens the candidate set: the router refuses to
    # serve an unsteered request from a baked or steer-only bundle, so this
    # lands on a genuine base model rather than borrowing a styled one.
    assert degraded.steering.enabled is False
    assert degraded.steering.vector_id == ""
    assert degraded.steering.behavior_profile_id == ""
    # Must not still look like a baked request, or the router would go hunting
    # for a baked artifact instead of a base model.
    assert degraded.steering.mode != SteeringMode.BAKED_PROFILE.value
    # The rest of the request must survive the downgrade untouched.
    assert degraded.task_id == "task-1"
    assert degraded.request_text == "Say hello."
    assert degraded.behavior_profile_id == "detailed"


def test_a_baked_realization_also_degrades(service: BrainService):
    degraded = service._unsteered_fallback_plan(
        _plan(SteeringMode.BAKED_PROFILE.value), "concise"
    )
    assert degraded is not None
    assert degraded.profile_realization == SteeringMode.NONE.value


def test_an_already_unsteered_plan_is_not_degraded_again(service: BrainService):
    assert service._unsteered_fallback_plan(
        _plan(SteeringMode.NONE.value), "detailed"
    ) is None


def test_a_profile_that_forbids_downgrade_still_fails_closed(service: BrainService):
    """Degrading is authorized by the profile, never inferred. A profile whose
    policy is exact_only or reject must not be quietly answered by a model that
    does not carry its behavior."""
    registry = BehaviorProfileRegistry.from_yaml(BEHAVIORS)
    strict = [
        profile.profile_id
        for profile in registry.all()
        if profile.fallback_policy
        in {BehaviorFallbackPolicy.EXACT_ONLY, BehaviorFallbackPolicy.REJECT}
        and profile.profile_id != "balanced"
    ]
    assert strict, "expected at least one strict profile to guard this rule"
    for profile_id in strict:
        assert service._unsteered_fallback_plan(
            _plan(SteeringMode.RUNTIME_VECTOR.value, profile_id), profile_id
        ) is None


def test_an_unknown_profile_does_not_degrade(service: BrainService):
    assert service._unsteered_fallback_plan(
        _plan(SteeringMode.RUNTIME_VECTOR.value, "nonexistent"), "nonexistent"
    ) is None


def test_without_a_behavior_registry_nothing_degrades():
    bare = BrainService()
    assert bare._unsteered_fallback_plan(
        _plan(SteeringMode.RUNTIME_VECTOR.value), "detailed"
    ) is None

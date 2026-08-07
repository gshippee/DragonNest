from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from dragon_nest.behavior import (
    BehaviorFallbackPolicy,
    BehaviorProfile,
    BehaviorProfileRegistry,
    SteeringRealization,
    SteeringRealizationMode,
)
from dragon_nest.deployments import ArtifactCatalog, ArtifactState, DeploymentIndex
from dragon_nest.models import (
    Device,
    HardwareInventory,
    HealthState,
    ModelCapability,
)
from dragon_nest.registry import DeviceRegistry
from dragon_nest.scheduler import DeploymentScheduler, RequestSpec
from dragon_nest.steering import SteeringRegistry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog() -> ArtifactCatalog:
    return ArtifactCatalog.from_yaml(ROOT / "configs/artifact-catalog.yaml")


@pytest.fixture(scope="module")
def behaviors() -> BehaviorProfileRegistry:
    return BehaviorProfileRegistry.from_yaml(ROOT / "configs/behavior-profiles.yaml")


@pytest.fixture(scope="module")
def steering() -> SteeringRegistry:
    return SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")


@pytest.fixture()
def scheduler(catalog, behaviors, steering) -> DeploymentScheduler:
    return DeploymentScheduler(catalog, behaviors, steering)


@pytest.fixture()
def hybrid_scheduler(catalog, steering) -> DeploymentScheduler:
    """Test-only profile retaining generic runtime-to-baked fallback coverage."""
    profile = BehaviorProfile(
        profile_id="concise",
        display_name="Hybrid test profile",
        description="",
        base_model_family="mock",
        version="test",
        fallback_policy=BehaviorFallbackPolicy.ALLOW_BAKED_EQUIVALENT,
        realizations=(
            SteeringRealization(
                mode=SteeringRealizationMode.RUNTIME_VECTOR,
                vector_id="concise-vs-verbose-layer-7",
                alpha=-2.0,
                alpha_min=-4.0,
                alpha_max=4.0,
                injection_layer=7,
            ),
            SteeringRealization(
                mode=SteeringRealizationMode.BAKED_PROFILE,
                baked_artifact_id="small-chat-v1-concise-baked",
            ),
        ),
    )
    return DeploymentScheduler(
        catalog,
        BehaviorProfileRegistry({"concise": profile}),
        steering,
    )


def _capability(
    model_id: str,
    warm: bool = True,
    supports_steering: bool = False,
) -> ModelCapability:
    return ModelCapability(
        model_id=model_id,
        model_family="mock",
        role="general",
        task_classes=("chat_qa", "reasoning_analysis", "summarization"),
        max_context_tokens=8192,
        warm=warm,
        quality_score=0.7,
        steering_vector_ids=(
            ("concise-vs-verbose-layer-7", "friendly-warmth-layer-7")
            if supports_steering
            else ()
        ),
        supported_steering_layers=(7,) if supports_steering else (),
        supports_steering=supports_steering,
    )


def _laptop(models, available_memory_mb=8192, thermal=0.2, rtt=6.0) -> Device:
    return Device(
        device_id="x-elite-01",
        display_name="Snapdragon X Elite Laptop",
        device_type="pc",
        platform="windows",
        total_memory_mb=32768,
        health=HealthState(
            battery_pct=95,
            charging=True,
            thermal_level=thermal,
            cpu_utilization=0.2,
            accelerator_utilization=0.1,
            available_memory_mb=available_memory_mb,
            network_rtt_ms=rtt,
        ),
        models=tuple(models),
        hardware=HardwareInventory(soc_model="Snapdragon X Elite X1E-80-100"),
    )


def _phone(models, available_memory_mb=6144, thermal=0.3, rtt=18.0, battery=80.0) -> Device:
    return Device(
        device_id="s25-ultra-01",
        display_name="Galaxy S25 Ultra",
        device_type="phone",
        platform="android",
        total_memory_mb=12288,
        health=HealthState(
            battery_pct=battery,
            charging=False,
            thermal_level=thermal,
            cpu_utilization=0.3,
            accelerator_utilization=0.2,
            available_memory_mb=available_memory_mb,
            network_rtt_ms=rtt,
        ),
        models=tuple(models),
        hardware=HardwareInventory(
            soc_model="Qualcomm SM8750 Snapdragon 8 Elite", npu_status="available"
        ),
    )


def _plan(scheduler, catalog, devices, request, **kwargs):
    registry = DeviceRegistry(clock=time.monotonic)
    for device in devices:
        registry.register(device)
    records = registry.records()
    deployments = DeploymentIndex.build(
        records, catalog, overrides=kwargs.pop("overrides", {})
    )
    return scheduler.plan(request, records, deployments, **kwargs)


# --- Scenario A: warm-device preference -----------------------------------


def test_warm_deployment_wins_over_cold(scheduler, catalog):
    laptop = _laptop([_capability("small-chat-v1", warm=True)])
    phone = _phone([_capability("small-chat-v1", warm=False)])
    plan = _plan(
        scheduler, catalog, [laptop, phone], RequestSpec(base_model_family="mock")
    )

    assert plan.chosen is not None
    assert plan.chosen.device_id == "x-elite-01"
    assert plan.chosen.deployment.state == ArtifactState.WARM
    phone_candidate = next(
        c for c in plan.candidates
        if c.device_id == "s25-ultra-01" and c.feasible
    )
    assert phone_candidate.cost.cold_load_ms > 0
    assert plan.chosen.cost.cold_load_ms == 0
    assert any("warm" in line.lower() for line in plan.explanation)


# --- Scenario B: behavior locality (runtime vs baked) ----------------------


def test_behavior_locality_explains_runtime_vs_baked_tradeoff(hybrid_scheduler, catalog):
    laptop = _laptop([_capability("small-chat-v1-concise-baked", warm=True)])
    phone = _phone([_capability("small-chat-v1", warm=True, supports_steering=True)])
    plan = _plan(
        hybrid_scheduler,
        catalog,
        [laptop, phone],
        RequestSpec(base_model_family="mock", behavior_profile_id="concise"),
    )

    assert plan.chosen is not None
    feasible_modes = {
        c.realization_mode for c in plan.candidates if c.feasible
    }
    assert feasible_modes == {"runtime_vector", "baked_profile"}
    # laptop's baked deployment wins on network/queue cost; the runtime
    # alternative must still be visible and explained
    assert plan.chosen.device_id == "x-elite-01"
    assert plan.chosen.realization_mode == "baked_profile"
    assert plan.chosen.artifact.behavior_profile_id == "concise"
    assert any("runtime" in line.lower() for line in plan.explanation)


# --- Scenario C: thermal pressure reroutes --------------------------------


def test_soft_thermal_pressure_flips_winner(scheduler, catalog):
    def run(phone_thermal: float):
        laptop = _laptop([_capability("small-chat-v1", warm=False)])
        phone = _phone(
            [_capability("small-chat-v1", warm=True)], thermal=phone_thermal
        )
        return _plan(
            scheduler, catalog, [laptop, phone], RequestSpec(base_model_family="mock")
        )

    cool = run(0.3)
    assert cool.chosen.device_id == "s25-ultra-01"  # warm phone wins when cool

    hot = run(0.75)
    assert hot.chosen.device_id == "x-elite-01"
    assert any("thermal" in line.lower() for line in hot.explanation)


def test_unhealthy_thermal_is_a_hard_rejection(scheduler, catalog):
    laptop = _laptop([_capability("small-chat-v1", warm=False)])
    phone = _phone([_capability("small-chat-v1", warm=True)], thermal=0.95)
    plan = _plan(
        scheduler, catalog, [laptop, phone], RequestSpec(base_model_family="mock")
    )

    assert plan.chosen.device_id == "x-elite-01"
    rejected = [
        c for c in plan.candidates if c.device_id == "s25-ultra-01"
    ]
    assert rejected and all(not c.feasible for c in rejected)
    assert any(
        "unhealthy" in reason.lower()
        for c in rejected
        for reason in c.rejection_reasons
    )


# --- Scenario D: memory projection rejects before dispatch ------------------


def test_long_context_memory_projection_rejects_phone(scheduler, catalog):
    laptop = _laptop([_capability("small-chat-v1", warm=True)])
    phone = _phone(
        [_capability("small-chat-v1", warm=True)], available_memory_mb=1000
    )
    request = RequestSpec(
        base_model_family="mock",
        estimated_input_tokens=2600,
        estimated_output_tokens=400,
    )
    plan = _plan(scheduler, catalog, [laptop, phone], request)

    assert plan.chosen.device_id == "x-elite-01"
    phone_candidates = [
        c for c in plan.candidates if c.device_id == "s25-ultra-01"
    ]
    assert phone_candidates and all(not c.feasible for c in phone_candidates)
    memory_reasons = [
        reason
        for c in phone_candidates
        for reason in c.rejection_reasons
        if "MB" in reason
    ]
    assert memory_reasons, "memory rejection must cite concrete numbers"
    assert phone_candidates[0].memory is not None
    assert phone_candidates[0].memory.estimated_fields  # estimates are flagged


# --- Scenario E: runtime steering unavailable -------------------------------


def test_runtime_steering_disabled_falls_back_to_baked(hybrid_scheduler, catalog):
    laptop = _laptop(
        [
            _capability("small-chat-v1", warm=True, supports_steering=True),
            _capability("small-chat-v1-concise-baked", warm=True),
        ]
    )
    plan = _plan(
        hybrid_scheduler,
        catalog,
        [laptop],
        RequestSpec(base_model_family="mock", behavior_profile_id="concise"),
        runtime_steering_disabled=frozenset({"x-elite-01"}),
    )

    assert plan.chosen is not None
    assert plan.chosen.realization_mode == "baked_profile"
    assert any(
        "runtime steering" in reason.lower()
        for c in plan.candidates
        if not c.feasible
        for reason in c.rejection_reasons
    )
    assert any("fell back" in line.lower() or "fallback" in line.lower()
               for line in plan.explanation)


def test_exact_only_override_rejects_instead_of_fallback(hybrid_scheduler, catalog):
    laptop = _laptop(
        [
            _capability("small-chat-v1", warm=True, supports_steering=True),
            _capability("small-chat-v1-concise-baked", warm=True),
        ]
    )
    plan = _plan(
        hybrid_scheduler,
        catalog,
        [laptop],
        RequestSpec(
            base_model_family="mock",
            behavior_profile_id="concise",
            fallback_policy_override="exact_only",
        ),
        runtime_steering_disabled=frozenset({"x-elite-01"}),
    )

    assert plan.chosen is None
    assert plan.error_code == "BEHAVIOR_UNAVAILABLE"


# --- Scenario G precondition: missing profile enters provisioning -----------


def test_missing_medical_safe_deployment_reports_behavior_unavailable(
    scheduler, catalog
):
    laptop = _laptop([_capability("large-reasoning-v1", warm=True)])
    plan = _plan(
        scheduler,
        catalog,
        [laptop],
        RequestSpec(base_model_family="mock", behavior_profile_id="medical-safe"),
    )

    assert plan.chosen is None
    assert plan.error_code == "BEHAVIOR_UNAVAILABLE"
    assert plan.provisioning_hint == "medical-safe"
    # the unsteered base model must NOT be silently substituted
    assert not any(c.feasible for c in plan.candidates)


def test_unvalidated_hardware_artifact_cannot_be_scheduled(scheduler, catalog):
    phone = _phone([_capability("qwen3-4b-qnn-s25", warm=True)])
    plan = _plan(
        scheduler,
        catalog,
        [phone],
        RequestSpec(base_model_family="qwen3"),
    )

    unsupported = next(
        candidate
        for candidate in plan.candidates
        if candidate.artifact.artifact_id == "qwen3-4b-qnn-s25"
    )
    assert unsupported.artifact.readiness == "unvalidated"
    assert not unsupported.feasible
    assert any(
        "has not been built/validated" in reason
        for reason in unsupported.rejection_reasons
    )
    assert plan.chosen is None


def test_never_silently_switches_behavior_profile(scheduler, catalog):
    laptop = _laptop(
        [
            _capability("small-chat-v1", warm=True, supports_steering=True),
            _capability("small-chat-v1-concise-baked", warm=True),
            _capability("large-reasoning-v1-medical-safe-baked", warm=True),
        ]
    )
    plan = _plan(
        scheduler,
        catalog,
        [laptop],
        RequestSpec(base_model_family="mock", behavior_profile_id="concise"),
    )

    for candidate in plan.candidates:
        assert candidate.artifact.behavior_profile_id in {"", "concise"}


# --- Cross-cutting behavior --------------------------------------------------


def test_private_request_restricts_to_origin(scheduler, catalog):
    laptop = _laptop([_capability("small-chat-v1", warm=True)])
    phone = _phone([_capability("small-chat-v1", warm=True)])
    plan = _plan(
        scheduler,
        catalog,
        [laptop, phone],
        RequestSpec(
            base_model_family="mock",
            privacy="private",
            origin_device_id="s25-ultra-01",
        ),
    )

    assert plan.chosen.device_id == "s25-ultra-01"
    laptop_reasons = [
        reason
        for c in plan.candidates
        if c.device_id == "x-elite-01"
        for reason in c.rejection_reasons
    ]
    assert any("privac" in reason.lower() for reason in laptop_reasons)


def test_artifact_absent_override_changes_route(scheduler, catalog):
    laptop = _laptop([_capability("small-chat-v1", warm=True)])
    phone = _phone([_capability("small-chat-v1", warm=True)])
    plan = _plan(
        scheduler,
        catalog,
        [laptop, phone],
        RequestSpec(base_model_family="mock"),
        overrides={("x-elite-01", "small-chat-v1"): ArtifactState.ABSENT},
    )

    assert plan.chosen.device_id == "s25-ultra-01"
    assert any(
        "absent" in reason.lower()
        for c in plan.candidates
        if c.device_id == "x-elite-01"
        for reason in c.rejection_reasons
    )


def test_plan_is_deterministic(scheduler, catalog):
    laptop = _laptop([_capability("small-chat-v1", warm=True)])
    phone = _phone([_capability("small-chat-v1", warm=True)])
    request = RequestSpec(base_model_family="mock", behavior_profile_id="")

    first = _plan(scheduler, catalog, [laptop, phone], request)
    second = _plan(scheduler, catalog, [laptop, phone], request)

    assert first.chosen.device_id == second.chosen.device_id
    assert first.chosen.cost.total_ms == second.chosen.cost.total_ms
    assert [c.device_id for c in first.candidates] == [
        c.device_id for c in second.candidates
    ]


def test_runtime_vector_choice_produces_steering_spec(hybrid_scheduler, catalog):
    phone = _phone([_capability("small-chat-v1", warm=True, supports_steering=True)])
    plan = _plan(
        hybrid_scheduler,
        catalog,
        [phone],
        RequestSpec(base_model_family="mock", behavior_profile_id="concise"),
    )

    assert plan.chosen.realization_mode == "runtime_vector"
    assert plan.steering.enabled
    assert plan.steering.vector_id == "concise-vs-verbose-layer-7"
    assert plan.steering.alpha == -2.0
    assert plan.prompt_prefix == ""


def test_prompt_fallback_is_labeled_not_activation_steering(scheduler, catalog):
    # friendly: runtime realization requires steering support; without it the
    # policy allows a prompt profile, which must not claim steering
    phone = _phone([_capability("small-chat-v1", warm=True)])
    plan = _plan(
        scheduler,
        catalog,
        [phone],
        RequestSpec(base_model_family="mock", behavior_profile_id="friendly"),
    )

    assert plan.chosen.realization_mode == "prompt_profile"
    assert plan.chosen.artifact.steering_realization == "none"
    assert not plan.steering.enabled
    assert plan.steering.vector_id == ""
    assert plan.prompt_prefix
    assert any("not activation steering" in line for line in plan.explanation)

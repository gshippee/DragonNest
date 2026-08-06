from __future__ import annotations

from pathlib import Path

import pytest

from dragon_nest.behavior import (
    BehaviorFallbackPolicy,
    BehaviorProfile,
    BehaviorProfileRegistry,
    SteeringRealization,
    SteeringRealizationMode,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def registry() -> BehaviorProfileRegistry:
    return BehaviorProfileRegistry.from_yaml(ROOT / "configs/behavior-profiles.yaml")


def test_registry_loads_demo_profiles(registry):
    ids = {profile.profile_id for profile in registry.all()}
    assert {
        "concise",
        "friendly",
        "medical-safe",
        "creative",
        "formal",
        "family-assistant",
    } <= ids


def test_concise_profile_declares_runtime_and_baked_realizations(registry):
    concise = registry.get("concise")
    modes = [realization.mode for realization in concise.realizations]
    assert SteeringRealizationMode.RUNTIME_VECTOR in modes
    assert SteeringRealizationMode.BAKED_PROFILE in modes
    runtime = next(
        r for r in concise.realizations
        if r.mode == SteeringRealizationMode.RUNTIME_VECTOR
    )
    assert runtime.vector_id == "concise-vs-verbose-layer-7"
    assert runtime.alpha_min <= runtime.alpha <= runtime.alpha_max


def test_fallback_ladder_expands_with_policy():
    realizations = (
        SteeringRealization(mode=SteeringRealizationMode.RUNTIME_VECTOR, vector_id="v"),
        SteeringRealization(mode=SteeringRealizationMode.BAKED_PROFILE, baked_artifact_id="a"),
        SteeringRealization(mode=SteeringRealizationMode.PROMPT_PROFILE, prompt_template="Be brief."),
    )

    def profile(policy: BehaviorFallbackPolicy) -> BehaviorProfile:
        return BehaviorProfile(
            profile_id="p",
            display_name="P",
            description="",
            base_model_family="mock",
            version="1",
            fallback_policy=policy,
            realizations=realizations,
        )

    exact = profile(BehaviorFallbackPolicy.EXACT_ONLY).allowed_modes()
    assert exact == (SteeringRealizationMode.RUNTIME_VECTOR,)

    baked = profile(BehaviorFallbackPolicy.ALLOW_BAKED_EQUIVALENT).allowed_modes()
    assert baked == (
        SteeringRealizationMode.RUNTIME_VECTOR,
        SteeringRealizationMode.BAKED_PROFILE,
    )

    prompt = profile(BehaviorFallbackPolicy.ALLOW_PROMPT_FALLBACK).allowed_modes()
    assert SteeringRealizationMode.PROMPT_PROFILE in prompt

    unsteered = profile(BehaviorFallbackPolicy.ALLOW_UNSTEERED).allowed_modes()
    assert unsteered[-1] == SteeringRealizationMode.NONE

    reject = profile(BehaviorFallbackPolicy.REJECT).allowed_modes()
    assert reject == (SteeringRealizationMode.RUNTIME_VECTOR,)


def test_prompt_realization_is_never_described_as_activation_steering():
    prompt = SteeringRealization(
        mode=SteeringRealizationMode.PROMPT_PROFILE, prompt_template="Be warm."
    )
    runtime = SteeringRealization(
        mode=SteeringRealizationMode.RUNTIME_VECTOR, vector_id="v", alpha=-2.0
    )

    assert "not activation steering" in prompt.describe()
    assert "activation steering" in runtime.describe()
    assert "prompt" not in runtime.describe()


def test_family_assistant_has_no_deployable_realizations(registry):
    profile = registry.get("family-assistant")
    deployable = [
        r for r in profile.realizations
        if r.mode != SteeringRealizationMode.NONE
    ]
    assert deployable == []
    assert profile.fallback_policy == BehaviorFallbackPolicy.REJECT


def test_unknown_profile_raises(registry):
    with pytest.raises(KeyError):
        registry.get("does-not-exist")

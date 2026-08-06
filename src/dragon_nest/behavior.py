"""Behavior profiles and their steering realizations.

A BehaviorProfile is the user-facing behavioral intent ("concise",
"medical-safe"). A SteeringRealization describes how that intent is
implemented on a concrete deployment: a runtime activation-steering vector,
a statically compiled ("baked") artifact variant, a prompt profile, or
nothing. Prompt conditioning is never activation steering and must never be
described as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml


class SteeringRealizationMode(StrEnum):
    RUNTIME_VECTOR = "runtime_vector"
    BAKED_PROFILE = "baked_profile"
    PROMPT_PROFILE = "prompt_profile"
    NONE = "none"


class BehaviorFallbackPolicy(StrEnum):
    EXACT_ONLY = "exact_only"
    ALLOW_BAKED_EQUIVALENT = "allow_baked_equivalent"
    ALLOW_RUNTIME_EQUIVALENT = "allow_runtime_equivalent"
    ALLOW_PROMPT_FALLBACK = "allow_prompt_fallback"
    ALLOW_UNSTEERED = "allow_unsteered"
    REJECT = "reject"


# Which realization modes each fallback policy admits, beyond the profile's
# first-preference realization. Order within the profile's declared
# realizations is preserved; the policy only widens the admissible set.
_POLICY_MODES: dict[BehaviorFallbackPolicy, frozenset[SteeringRealizationMode]] = {
    BehaviorFallbackPolicy.EXACT_ONLY: frozenset(),
    BehaviorFallbackPolicy.REJECT: frozenset(),
    BehaviorFallbackPolicy.ALLOW_BAKED_EQUIVALENT: frozenset(
        {SteeringRealizationMode.BAKED_PROFILE}
    ),
    BehaviorFallbackPolicy.ALLOW_RUNTIME_EQUIVALENT: frozenset(
        {
            SteeringRealizationMode.BAKED_PROFILE,
            SteeringRealizationMode.RUNTIME_VECTOR,
        }
    ),
    BehaviorFallbackPolicy.ALLOW_PROMPT_FALLBACK: frozenset(
        {
            SteeringRealizationMode.BAKED_PROFILE,
            SteeringRealizationMode.RUNTIME_VECTOR,
            SteeringRealizationMode.PROMPT_PROFILE,
        }
    ),
    BehaviorFallbackPolicy.ALLOW_UNSTEERED: frozenset(
        {
            SteeringRealizationMode.BAKED_PROFILE,
            SteeringRealizationMode.RUNTIME_VECTOR,
            SteeringRealizationMode.PROMPT_PROFILE,
            SteeringRealizationMode.NONE,
        }
    ),
}


@dataclass(frozen=True)
class SteeringRealization:
    mode: SteeringRealizationMode
    vector_id: str = ""
    alpha: float = 0.0
    alpha_min: float = 0.0
    alpha_max: float = 0.0
    injection_layer: int = -1
    positions: str = "last"
    baked_artifact_id: str = ""
    prompt_template: str = ""
    compatible_model_families: tuple[str, ...] = ()
    compatible_runtimes: tuple[str, ...] = ()
    compatible_quantizations: tuple[str, ...] = ()
    verification_status: str = "unverified"  # unverified | calibrated | verified

    def describe(self) -> str:
        if self.mode == SteeringRealizationMode.RUNTIME_VECTOR:
            return (
                f"runtime activation steering: vector {self.vector_id} "
                f"alpha={self.alpha} layer={self.injection_layer} "
                f"positions={self.positions}"
            )
        if self.mode == SteeringRealizationMode.BAKED_PROFILE:
            return (
                "statically compiled steering profile baked into artifact "
                f"{self.baked_artifact_id}"
            )
        if self.mode == SteeringRealizationMode.PROMPT_PROFILE:
            return "prompt profile (not activation steering)"
        return "no behavior conditioning"


@dataclass(frozen=True)
class BehaviorProfile:
    profile_id: str
    display_name: str
    description: str
    base_model_family: str
    version: str
    policy_tags: tuple[str, ...] = ()
    fallback_policy: BehaviorFallbackPolicy = BehaviorFallbackPolicy.ALLOW_UNSTEERED
    provenance: str = ""
    evaluation_status: str = "draft"  # draft | calibrated | validated
    realizations: tuple[SteeringRealization, ...] = ()

    def allowed_modes(self) -> tuple[SteeringRealizationMode, ...]:
        """Realization modes admissible under the fallback policy, in the
        profile's declared preference order.

        The first declared realization is always admissible (that is the
        "exact" request). Wider policies admit additional declared modes but
        never modes the profile does not declare, except ALLOW_UNSTEERED
        which always admits NONE as a last resort.
        """
        declared = [realization.mode for realization in self.realizations]
        allowed: list[SteeringRealizationMode] = []
        widened = _POLICY_MODES[self.fallback_policy]
        for index, mode in enumerate(declared):
            if (index == 0 or mode in widened) and mode not in allowed:
                allowed.append(mode)
        if (
            self.fallback_policy == BehaviorFallbackPolicy.ALLOW_UNSTEERED
            and SteeringRealizationMode.NONE not in allowed
        ):
            allowed.append(SteeringRealizationMode.NONE)
        return tuple(allowed)

    def realization_for(
        self, mode: SteeringRealizationMode
    ) -> SteeringRealization | None:
        for realization in self.realizations:
            if realization.mode == mode:
                return realization
        if mode == SteeringRealizationMode.NONE:
            return SteeringRealization(mode=SteeringRealizationMode.NONE)
        return None


class BehaviorProfileRegistry:
    def __init__(self, profiles: dict[str, BehaviorProfile]):
        self._profiles = profiles

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BehaviorProfileRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        profiles: dict[str, BehaviorProfile] = {}
        for item in raw.get("profiles", []):
            realizations = tuple(
                SteeringRealization(
                    mode=SteeringRealizationMode(entry["mode"]),
                    vector_id=str(entry.get("vector_id", "")),
                    alpha=float(entry.get("alpha", 0.0)),
                    alpha_min=float(entry.get("alpha_min", 0.0)),
                    alpha_max=float(entry.get("alpha_max", 0.0)),
                    injection_layer=int(entry.get("injection_layer", -1)),
                    positions=str(entry.get("positions", "last")),
                    baked_artifact_id=str(entry.get("baked_artifact_id", "")),
                    prompt_template=str(entry.get("prompt_template", "")),
                    compatible_model_families=tuple(
                        entry.get("compatible_model_families", [])
                    ),
                    compatible_runtimes=tuple(entry.get("compatible_runtimes", [])),
                    compatible_quantizations=tuple(
                        entry.get("compatible_quantizations", [])
                    ),
                    verification_status=str(
                        entry.get("verification_status", "unverified")
                    ),
                )
                for entry in item.get("realizations", [])
            )
            profile = BehaviorProfile(
                profile_id=str(item["profile_id"]),
                display_name=str(item.get("display_name", item["profile_id"])),
                description=str(item.get("description", "")),
                base_model_family=str(item.get("base_model_family", "")),
                version=str(item.get("version", "1")),
                policy_tags=tuple(item.get("policy_tags", [])),
                fallback_policy=BehaviorFallbackPolicy(
                    item.get("fallback_policy", "allow_unsteered")
                ),
                provenance=str(item.get("provenance", "")),
                evaluation_status=str(item.get("evaluation_status", "draft")),
                realizations=realizations,
            )
            profiles[profile.profile_id] = profile
        return cls(profiles)

    def get(self, profile_id: str) -> BehaviorProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise KeyError(f"unknown behavior profile {profile_id}") from exc

    def all(self) -> tuple[BehaviorProfile, ...]:
        return tuple(self._profiles[key] for key in sorted(self._profiles))

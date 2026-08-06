"""Behavior-aware deployment scheduler.

Routes executable deployments — (device, artifact, behavior realization)
triples — never abstract models and never raw free memory. Feasibility is
decided by hard constraints before any scoring; scoring is an explicit,
deterministic, configurable cost model in milliseconds-equivalent. Behavior
fallback follows the profile's explicit policy and never silently switches to
a different behavior profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .behavior import (
    BehaviorFallbackPolicy,
    BehaviorProfile,
    BehaviorProfileRegistry,
    SteeringRealization,
    SteeringRealizationMode,
)
from .deployments import (
    ArtifactCatalog,
    ArtifactSpec,
    ArtifactState,
    DeploymentIndex,
    DeploymentState,
    EXECUTABLE_STATES,
    device_compatibility_classes,
)
from .models import HealthStatus, SteeringSpec
from .registry import DeviceRecord
from .steering import SteeringRegistry


@dataclass(frozen=True)
class RequestSpec:
    """What the user asked for, in scheduler terms."""

    request_text: str = ""
    base_model_family: str = "mock"
    behavior_profile_id: str = ""
    estimated_input_tokens: int = 256
    estimated_output_tokens: int = 128
    privacy: str = "trusted_fabric"  # trusted_fabric | private
    latency_preference: str = "interactive"  # realtime | interactive | background
    origin_device_id: str = ""
    fallback_policy_override: str = ""


@dataclass(frozen=True)
class MemoryProjection:
    fixed_runtime_mb: int
    artifact_mb: int
    kv_cache_mb: int
    boundary_mb: int
    safety_margin_mb: int
    total_mb: int
    available_mb: int
    estimated_fields: tuple[str, ...]


@dataclass(frozen=True)
class CostBreakdown:
    queue_delay_ms: float
    cold_load_ms: float
    prefill_ms: float
    decode_ms: float
    network_ms: float
    thermal_battery_penalty_ms: float
    eviction_penalty_ms: float
    failure_risk_ms: float
    total_ms: float


@dataclass(frozen=True)
class ExecutionCandidate:
    device_id: str
    artifact: ArtifactSpec
    realization_mode: str
    realization: SteeringRealization | None
    deployment: DeploymentState
    feasible: bool
    rejection_reasons: tuple[str, ...]
    memory: MemoryProjection | None
    cost: CostBreakdown | None


@dataclass(frozen=True)
class RoutePlan:
    request: RequestSpec
    profile: BehaviorProfile | None
    fallback_policy: str
    candidates: tuple[ExecutionCandidate, ...]
    chosen: ExecutionCandidate | None
    steering: SteeringSpec
    prompt_prefix: str
    explanation: tuple[str, ...]
    error_code: str = ""
    provisioning_hint: str = ""


@dataclass(frozen=True)
class SchedulerConfig:
    """Explicit deterministic scoring knobs. No learned components."""

    fixed_runtime_mb: int = 512
    safety_margin_mb: int = 384
    boundary_mb: int = 8
    default_kv_bytes_per_token: int = 131072
    default_load_time_ms: int = 8000
    default_prefill_tps: float = 120.0
    default_decode_tps: float = 12.0
    default_rtt_ms: float = 50.0
    queue_delay_per_task_ms: float = 1500.0
    thermal_soft_threshold: float = 0.55
    thermal_penalty_ms: float = 6000.0
    low_battery_pct: float = 25.0
    battery_penalty_ms: float = 4000.0
    eviction_headroom_mb: int = 1024
    eviction_penalty_ms: float = 3000.0
    failure_risk_stale_ms: float = 5000.0
    failure_risk_degraded_ms: float = 1500.0
    realtime_load_factor: float = 1.5
    background_load_factor: float = 0.5


_MODE_ORDER = {mode: index for index, mode in enumerate(SteeringRealizationMode)}


class DeploymentScheduler:
    def __init__(
        self,
        catalog: ArtifactCatalog,
        behaviors: BehaviorProfileRegistry,
        steering_registry: SteeringRegistry,
        config: SchedulerConfig | None = None,
    ):
        self.catalog = catalog
        self.behaviors = behaviors
        self.steering_registry = steering_registry
        self.config = config or SchedulerConfig()

    def plan(
        self,
        request: RequestSpec,
        records: tuple[DeviceRecord, ...],
        deployments: DeploymentIndex,
        runtime_steering_disabled: frozenset[str] = frozenset(),
    ) -> RoutePlan:
        profile: BehaviorProfile | None = None
        if request.behavior_profile_id:
            profile = self.behaviors.get(request.behavior_profile_id)
            if request.fallback_policy_override:
                profile = replace(
                    profile,
                    fallback_policy=BehaviorFallbackPolicy(
                        request.fallback_policy_override
                    ),
                )
        fallback_policy = profile.fallback_policy.value if profile else ""
        allowed_modes = (
            profile.allowed_modes()
            if profile is not None
            else (SteeringRealizationMode.NONE,)
        )

        candidates: list[ExecutionCandidate] = []
        for record in sorted(records, key=lambda item: item.device.device_id):
            for mode in allowed_modes:
                realization = (
                    profile.realization_for(mode)
                    if profile is not None
                    else SteeringRealization(mode=SteeringRealizationMode.NONE)
                )
                if realization is None:
                    continue
                for artifact in self._artifacts_for(request, profile, mode):
                    candidates.append(
                        self._evaluate(
                            request,
                            record,
                            artifact,
                            mode,
                            realization,
                            deployments,
                            runtime_steering_disabled,
                        )
                    )

        candidates.sort(
            key=lambda c: (
                c.device_id,
                c.artifact.artifact_id,
                _MODE_ORDER.get(SteeringRealizationMode(c.realization_mode), 9),
            )
        )
        feasible = [candidate for candidate in candidates if candidate.feasible]
        chosen = min(
            feasible,
            key=lambda c: (
                c.cost.total_ms,
                c.device_id,
                c.artifact.artifact_id,
                c.realization_mode,
            ),
            default=None,
        )

        error_code = ""
        provisioning_hint = ""
        if chosen is None:
            if profile is not None:
                error_code = "BEHAVIOR_UNAVAILABLE"
                provisioning_hint = profile.profile_id
            else:
                error_code = "NO_FEASIBLE_DEPLOYMENT"

        steering = SteeringSpec()
        prompt_prefix = ""
        if chosen is not None and chosen.realization is not None:
            realization = chosen.realization
            if chosen.realization_mode == SteeringRealizationMode.RUNTIME_VECTOR:
                steering = SteeringSpec(
                    enabled=True,
                    vector_id=realization.vector_id,
                    model_family=chosen.artifact.base_model_family,
                    target_layer=realization.injection_layer,
                    alpha=realization.alpha,
                    positions=realization.positions,
                )
            elif chosen.realization_mode == SteeringRealizationMode.PROMPT_PROFILE:
                prompt_prefix = realization.prompt_template

        explanation = self._explain(
            request,
            profile,
            allowed_modes,
            tuple(candidates),
            chosen,
            error_code,
        )
        return RoutePlan(
            request=request,
            profile=profile,
            fallback_policy=fallback_policy,
            candidates=tuple(candidates),
            chosen=chosen,
            steering=steering,
            prompt_prefix=prompt_prefix,
            explanation=explanation,
            error_code=error_code,
            provisioning_hint=provisioning_hint,
        )

    # -- candidate generation -------------------------------------------------

    def _artifacts_for(
        self,
        request: RequestSpec,
        profile: BehaviorProfile | None,
        mode: SteeringRealizationMode,
    ) -> tuple[ArtifactSpec, ...]:
        if mode == SteeringRealizationMode.BAKED_PROFILE:
            if profile is None:
                return ()
            return self.catalog.baked_for(
                profile.profile_id, request.base_model_family
            )
        # runtime_vector / prompt_profile / none all execute an unbaked base
        # artifact. Baked variants of *other* profiles are never candidates:
        # substituting them would silently switch behavior.
        return tuple(
            artifact
            for artifact in self.catalog.full_models(request.base_model_family)
            if not artifact.behavior_profile_id
        )

    def _evaluate(
        self,
        request: RequestSpec,
        record: DeviceRecord,
        artifact: ArtifactSpec,
        mode: SteeringRealizationMode,
        realization: SteeringRealization,
        deployments: DeploymentIndex,
        runtime_steering_disabled: frozenset[str],
    ) -> ExecutionCandidate:
        device = record.device
        deployment = deployments.state(device.device_id, artifact.artifact_id)
        reasons: list[str] = []

        if record.status in {HealthStatus.OFFLINE, HealthStatus.UNHEALTHY}:
            reasons.append(
                f"device health is {record.status.value} "
                f"(thermal={device.health.thermal_level:.2f}, "
                f"battery={device.health.battery_pct:.0f}%): unhealthy devices "
                "are excluded before scoring"
            )
        if not record.stream_connected or not device.health.reachable:
            reasons.append("device is disconnected or unreachable")

        classes = device_compatibility_classes(device)
        if not set(classes) & set(artifact.compatibility_classes):
            reasons.append(
                f"artifact targets {'/'.join(artifact.compatibility_classes)}; "
                f"device compatibility classes are {'/'.join(classes)}"
            )

        if artifact.readiness != "ready":
            reasons.append(
                f"artifact readiness is {artifact.readiness!r}; it has not been "
                "built/validated"
            )

        if deployment.state not in EXECUTABLE_STATES:
            reasons.append(
                f"artifact is {deployment.state.value} on this device; only "
                "installed or warm deployments are dispatchable"
            )

        total_tokens = (
            request.estimated_input_tokens + request.estimated_output_tokens
        )
        if not artifact.supports_context(total_tokens):
            reasons.append(
                f"projected sequence of {total_tokens} tokens exceeds the "
                f"artifact context profile of {artifact.max_context_tokens}"
            )

        if request.privacy == "private" and (
            device.device_id != request.origin_device_id
        ):
            reasons.append(
                "privacy policy restricts this request to the origin device "
                f"{request.origin_device_id or '(unspecified)'}"
            )

        if mode == SteeringRealizationMode.RUNTIME_VECTOR:
            reasons.extend(
                self._runtime_vector_reasons(
                    record, artifact, realization, runtime_steering_disabled
                )
            )

        memory = self._project_memory(request, device, artifact, deployment)
        if memory.available_mb <= 0:
            reasons.append(
                "device telemetry reports unknown available memory; refusing "
                "to project a fit"
            )
        elif memory.total_mb > memory.available_mb:
            reasons.append(
                f"projected memory {memory.total_mb} MB exceeds available "
                f"{memory.available_mb} MB "
                f"(fixed {memory.fixed_runtime_mb} + artifact "
                f"{memory.artifact_mb} + KV cache {memory.kv_cache_mb} + "
                f"margin {memory.safety_margin_mb})"
            )

        feasible = not reasons
        cost = (
            self._score(request, record, artifact, deployment, deployments)
            if feasible
            else None
        )
        return ExecutionCandidate(
            device_id=device.device_id,
            artifact=artifact,
            realization_mode=mode.value,
            realization=realization,
            deployment=deployment,
            feasible=feasible,
            rejection_reasons=tuple(reasons),
            memory=memory,
            cost=cost,
        )

    def _runtime_vector_reasons(
        self,
        record: DeviceRecord,
        artifact: ArtifactSpec,
        realization: SteeringRealization,
        runtime_steering_disabled: frozenset[str],
    ) -> list[str]:
        device = record.device
        reasons: list[str] = []
        if device.device_id in runtime_steering_disabled:
            reasons.append(
                "runtime steering support is disabled on this device"
            )
        capability = next(
            (
                model
                for model in device.models
                if model.model_id == artifact.artifact_id
            ),
            None,
        )
        if capability is None:
            reasons.append(
                "device does not advertise this artifact, so runtime vector "
                "injection capability is unknown"
            )
            return reasons
        if not capability.supports_steering:
            reasons.append(
                "device runtime for this artifact does not support runtime "
                "activation-steering injection"
            )
        if (
            capability.steering_vector_ids
            and realization.vector_id not in capability.steering_vector_ids
        ):
            reasons.append(
                f"device does not hold steering vector {realization.vector_id}"
            )
        if (
            capability.supported_steering_layers
            and realization.injection_layer
            not in capability.supported_steering_layers
        ):
            reasons.append(
                f"device runtime cannot inject at layer "
                f"{realization.injection_layer}"
            )
        if realization.vector_id:
            ok, reason = self.steering_registry.runtime_compatible(
                realization.vector_id,
                model_family=artifact.base_model_family,
                model_revision=artifact.model_version,
                runtime=artifact.runtime,
                quantization=artifact.quantization,
                injection_layer=realization.injection_layer,
            )
            if not ok:
                reasons.append(reason)
        return reasons

    # -- projections and scoring ---------------------------------------------

    def _project_memory(
        self,
        request: RequestSpec,
        device,
        artifact: ArtifactSpec,
        deployment: DeploymentState,
    ) -> MemoryProjection:
        estimated: list[str] = ["fixed_runtime"]
        artifact_memory, is_estimate = artifact.memory_mb()
        if artifact_memory <= 0:
            artifact_memory = max(artifact.artifact_size_mb, 1024)
            is_estimate = True
        if is_estimate:
            estimated.append("artifact_memory")
        newly_resident = (
            0 if deployment.state == ArtifactState.WARM else artifact_memory
        )
        kv_per_token = artifact.kv_cache_bytes_per_token
        if kv_per_token <= 0:
            kv_per_token = self.config.default_kv_bytes_per_token
        total_tokens = (
            request.estimated_input_tokens + request.estimated_output_tokens
        )
        kv_mb = math.ceil(kv_per_token * total_tokens / (1024 * 1024))
        estimated.append("kv_cache")
        boundary_mb = (
            self.config.boundary_mb if artifact.topology == "pipeline_stage" else 0
        )
        total = (
            self.config.fixed_runtime_mb
            + newly_resident
            + kv_mb
            + boundary_mb
            + self.config.safety_margin_mb
        )
        return MemoryProjection(
            fixed_runtime_mb=self.config.fixed_runtime_mb,
            artifact_mb=newly_resident,
            kv_cache_mb=kv_mb,
            boundary_mb=boundary_mb,
            safety_margin_mb=self.config.safety_margin_mb,
            total_mb=total,
            available_mb=max(device.health.available_memory_mb, 0),
            estimated_fields=tuple(estimated),
        )

    def _score(
        self,
        request: RequestSpec,
        record: DeviceRecord,
        artifact: ArtifactSpec,
        deployment: DeploymentState,
        deployments: DeploymentIndex,
    ) -> CostBreakdown:
        config = self.config
        device = record.device
        load_factor = {
            "realtime": config.realtime_load_factor,
            "background": config.background_load_factor,
        }.get(request.latency_preference, 1.0)

        queue = len(record.active_task_ids) * config.queue_delay_per_task_ms
        queue *= load_factor

        if deployment.state == ArtifactState.WARM:
            cold = 0.0
        else:
            cold = float(
                artifact.measured_load_time_ms or config.default_load_time_ms
            )
        cold *= load_factor

        prefill_tps = (
            deployment.measured_prefill_tokens_per_s
            or artifact.prefill_tokens_per_s
            or config.default_prefill_tps
        )
        decode_tps = (
            deployment.measured_decode_tokens_per_s
            or artifact.decode_tokens_per_s
            or config.default_decode_tps
        )
        prefill = request.estimated_input_tokens / prefill_tps * 1000
        decode = request.estimated_output_tokens / decode_tps * 1000

        rtt = device.health.network_rtt_ms
        network = 2 * (rtt if rtt >= 0 else config.default_rtt_ms)

        penalty = 0.0
        thermal = device.health.thermal_level
        if thermal > config.thermal_soft_threshold:
            penalty += (
                config.thermal_penalty_ms
                * (thermal - config.thermal_soft_threshold)
                / (1.0 - config.thermal_soft_threshold)
            )
        battery = device.health.battery_pct
        if 0 <= battery < config.low_battery_pct and not device.health.charging:
            penalty += config.battery_penalty_ms

        eviction = 0.0
        if deployment.state != ArtifactState.WARM:
            other_warm = [
                artifact_id
                for artifact_id in deployments.warm_artifacts(device.device_id)
                if artifact_id != artifact.artifact_id
            ]
            artifact_memory, _ = artifact.memory_mb()
            headroom = device.health.available_memory_mb - artifact_memory
            if other_warm and headroom < config.eviction_headroom_mb:
                eviction = config.eviction_penalty_ms

        risk = 0.0
        if record.status == HealthStatus.STALE:
            risk = config.failure_risk_stale_ms
        elif record.status == HealthStatus.DEGRADED:
            risk = config.failure_risk_degraded_ms

        total = queue + cold + prefill + decode + network + penalty + eviction + risk
        return CostBreakdown(
            queue_delay_ms=round(queue, 1),
            cold_load_ms=round(cold, 1),
            prefill_ms=round(prefill, 1),
            decode_ms=round(decode, 1),
            network_ms=round(network, 1),
            thermal_battery_penalty_ms=round(penalty, 1),
            eviction_penalty_ms=round(eviction, 1),
            failure_risk_ms=round(risk, 1),
            total_ms=round(total, 1),
        )

    # -- explanation ----------------------------------------------------------

    def _explain(
        self,
        request: RequestSpec,
        profile: BehaviorProfile | None,
        allowed_modes: tuple[SteeringRealizationMode, ...],
        candidates: tuple[ExecutionCandidate, ...],
        chosen: ExecutionCandidate | None,
        error_code: str,
    ) -> tuple[str, ...]:
        lines: list[str] = []
        total_tokens = (
            request.estimated_input_tokens + request.estimated_output_tokens
        )
        lines.append(
            f"Request: base model family {request.base_model_family!r}, "
            f"behavior profile {request.behavior_profile_id or 'none'}, "
            f"~{request.estimated_input_tokens} input + "
            f"~{request.estimated_output_tokens} output tokens "
            f"({total_tokens} total, estimates), privacy={request.privacy}, "
            f"latency={request.latency_preference}."
        )
        if profile is not None:
            lines.append(
                f"Behavior fallback policy {profile.fallback_policy.value!r} "
                "admits realizations: "
                + ", ".join(mode.value for mode in allowed_modes)
                + " (in preference order)."
            )
        feasible = [candidate for candidate in candidates if candidate.feasible]
        devices = {candidate.device_id for candidate in candidates}
        lines.append(
            f"Considered {len(candidates)} deployment candidates across "
            f"{len(devices)} device(s); {len(feasible)} feasible after hard "
            "constraints."
        )
        if chosen is not None:
            realization_text = (
                chosen.realization.describe()
                if chosen.realization is not None
                else "no behavior conditioning"
            )
            cost = chosen.cost
            lines.append(
                f"Chose {chosen.device_id} running artifact "
                f"{chosen.artifact.artifact_id} "
                f"({chosen.deployment.state.value}); behavior realized via "
                f"{realization_text}."
            )
            lines.append(
                f"Predicted cost ≈{cost.total_ms:.0f} ms: queue "
                f"{cost.queue_delay_ms:.0f}, cold load {cost.cold_load_ms:.0f}, "
                f"prefill {cost.prefill_ms:.0f}, decode {cost.decode_ms:.0f}, "
                f"network {cost.network_ms:.0f}, thermal/battery penalty "
                f"{cost.thermal_battery_penalty_ms:.0f}, eviction "
                f"{cost.eviction_penalty_ms:.0f}, failure risk "
                f"{cost.failure_risk_ms:.0f}."
            )
            if chosen.memory is not None:
                memory = chosen.memory
                lines.append(
                    f"Memory projection on {chosen.device_id}: "
                    f"{memory.total_mb} MB of {memory.available_mb} MB "
                    f"available (fixed {memory.fixed_runtime_mb} + newly "
                    f"resident artifact {memory.artifact_mb} + KV cache "
                    f"{memory.kv_cache_mb} + margin {memory.safety_margin_mb}); "
                    f"estimated components: "
                    f"{', '.join(memory.estimated_fields)}."
                )
            if (
                profile is not None
                and allowed_modes
                and chosen.realization_mode != allowed_modes[0].value
            ):
                lines.append(
                    f"Preferred realization {allowed_modes[0].value!r} was not "
                    f"feasible on any device; fell back to "
                    f"{chosen.realization_mode!r} under policy "
                    f"{profile.fallback_policy.value!r} without changing the "
                    "requested behavior profile."
                )
            for candidate in sorted(
                feasible, key=lambda c: c.cost.total_ms
            ):
                if candidate is chosen:
                    continue
                delta = candidate.cost.total_ms - chosen.cost.total_ms
                lines.append(
                    f"Alternative: {candidate.device_id} / "
                    f"{candidate.artifact.artifact_id} via "
                    f"{candidate.realization_mode} would cost "
                    f"≈{candidate.cost.total_ms:.0f} ms (+{delta:.0f} ms)."
                )
        elif error_code == "BEHAVIOR_UNAVAILABLE":
            lines.append(
                f"No deployment can realize behavior profile "
                f"{request.behavior_profile_id!r} under policy constraints; "
                "the request was rejected rather than silently degraded. "
                "The profile can be provisioned (built, validated, and "
                "deployed) instead."
            )
        else:
            lines.append(
                "No feasible deployment exists for this request; see "
                "per-candidate rejection reasons."
            )
        for candidate in candidates:
            if candidate.feasible:
                continue
            lines.append(
                f"Rejected {candidate.device_id} / "
                f"{candidate.artifact.artifact_id} "
                f"[{candidate.realization_mode}]: "
                + "; ".join(candidate.rejection_reasons)
            )
        return tuple(lines)

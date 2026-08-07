from __future__ import annotations

from dataclasses import replace

from .models import (
    Device,
    ExecutionMode,
    ExecutionPlan,
    HealthStatus,
    ModelCapability,
    PipelineStage,
    RouteDecision,
    SteeringSpec,
    TaskProfile,
)
from .steering import SteeringRegistry


class DeterministicRouter:
    def __init__(self, steering_registry: SteeringRegistry | None = None):
        self.steering_registry = steering_registry

    def route(
        self,
        plan: ExecutionPlan,
        profile: TaskProfile,
        devices: list[Device],
    ) -> tuple[ExecutionPlan, RouteDecision]:
        devices = self._eligible_health_tier(devices)
        if plan.execution_mode == ExecutionMode.LAYER_PIPELINE:
            return self._route_pipeline(plan, profile, devices)
        if plan.execution_mode == ExecutionMode.DATA_PARALLEL:
            return self._route_data_parallel(plan, profile, devices)
        return self._route_single(plan, profile, devices)

    def _route_single(
        self,
        plan: ExecutionPlan,
        profile: TaskProfile,
        devices: list[Device],
    ) -> tuple[ExecutionPlan, RouteDecision]:
        ranked = self._rank_models(
            profile,
            devices,
            plan.steering,
            origin_device_id=(
                "" if plan.preferred_mode == "quality" else plan.origin_device_id
            ),
            prefer_quality=plan.preferred_mode in {"local", "private", "quality"},
        )
        if not ranked:
            raise ValueError("no eligible device/model for task")
        score, device, model, reason = ranked[0]
        fallbacks = tuple(item[1].device_id for item in ranked[1:3])
        task = replace(
            plan.tasks[0],
            selected_device_id=device.device_id,
            selected_model_id=model.model_id,
            fallback_device_ids=fallbacks,
        )
        routed_plan = replace(plan, tasks=(task,))
        reasons = (
            *plan.reasons,
            reason,
            *self._origin_route_reason(plan.origin_device_id, device.device_id),
            f"Fallbacks: {', '.join(fallbacks) or 'none'}.",
        )
        return routed_plan, RouteDecision(
            execution_mode=ExecutionMode.SINGLE,
            selected_device_id=device.device_id,
            selected_model_id=model.model_id,
            fallback_device_ids=fallbacks,
            reasons=reasons,
            route_score=score,
        )

    def _route_data_parallel(
        self,
        plan: ExecutionPlan,
        profile: TaskProfile,
        devices: list[Device],
    ) -> tuple[ExecutionPlan, RouteDecision]:
        ranked = self._rank_models(
            profile,
            devices,
            plan.steering,
            require_data_parallel=True,
            origin_device_id=plan.origin_device_id,
        )
        if not ranked:
            raise ValueError("no eligible device/model for data-parallel task")
        routed_tasks = []
        reasons = list(plan.reasons)
        for idx, task in enumerate(plan.tasks):
            score, device, model, reason = ranked[idx % len(ranked)]
            fallbacks = tuple(
                item[1].device_id
                for item in ranked
                if item[1].device_id != device.device_id
            )[:2]
            routed_tasks.append(
                replace(
                    task,
                    selected_device_id=device.device_id,
                    selected_model_id=model.model_id,
                    fallback_device_ids=fallbacks,
                )
            )
            reasons.append(
                f"{task.shard_id} -> {device.device_id}/{model.model_id}: {reason}"
            )
        first = routed_tasks[0]
        if plan.origin_device_id and first.selected_device_id == plan.origin_device_id:
            reasons.append(
                f"Origin preference assigned {first.shard_id} to {plan.origin_device_id}."
            )
        routed_plan = replace(plan, tasks=tuple(routed_tasks))
        return routed_plan, RouteDecision(
            execution_mode=ExecutionMode.DATA_PARALLEL,
            selected_device_id=first.selected_device_id,
            selected_model_id=first.selected_model_id,
            fallback_device_ids=first.fallback_device_ids,
            reasons=tuple(reasons),
            route_score=sum(item[0] for item in ranked[: len(routed_tasks)])
            / min(len(ranked), len(routed_tasks)),
        )

    def _route_pipeline(
        self,
        plan: ExecutionPlan,
        profile: TaskProfile,
        devices: list[Device],
    ) -> tuple[ExecutionPlan, RouteDecision]:
        indexed = self._route_indexed_pipeline(plan, profile, devices)
        if indexed is not None:
            return indexed

        # Backward-compatible reconstruction for the original two-stage
        # Qwen3-0.6B proof, whose advertisements predate stage_index/count.
        segments: list[tuple[Device, ModelCapability]] = []
        for device in devices:
            if not self._device_usable(device):
                continue
            for model in device.models:
                if (
                    model.segment
                    and model.segment.stage_count == 0
                    and model.supports_layer_pipeline
                    and profile.task_class in model.task_classes
                    and self._has_model_memory(device, model)
                ):
                    segments.append((device, model))
        for left_device, left_model in segments:
            left = left_model.segment
            if (
                left is None
                or left.start_layer is None
                or left.end_layer is None
                or not left.includes_embedding
            ):
                continue
            for right_device, right_model in segments:
                right = right_model.segment
                if (
                    right is None
                    or right.start_layer is None
                    or right.end_layer is None
                    or not right.includes_lm_head
                ):
                    continue
                contiguous = (
                    left.pipeline_id == right.pipeline_id
                    and left.end_layer == right.start_layer
                    and left.start_layer == 0
                    and right.end_layer == right.total_layers
                    and left_model.model_family == right_model.model_family
                    and left_model.model_version == right_model.model_version
                    and left_model.tokenizer_id == right_model.tokenizer_id
                    and left_model.precision == right_model.precision
                    and left_model.boundary_format == right_model.boundary_format
                )
                if not contiguous:
                    continue
                if plan.steering.enabled:
                    steering_model = None
                    if left.start_layer <= plan.steering.target_layer < left.end_layer:
                        steering_model = left_model
                    elif (
                        right.start_layer
                        <= plan.steering.target_layer
                        < right.end_layer
                    ):
                        steering_model = right_model
                    if steering_model is None or self.steering_registry is None:
                        continue
                    compatible, _ = self.steering_registry.validate(
                        plan.steering, steering_model
                    )
                    if not compatible:
                        continue
                stages = (
                    PipelineStage(
                        stage_id="stage-1",
                        stage_index=0,
                        pipeline_id=left.pipeline_id,
                        selected_device_id=left_device.device_id,
                        selected_model_id=left_model.model_id,
                        start_layer=left.start_layer,
                        end_layer=left.end_layer,
                        model_family=left_model.model_family,
                        model_version=left_model.model_version,
                        tokenizer_id=left_model.tokenizer_id,
                        precision=left_model.precision,
                        boundary_format=left_model.boundary_format,
                    ),
                    PipelineStage(
                        stage_id="stage-2",
                        stage_index=1,
                        pipeline_id=right.pipeline_id,
                        selected_device_id=right_device.device_id,
                        selected_model_id=right_model.model_id,
                        start_layer=right.start_layer,
                        end_layer=right.end_layer,
                        model_family=right_model.model_family,
                        model_version=right_model.model_version,
                        tokenizer_id=right_model.tokenizer_id,
                        precision=right_model.precision,
                        boundary_format=right_model.boundary_format,
                    ),
                )
                reason_list = [
                    *plan.reasons,
                    f"Selected layer pipeline {left.pipeline_id}: "
                    f"{left_device.device_id} layers {left.start_layer}..{left.end_layer}, "
                    f"{right_device.device_id} layers {right.start_layer}..{right.end_layer}.",
                ]
                if plan.steering.enabled and steering_model is not None:
                    owner = (
                        left_device.device_id
                        if steering_model is left_model
                        else right_device.device_id
                    )
                    reason_list.append(
                        f"Applied steering vector {plan.steering.vector_id} at layer "
                        f"{plan.steering.target_layer} on {owner}."
                    )
                reasons = tuple(reason_list)
                return replace(plan, stages=stages, reasons=reasons), RouteDecision(
                    execution_mode=ExecutionMode.LAYER_PIPELINE,
                    selected_device_id=left_device.device_id,
                    selected_model_id=left_model.model_id,
                    fallback_device_ids=(),
                    reasons=reasons,
                    route_score=0.80,
                )
        raise ValueError("no compatible layer pipeline found")

    def _route_indexed_pipeline(
        self,
        plan: ExecutionPlan,
        profile: TaskProfile,
        devices: list[Device],
    ) -> tuple[ExecutionPlan, RouteDecision] | None:
        groups: dict[
            tuple[str, int, str, str, str, str, str],
            list[tuple[Device, ModelCapability]],
        ] = {}
        for device in devices:
            if not self._device_usable(device):
                continue
            for model in device.models:
                segment = model.segment
                if (
                    segment is None
                    or segment.stage_count <= 0
                    or not model.supports_layer_pipeline
                    or profile.task_class not in model.task_classes
                    or not self._target_matches_device(device, model)
                ):
                    continue
                boundary_format = segment.boundary_format or model.boundary_format
                key = (
                    segment.pipeline_id,
                    segment.stage_count,
                    model.model_family,
                    model.model_version,
                    model.tokenizer_id,
                    model.precision,
                    boundary_format,
                )
                groups.setdefault(key, []).append((device, model))

        placements: list[
            tuple[
                int,
                int,
                str,
                str,
                tuple[PipelineStage, ...],
                tuple[str, ...],
            ]
        ] = []
        for key, records in sorted(groups.items()):
            pipeline_id, stage_count, *_identity = key
            if plan.pipeline_id and pipeline_id != plan.pipeline_id:
                continue
            by_device: dict[str, dict[int, ModelCapability]] = {}
            device_by_id: dict[str, Device] = {}
            for device, model in records:
                segment = model.segment
                assert segment is not None
                if not (0 <= segment.stage_index < stage_count):
                    continue
                device_by_id[device.device_id] = device
                by_device.setdefault(device.device_id, {})[segment.stage_index] = model

            prefix_devices = sorted(
                (
                    device
                    for device in device_by_id.values()
                    if device.device_type != "phone" and device.platform != "android"
                ),
                key=lambda device: device.device_id,
            )
            suffix_devices = sorted(
                (
                    device
                    for device in device_by_id.values()
                    if device.device_type == "phone" or device.platform == "android"
                ),
                key=lambda device: (
                    device.device_id != plan.origin_device_id,
                    device.device_id,
                ),
            )
            if not prefix_devices:
                prefix_devices = sorted(device_by_id.values(), key=lambda d: d.device_id)

            # For the recovered four-stage demo, S0+S1 is the minimum initial
            # laptop prefix. The architecture still accepts one-stage prefixes
            # for other explicitly indexed pipelines.
            minimum_cut = 2 if stage_count == 4 else 1
            for prefix in prefix_devices:
                candidate_suffixes: list[Device | None] = [*suffix_devices, None]
                for suffix in candidate_suffixes:
                    cuts = (
                        range(minimum_cut, stage_count)
                        if suffix is not None and suffix.device_id != prefix.device_id
                        else ()
                    )
                    for cut in (*cuts, stage_count):
                        selected: list[tuple[Device, ModelCapability]] = []
                        complete = True
                        for stage_index in range(stage_count):
                            owner = prefix if stage_index < cut or suffix is None else suffix
                            model = by_device.get(owner.device_id, {}).get(stage_index)
                            if model is None:
                                complete = False
                                break
                            selected.append((owner, model))
                        if not complete or not self._indexed_chain_compatible(selected):
                            continue
                        memory_by_device: dict[str, int] = {}
                        for owner, model in selected:
                            memory_by_device[owner.device_id] = (
                                memory_by_device.get(owner.device_id, 0)
                                + model.min_memory_mb
                            )
                        if any(
                            device_by_id[device_id].health.available_memory_mb <= 0
                            or required
                            > device_by_id[device_id].health.available_memory_mb
                            for device_id, required in memory_by_device.items()
                        ):
                            continue
                        if not self._pipeline_steering_compatible(plan, selected):
                            continue

                        stages = tuple(
                            self._pipeline_stage(index, owner, model)
                            for index, (owner, model) in enumerate(selected)
                        )
                        suffix_count = stage_count - cut
                        phone_id = suffix.device_id if suffix_count and suffix else "none"
                        memory_parts = []
                        for owner in dict.fromkeys(item[0] for item in selected):
                            required = memory_by_device[owner.device_id]
                            assigned = [
                                f"S{index}"
                                for index, (stage_owner, _model) in enumerate(selected)
                                if stage_owner.device_id == owner.device_id
                            ]
                            memory_parts.append(
                                f"{owner.device_id} {'+'.join(assigned)} requires "
                                f"{required} MB of {owner.health.available_memory_mb} MB"
                            )
                        reasons = (
                            *plan.reasons,
                            f"Selected {pipeline_id} cut {cut}+{suffix_count}: "
                            f"laptop prefix={prefix.device_id}, phone suffix={phone_id}; "
                            + "; ".join(memory_parts)
                            + ".",
                            "Stage memory uses conservative artifact-size-plus-runtime-margin "
                            "estimates, not physical measurements; placement has at most one "
                            "cross-device transition.",
                        )
                        placements.append(
                            (
                                -suffix_count,
                                cut,
                                prefix.device_id,
                                phone_id,
                                stages,
                                reasons,
                            )
                        )

        if not placements:
            return None
        _suffix_rank, _cut, _prefix, _phone, stages, reasons = min(placements)
        routed = replace(plan, stages=stages, reasons=reasons)
        return routed, RouteDecision(
            execution_mode=ExecutionMode.LAYER_PIPELINE,
            selected_device_id=stages[0].selected_device_id,
            selected_model_id=stages[0].selected_model_id,
            fallback_device_ids=(),
            reasons=reasons,
            route_score=0.85,
        )

    @staticmethod
    def _target_matches_device(device: Device, model: ModelCapability) -> bool:
        advertised = model.target_compatibility_class
        hardware = device.hardware.compatibility_key
        if not advertised or not hardware:
            return True
        if advertised == hardware:
            return True
        # QAIRT context/runtime minor-version compatibility is established by
        # the Agent before it advertises an installed artifact. The Brain still
        # enforces the immutable OS/ABI/SoC/Hexagon target family.
        return advertised.split("-qairt-", 1)[0] == hardware.split("-qairt-", 1)[0]

    @staticmethod
    def _indexed_chain_compatible(
        selected: list[tuple[Device, ModelCapability]],
    ) -> bool:
        if not selected:
            return False
        segments = [model.segment for _device, model in selected]
        if any(segment is None for segment in segments):
            return False
        concrete = [segment for segment in segments if segment is not None]
        if not concrete[0].includes_embedding or not concrete[-1].includes_lm_head:
            return False
        if [segment.stage_index for segment in concrete] != list(range(len(concrete))):
            return False
        for left, right in zip(concrete, concrete[1:]):
            if left.stage_count != right.stage_count:
                return False
            if left.output_tensor and right.input_tensor:
                if left.output_tensor != right.input_tensor:
                    return False
            if (left.boundary_format or right.boundary_format) and (
                left.boundary_format != right.boundary_format
            ):
                return False
        return True

    def _pipeline_steering_compatible(
        self,
        plan: ExecutionPlan,
        selected: list[tuple[Device, ModelCapability]],
    ) -> bool:
        if not plan.steering.enabled:
            return True
        if self.steering_registry is None:
            return False
        for _device, model in selected:
            segment = model.segment
            assert segment is not None
            start = segment.transformer_start_layer
            end = segment.transformer_end_layer
            if start is not None and end is not None and (
                start <= plan.steering.target_layer <= end
            ):
                compatible, _reason = self.steering_registry.validate(
                    plan.steering, model
                )
                return compatible
        return False

    @staticmethod
    def _pipeline_stage(
        stage_index: int, device: Device, model: ModelCapability
    ) -> PipelineStage:
        segment = model.segment
        assert segment is not None
        return PipelineStage(
            stage_id=f"stage-{stage_index}",
            stage_index=stage_index,
            pipeline_id=segment.pipeline_id,
            selected_device_id=device.device_id,
            selected_model_id=model.model_id,
            start_layer=segment.transformer_start_layer,
            end_layer=segment.transformer_end_layer,
            model_family=model.model_family,
            model_version=model.model_version,
            tokenizer_id=model.tokenizer_id,
            precision=model.precision,
            boundary_format=segment.boundary_format or model.boundary_format,
            stage_count=segment.stage_count,
            input_tensor=segment.input_tensor,
            output_tensor=segment.output_tensor,
            includes_embedding=segment.includes_embedding,
            includes_lm_head=segment.includes_lm_head,
            min_memory_mb=model.min_memory_mb,
        )

    def _rank_models(
        self,
        profile: TaskProfile,
        devices: list[Device],
        steering: SteeringSpec,
        require_data_parallel: bool = False,
        origin_device_id: str = "",
        prefer_quality: bool = False,
    ) -> list[tuple[float, Device, ModelCapability, str]]:
        ranked = []
        for device in devices:
            if not self._device_usable(device):
                continue
            for model in device.models:
                if model.segment is not None:
                    continue
                if require_data_parallel and not model.supports_data_parallel:
                    continue
                if not self._has_model_memory(device, model):
                    continue
                if profile.task_class not in model.task_classes:
                    continue
                if profile.estimated_input_tokens > model.max_context_tokens:
                    continue
                if steering.enabled and self.steering_registry is not None:
                    ok, steering_reason = self.steering_registry.validate(
                        steering, model
                    )
                    if not ok:
                        continue
                else:
                    steering_reason = "steering disabled"
                score = self._score(device, model)
                reason = (
                    f"{model.model_id} supports {profile.task_class}; "
                    f"thermal={device.health.thermal_level:.2f}, "
                    f"memory={device.health.available_memory_mb} MB; {steering_reason}"
                )
                ranked.append((score, device, model, reason))
        if prefer_quality:
            ranked.sort(
                key=lambda item: (
                    -item[2].quality_score,
                    -item[0],
                    item[1].device_id,
                    item[2].model_id,
                )
            )
        else:
            ranked.sort(
                key=lambda item: (
                    item[1].device_id != origin_device_id,
                    -item[0],
                    item[1].device_id,
                    item[2].model_id,
                )
            )
        return ranked

    @staticmethod
    def _origin_route_reason(
        origin_device_id: str, selected_device_id: str
    ) -> tuple[str, ...]:
        if not origin_device_id:
            return ()
        if selected_device_id == origin_device_id:
            return (
                f"Origin preference selected {origin_device_id}: compatible local capacity is available.",
            )
        return (
            f"Origin {origin_device_id} has no eligible compatible local capacity; selected {selected_device_id}.",
        )

    @staticmethod
    def _has_model_memory(device: Device, model: ModelCapability) -> bool:
        available = device.health.available_memory_mb
        return available > 0 and available >= model.min_memory_mb

    def _device_usable(self, device: Device) -> bool:
        if not device.health.reachable:
            return False
        return device.health.status in {
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.STALE,
        }

    def _eligible_health_tier(self, devices: list[Device]) -> list[Device]:
        normal = [
            device
            for device in devices
            if device.health.reachable
            and device.health.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}
        ]
        if normal:
            return normal
        return [
            device
            for device in devices
            if device.health.reachable and device.health.status == HealthStatus.STALE
        ]

    def _score(self, device: Device, model: ModelCapability) -> float:
        observed_load = [
            value
            for value in (
                device.health.thermal_level,
                device.health.accelerator_utilization,
            )
            if value >= 0
        ]
        health = (
            1.0 - max(observed_load) * 0.5 if observed_load else 0.5
        )
        memory = (
            min(device.health.available_memory_mb / 8192, 1.0)
            if device.health.available_memory_mb > 0
            else 0.5
        )
        readiness = 1.0 if model.warm else 0.5
        latency = (
            0.5
            if device.health.network_rtt_ms < 0
            else 1.0 / (1.0 + device.health.network_rtt_ms / 100.0)
        )
        return round(
            0.35 * model.quality_score
            + 0.25 * health
            + 0.20 * latency
            + 0.10 * memory
            + 0.10 * readiness,
            4,
        )

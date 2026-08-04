from __future__ import annotations

from dataclasses import replace

from .models import (
    Device,
    ExecutionMode,
    ExecutionPlan,
    HealthStatus,
    ModelCapability,
    PipelineStage,
    PlannedTask,
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
        ranked = self._rank_models(profile, devices, plan.steering)
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
        reasons = (*plan.reasons, reason, f"Fallbacks: {', '.join(fallbacks) or 'none'}.")
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
        ranked = self._rank_models(profile, devices, plan.steering)
        if not ranked:
            raise ValueError("no eligible device/model for data-parallel task")
        routed_tasks = []
        reasons = list(plan.reasons)
        for idx, task in enumerate(plan.tasks):
            score, device, model, reason = ranked[idx % len(ranked)]
            fallbacks = tuple(item[1].device_id for item in ranked if item[1].device_id != device.device_id)[:2]
            routed_tasks.append(
                replace(
                    task,
                    selected_device_id=device.device_id,
                    selected_model_id=model.model_id,
                    fallback_device_ids=fallbacks,
                )
            )
            reasons.append(f"{task.shard_id} -> {device.device_id}/{model.model_id}: {reason}")
        first = routed_tasks[0]
        routed_plan = replace(plan, tasks=tuple(routed_tasks))
        return routed_plan, RouteDecision(
            execution_mode=ExecutionMode.DATA_PARALLEL,
            selected_device_id=first.selected_device_id,
            selected_model_id=first.selected_model_id,
            fallback_device_ids=first.fallback_device_ids,
            reasons=tuple(reasons),
            route_score=sum(item[0] for item in ranked[: len(routed_tasks)]) / min(len(ranked), len(routed_tasks)),
        )

    def _route_pipeline(
        self,
        plan: ExecutionPlan,
        profile: TaskProfile,
        devices: list[Device],
    ) -> tuple[ExecutionPlan, RouteDecision]:
        segments: list[tuple[Device, ModelCapability]] = []
        for device in devices:
            if not self._device_usable(device):
                continue
            for model in device.models:
                if model.segment and profile.task_class in model.task_classes:
                    segments.append((device, model))
        for left_device, left_model in segments:
            left = left_model.segment
            if left is None or not left.includes_embedding:
                continue
            for right_device, right_model in segments:
                right = right_model.segment
                if right is None or not right.includes_lm_head:
                    continue
                contiguous = (
                    left.pipeline_id == right.pipeline_id
                    and left.end_layer == right.start_layer
                    and left.start_layer == 0
                    and right.end_layer == right.total_layers
                )
                if not contiguous:
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
                    ),
                    PipelineStage(
                        stage_id="stage-2",
                        stage_index=1,
                        pipeline_id=right.pipeline_id,
                        selected_device_id=right_device.device_id,
                        selected_model_id=right_model.model_id,
                        start_layer=right.start_layer,
                        end_layer=right.end_layer,
                    ),
                )
                reasons = (
                    *plan.reasons,
                    f"Selected layer pipeline {left.pipeline_id}: "
                    f"{left_device.device_id} layers {left.start_layer}..{left.end_layer}, "
                    f"{right_device.device_id} layers {right.start_layer}..{right.end_layer}.",
                )
                return replace(plan, stages=stages, reasons=reasons), RouteDecision(
                    execution_mode=ExecutionMode.LAYER_PIPELINE,
                    selected_device_id=left_device.device_id,
                    selected_model_id=left_model.model_id,
                    fallback_device_ids=(),
                    reasons=reasons,
                    route_score=0.80,
                )
        raise ValueError("no compatible layer pipeline found")

    def _rank_models(
        self,
        profile: TaskProfile,
        devices: list[Device],
        steering: SteeringSpec,
    ) -> list[tuple[float, Device, ModelCapability, str]]:
        ranked = []
        for device in devices:
            if not self._device_usable(device):
                continue
            for model in device.models:
                if model.segment is not None:
                    continue
                if profile.task_class not in model.task_classes:
                    continue
                if profile.estimated_input_tokens > model.max_context_tokens:
                    continue
                if steering.enabled and self.steering_registry is not None:
                    ok, steering_reason = self.steering_registry.validate(steering, model)
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
        ranked.sort(key=lambda item: (-item[0], item[1].device_id, item[2].model_id))
        return ranked

    def _device_usable(self, device: Device) -> bool:
        if not device.health.reachable:
            return False
        return device.health.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}

    def _score(self, device: Device, model: ModelCapability) -> float:
        health = 1.0 - max(device.health.thermal_level, device.health.accelerator_utilization) * 0.5
        memory = min(device.health.available_memory_mb / 8192, 1.0)
        readiness = 1.0 if model.warm else 0.5
        latency = 1.0 / (1.0 + device.health.network_rtt_ms / 100.0)
        return round(
            0.35 * model.quality_score + 0.25 * health + 0.20 * latency + 0.10 * memory + 0.10 * readiness,
            4,
        )


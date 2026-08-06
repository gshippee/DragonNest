"""Compute / memory / communication tradeoff classification for the fleet.

Every registered device is scored on three independent resource axes and
labeled with whichever axis is currently its tightest constraint (its
"regime"). The same axes gate whether each advertised model capability is
actually achievable on that device right now, and whether any two devices
can jointly serve a split layer-pipeline model without the network becoming
the limiting factor.

This mirrors the eligibility predicates already used by `DeterministicRouter`
(memory gate, task-class membership, pipeline contiguity) but reports them as
a standalone, read-only view instead of a live routing decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .models import Device, ModelCapability

BALANCED_THRESHOLD = 0.65
"""A device's tightest axis must clear this score to be called "balanced"
rather than bound by that axis."""

PIPELINE_LATENCY_BUDGET_MS = 120
"""Combined RTT above this budget makes a two-device layer pipeline
communication-bound rather than achievable."""

_NPU_SCORE = {"available": 1.0, "not_probed": 0.5, "unavailable": 0.0}


@dataclass(frozen=True)
class DeviceRegime:
    device_id: str
    display_name: str
    compute: float
    memory: float
    communication: float
    bottleneck: str
    reasons: tuple[str, str, str]


@dataclass(frozen=True)
class CapabilityAchievability:
    device_id: str
    model_id: str
    task_classes: tuple[str, ...]
    achievable: bool
    limiting_factor: str | None
    detail: str


@dataclass(frozen=True)
class PipelineOpportunity:
    pipeline_id: str
    left_device_id: str
    right_device_id: str
    combined_rtt_ms: float
    achievable: bool
    limiting_factor: str | None


def _compute_score(device: Device) -> tuple[float, str]:
    load = [
        value
        for value in (
            device.health.cpu_utilization,
            device.health.accelerator_utilization,
            device.health.thermal_level,
        )
        if value >= 0
    ]
    headroom = 1.0 - max(load) if load else 0.5
    cores = device.hardware.cpu_core_count
    core_capacity = min(cores / 8, 1.0) if cores > 0 else 0.5
    npu = _NPU_SCORE.get(device.hardware.npu_status, 0.5)
    score = round(0.5 * headroom + 0.3 * core_capacity + 0.2 * npu, 4)
    detail = (
        f"compute headroom {headroom:.2f}, "
        f"{cores or 'unknown'} CPU cores, NPU {device.hardware.npu_status}"
    )
    return score, detail


def _memory_score(device: Device) -> tuple[float, str]:
    total = device.total_memory_mb
    available = device.health.available_memory_mb
    if total > 0:
        ratio = max(0.0, min(available / total, 1.0))
    elif available > 0:
        ratio = min(available / 8192, 1.0)
    else:
        ratio = 0.5
    detail = f"{available} MB available of {total or 'unknown'} MB total"
    return round(ratio, 4), detail


def _communication_score(device: Device) -> tuple[float, str]:
    if not device.health.reachable:
        return 0.0, "device unreachable"
    rtt = device.health.network_rtt_ms
    if rtt < 0:
        return 0.5, "network RTT unknown"
    score = 1.0 / (1.0 + rtt / 50.0)
    return round(score, 4), f"network RTT {rtt:.0f} ms"


def classify_device(device: Device) -> DeviceRegime:
    compute, compute_detail = _compute_score(device)
    memory, memory_detail = _memory_score(device)
    communication, communication_detail = _communication_score(device)
    scores = {
        "compute": compute,
        "memory": memory,
        "communication": communication,
    }
    if not device.health.reachable:
        tightest_axis = "communication"
    else:
        tightest_axis = min(scores, key=scores.get)
    bottleneck = "balanced" if scores[tightest_axis] >= BALANCED_THRESHOLD else tightest_axis
    return DeviceRegime(
        device_id=device.device_id,
        display_name=device.display_name,
        compute=compute,
        memory=memory,
        communication=communication,
        bottleneck=bottleneck,
        reasons=(compute_detail, memory_detail, communication_detail),
    )


def _model_achievability(device: Device, model: ModelCapability) -> CapabilityAchievability:
    reasons: list[str] = []
    limiting: str | None = None

    if not device.health.reachable:
        limiting = "communication"
        reasons.append("device is unreachable")

    memory_ok = device.health.available_memory_mb >= model.min_memory_mb
    if not memory_ok:
        limiting = limiting or "memory"
        reasons.append(
            f"needs {model.min_memory_mb} MB, {device.health.available_memory_mb} MB available"
        )

    accelerator_ok = True
    non_cpu_accelerators = [a for a in model.supported_accelerators if a != "cpu"]
    if non_cpu_accelerators and "cpu" not in model.supported_accelerators:
        accelerator_ok = device.hardware.npu_status == "available"
        if not accelerator_ok:
            limiting = limiting or "compute"
            reasons.append(
                f"requires {', '.join(non_cpu_accelerators)}, "
                f"NPU status is {device.hardware.npu_status}"
            )

    achievable = device.health.reachable and memory_ok and accelerator_ok
    return CapabilityAchievability(
        device_id=device.device_id,
        model_id=model.model_id,
        task_classes=model.task_classes,
        achievable=achievable,
        limiting_factor=None if achievable else limiting,
        detail="; ".join(reasons) if reasons else "memory, compute, and reachability all clear",
    )


def capability_matrix(devices: Sequence[Device]) -> list[CapabilityAchievability]:
    return [
        _model_achievability(device, model)
        for device in devices
        for model in device.models
        if model.segment is None
    ]


def pipeline_opportunities(devices: Sequence[Device]) -> list[PipelineOpportunity]:
    segments = [
        (device, model)
        for device in devices
        for model in device.models
        if model.segment is not None and model.supports_layer_pipeline
    ]
    opportunities: list[PipelineOpportunity] = []
    for left_device, left_model in segments:
        left = left_model.segment
        assert left is not None
        if not left.includes_embedding or left.start_layer != 0:
            continue
        for right_device, right_model in segments:
            if right_device.device_id == left_device.device_id:
                continue
            right = right_model.segment
            assert right is not None
            contiguous = (
                right.includes_lm_head
                and left.pipeline_id == right.pipeline_id
                and left.end_layer == right.start_layer
                and right.end_layer == right.total_layers
                and left_model.model_family == right_model.model_family
                and left_model.model_version == right_model.model_version
                and left_model.tokenizer_id == right_model.tokenizer_id
                and left_model.precision == right_model.precision
                and left_model.boundary_format == right_model.boundary_format
            )
            if not contiguous:
                continue
            reachable = left_device.health.reachable and right_device.health.reachable
            rtts = [
                value
                for value in (
                    left_device.health.network_rtt_ms,
                    right_device.health.network_rtt_ms,
                )
                if value >= 0
            ]
            combined_rtt = sum(rtts) if rtts else -1.0
            achievable = reachable and (
                combined_rtt < 0 or combined_rtt <= PIPELINE_LATENCY_BUDGET_MS
            )
            opportunities.append(
                PipelineOpportunity(
                    pipeline_id=left.pipeline_id,
                    left_device_id=left_device.device_id,
                    right_device_id=right_device.device_id,
                    combined_rtt_ms=combined_rtt,
                    achievable=achievable,
                    limiting_factor=None if achievable else "communication",
                )
            )
    return opportunities


def build_regime_report(devices: Sequence[Device]) -> dict:
    device_regimes = [classify_device(device) for device in devices]
    capabilities = capability_matrix(devices)
    pipelines = pipeline_opportunities(devices)

    task_classes: dict[str, dict[str, list[str]]] = {}
    for row in capabilities:
        for task_class in row.task_classes:
            bucket = task_classes.setdefault(
                task_class, {"achievable_on": [], "blocked_on": []}
            )
            key = "achievable_on" if row.achievable else "blocked_on"
            if row.device_id not in bucket[key]:
                bucket[key].append(row.device_id)

    return {
        "devices": [asdict(regime) for regime in device_regimes],
        "capabilities": [asdict(row) for row in capabilities],
        "pipelines": [asdict(item) for item in pipelines],
        "task_classes": task_classes,
    }

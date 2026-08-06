from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from dragon_nest.config import load_devices
from dragon_nest.dashboard import create_dashboard_app
from dragon_nest.models import (
    Device,
    HardwareInventory,
    HealthState,
    ModelCapability,
)
from dragon_nest.regimes import (
    build_regime_report,
    capability_matrix,
    classify_device,
    pipeline_opportunities,
)
from dragon_nest.transport.brain import BrainService


ROOT = Path(__file__).resolve().parents[1]


def _device(**overrides) -> Device:
    defaults = dict(
        device_id="dev-01",
        display_name="Device",
        device_type="phone",
        platform="android",
        total_memory_mb=8192,
        health=HealthState(available_memory_mb=4096, network_rtt_ms=10, reachable=True),
        models=(),
        hardware=HardwareInventory(),
    )
    defaults.update(overrides)
    return Device(**defaults)


def test_classify_device_flags_memory_bottleneck_when_headroom_is_low():
    device = _device(
        total_memory_mb=8192,
        health=HealthState(
            thermal_level=0.1,
            cpu_utilization=0.1,
            accelerator_utilization=0.1,
            available_memory_mb=200,
            network_rtt_ms=5,
            reachable=True,
        ),
    )
    regime = classify_device(device)
    assert regime.bottleneck == "memory"
    assert regime.memory < regime.compute
    assert regime.memory < regime.communication


def test_classify_device_flags_compute_bottleneck_under_high_load():
    device = _device(
        total_memory_mb=8192,
        health=HealthState(
            thermal_level=0.95,
            cpu_utilization=0.95,
            accelerator_utilization=0.95,
            available_memory_mb=8192,
            network_rtt_ms=5,
            reachable=True,
        ),
    )
    regime = classify_device(device)
    assert regime.bottleneck == "compute"


def test_classify_device_flags_communication_bottleneck_over_slow_link():
    device = _device(
        total_memory_mb=8192,
        health=HealthState(
            thermal_level=0.1,
            cpu_utilization=0.1,
            accelerator_utilization=0.1,
            available_memory_mb=8192,
            network_rtt_ms=900,
            reachable=True,
        ),
        hardware=HardwareInventory(cpu_core_count=8, npu_status="available"),
    )
    regime = classify_device(device)
    assert regime.bottleneck == "communication"


def test_classify_device_unreachable_device_is_communication_bound():
    device = _device(health=HealthState(reachable=False))
    regime = classify_device(device)
    assert regime.bottleneck == "communication"
    assert regime.communication == 0.0


def test_classify_device_is_balanced_when_every_axis_has_headroom():
    device = _device(
        total_memory_mb=8192,
        health=HealthState(
            thermal_level=0.1,
            cpu_utilization=0.1,
            accelerator_utilization=0.1,
            available_memory_mb=8000,
            network_rtt_ms=5,
            reachable=True,
        ),
        hardware=HardwareInventory(cpu_core_count=8, npu_status="available"),
    )
    regime = classify_device(device)
    assert regime.bottleneck == "balanced"


def test_capability_matrix_flags_memory_blocked_model():
    model = ModelCapability(
        model_id="big-model",
        model_family="mock",
        role="large_reasoning",
        task_classes=("chat_qa",),
        max_context_tokens=4096,
        warm=True,
        quality_score=0.9,
        min_memory_mb=8000,
    )
    device = _device(health=HealthState(available_memory_mb=1024, reachable=True), models=(model,))
    rows = capability_matrix([device])
    assert len(rows) == 1
    assert rows[0].achievable is False
    assert rows[0].limiting_factor == "memory"


def test_capability_matrix_flags_compute_blocked_model_needing_npu():
    model = ModelCapability(
        model_id="npu-model",
        model_family="mock",
        role="accelerated",
        task_classes=("chat_qa",),
        max_context_tokens=4096,
        warm=True,
        quality_score=0.9,
        min_memory_mb=100,
        supported_accelerators=("npu",),
    )
    device = _device(
        health=HealthState(available_memory_mb=4096, reachable=True),
        hardware=HardwareInventory(npu_status="unavailable"),
        models=(model,),
    )
    rows = capability_matrix([device])
    assert rows[0].achievable is False
    assert rows[0].limiting_factor == "compute"


def test_capability_matrix_marks_achievable_when_all_gates_clear():
    model = ModelCapability(
        model_id="ok-model",
        model_family="mock",
        role="small_chat",
        task_classes=("chat_qa",),
        max_context_tokens=4096,
        warm=True,
        quality_score=0.6,
        min_memory_mb=100,
    )
    device = _device(health=HealthState(available_memory_mb=4096, reachable=True), models=(model,))
    rows = capability_matrix([device])
    assert rows[0].achievable is True
    assert rows[0].limiting_factor is None


def test_pipeline_opportunities_from_dev_fabric_are_communication_achievable():
    devices = load_devices(ROOT / "configs/dev-fabric.yaml")
    opportunities = pipeline_opportunities(devices)
    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert {opportunity.left_device_id, opportunity.right_device_id} == {"phone-01", "pc-01"}
    assert opportunity.achievable is True
    assert opportunity.combined_rtt_ms == 24


def test_pipeline_opportunity_blocked_when_combined_rtt_exceeds_budget():
    left = _device(
        device_id="left",
        health=HealthState(network_rtt_ms=200, reachable=True),
        models=(
            ModelCapability(
                model_id="part-a",
                model_family="qwen3",
                role="pipeline_segment",
                task_classes=("chat_qa",),
                max_context_tokens=2048,
                warm=True,
                quality_score=0.7,
                model_version="v1",
                tokenizer_id="tok",
                precision="fp16",
                boundary_format="raw-fp32",
                supports_layer_pipeline=True,
                segment=__import__("dragon_nest.models", fromlist=["ModelSegment"]).ModelSegment(
                    pipeline_id="split-14",
                    start_layer=0,
                    end_layer=14,
                    total_layers=28,
                    includes_embedding=True,
                ),
            ),
        ),
    )
    right = _device(
        device_id="right",
        health=HealthState(network_rtt_ms=200, reachable=True),
        models=(
            ModelCapability(
                model_id="part-b",
                model_family="qwen3",
                role="pipeline_segment",
                task_classes=("chat_qa",),
                max_context_tokens=2048,
                warm=True,
                quality_score=0.7,
                model_version="v1",
                tokenizer_id="tok",
                precision="fp16",
                boundary_format="raw-fp32",
                supports_layer_pipeline=True,
                segment=__import__("dragon_nest.models", fromlist=["ModelSegment"]).ModelSegment(
                    pipeline_id="split-14",
                    start_layer=14,
                    end_layer=28,
                    total_layers=28,
                    includes_lm_head=True,
                ),
            ),
        ),
    )
    opportunities = pipeline_opportunities([left, right])
    assert len(opportunities) == 1
    assert opportunities[0].achievable is False
    assert opportunities[0].limiting_factor == "communication"


def test_build_regime_report_summarizes_task_classes():
    devices = load_devices(ROOT / "configs/dev-fabric.yaml")
    report = build_regime_report(devices)
    assert {device["device_id"] for device in report["devices"]} == {"phone-01", "pc-01"}
    assert "chat_qa" in report["task_classes"]
    assert set(report["task_classes"]["chat_qa"]["achievable_on"]) == {"phone-01", "pc-01"}
    assert len(report["pipelines"]) == 1


def test_dashboard_serves_regimes_page_and_api():
    async def scenario() -> None:
        service = BrainService()
        for device in load_devices(ROOT / "configs/dev-fabric.yaml"):
            service.registry.register(device)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            page = await client.get("/regimes")
            report = (await client.get("/api/regimes")).json()

        assert page.status_code == 200
        assert "Tradeoff Map" in page.text
        assert "Cross-Device Pipeline Opportunities" in page.text
        assert {device["device_id"] for device in report["devices"]} == {"phone-01", "pc-01"}
        assert len(report["capabilities"]) == 2
        assert len(report["pipelines"]) == 1

    asyncio.run(scenario())

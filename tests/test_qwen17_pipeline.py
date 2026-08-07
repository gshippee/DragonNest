from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import grpc
import pytest

from dragon_nest.artifacts import ArtifactRegistry
from dragon_nest.classifier import RuleBasedTaskClassifier
from dragon_nest.models import (
    Device,
    HardwareInventory,
    HealthState,
    ModelCapability,
    ModelSegment,
)
from dragon_nest.pipeline_sessions import PipelineSessionKey, PipelineSessionStore
from dragon_nest.planner import ExecutionPlanner
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.proto import dragonnest_pb2_grpc as pb_grpc
from dragon_nest.router import DeterministicRouter
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ID = "qwen3-1.7b-w4a16-demo-v1"
MEMORY_MB = (1024, 768, 768, 1152)
TENSORS = (
    ("input_ids", "embedding"),
    ("embedding", "add_21844"),
    ("add_21844", "add_42314"),
    ("add_42314", "logits"),
)
LAYERS = ((None, None), (0, 9), (10, 19), (20, 27))


def _models(target: str, compatibility: str) -> tuple[ModelCapability, ...]:
    models = []
    for index in range(4):
        start, end = LAYERS[index]
        input_tensor, output_tensor = TENSORS[index]
        models.append(
            ModelCapability(
                model_id=f"qwen3-1.7b-s{index}-{target}",
                model_family="qwen3-1.7b",
                role="pipeline_segment",
                task_classes=("reasoning_analysis",),
                max_context_tokens=512,
                warm=False,
                quality_score=0.8,
                model_version="qwen3-1.7b-demo-v1-unpinned-main",
                tokenizer_id="Qwen/Qwen3-1.7B",
                precision="w4a16-name-w8a16-compile-observed",
                boundary_format="qnn-raw-tensor-v1",
                segment=ModelSegment(
                    pipeline_id=PIPELINE_ID,
                    stage_index=index,
                    stage_count=4,
                    transformer_start_layer=start,
                    transformer_end_layer=end,
                    total_layers=28,
                    includes_embedding=index == 0,
                    includes_lm_head=index == 3,
                    input_tensor=input_tensor,
                    output_tensor=output_tensor,
                    boundary_format="qnn-raw-tensor-v1",
                ),
                runtime_name="qnn",
                supported_accelerators=("htp",),
                min_memory_mb=MEMORY_MB[index],
                supports_data_parallel=False,
                supports_layer_pipeline=True,
                target_compatibility_class=compatibility,
            )
        )
    return tuple(models)


def _devices(phone_memory_mb: int) -> list[Device]:
    pc_compat = "windows-arm64-x1e-v73-qairt-2.45"
    phone_compat = "android-arm64-sm8750-v79-qairt-2.45"
    return [
        Device(
            "pc-01",
            "X Elite",
            "pc",
            "windows",
            32768,
            HealthState(available_memory_mb=12000),
            _models("xelite", pc_compat),
            HardwareInventory(compatibility_key=pc_compat),
        ),
        Device(
            "phone-01",
            "S25",
            "phone",
            "android",
            12288,
            HealthState(available_memory_mb=phone_memory_mb),
            _models("s25", phone_compat),
            HardwareInventory(compatibility_key=phone_compat),
        ),
    ]


def _route(phone_memory_mb: int):
    request = "Analyze this complex quality trade-off."
    profile = RuleBasedTaskClassifier().classify(request, preferred_mode="quality")
    plan = ExecutionPlanner().plan(
        request,
        profile,
        requested_execution_mode="layer_pipeline",
        origin_device_id="phone-01",
    )
    return DeterministicRouter().route(plan, profile, _devices(phone_memory_mb))


@pytest.mark.parametrize(
    ("phone_memory_mb", "expected_devices"),
    [
        (2000, ["pc-01", "pc-01", "phone-01", "phone-01"]),
        (1300, ["pc-01", "pc-01", "pc-01", "phone-01"]),
        (1000, ["pc-01", "pc-01", "pc-01", "pc-01"]),
    ],
)
def test_variable_single_cut_placement(phone_memory_mb, expected_devices):
    routed, decision = _route(phone_memory_mb)

    assert [stage.stage_index for stage in routed.stages] == [0, 1, 2, 3]
    assert [stage.selected_device_id for stage in routed.stages] == expected_devices
    transitions = sum(
        left.selected_device_id != right.selected_device_id
        for left, right in zip(routed.stages, routed.stages[1:])
    )
    assert transitions <= 1
    assert any("requires" in reason for reason in decision.reasons)
    assert any("not physical" in reason for reason in decision.reasons)


def test_cumulative_memory_rejects_individually_fitting_suffix():
    routed, _decision = _route(1500)

    assert MEMORY_MB[2] < 1500 and MEMORY_MB[3] < 1500
    assert MEMORY_MB[2] + MEMORY_MB[3] > 1500
    assert [stage.selected_device_id for stage in routed.stages] == [
        "pc-01",
        "pc-01",
        "pc-01",
        "phone-01",
    ]


def test_missing_intermediate_stage_is_rejected():
    devices = [
        replace(device, models=tuple(model for model in device.models if model.segment.stage_index != 2))
        for device in _devices(2000)
    ]
    request = "Analyze this complex quality trade-off."
    profile = RuleBasedTaskClassifier().classify(request, preferred_mode="quality")
    plan = ExecutionPlanner().plan(request, profile, requested_execution_mode="layer_pipeline")

    with pytest.raises(ValueError, match="no compatible layer pipeline"):
        DeterministicRouter().route(plan, profile, devices)


def test_incompatible_cross_target_boundary_is_rejected():
    devices = []
    for device in _devices(2000):
        models = []
        for model in device.models:
            if model.segment.stage_index == 2:
                model = replace(
                    model,
                    segment=replace(model.segment, input_tensor="wrong_boundary"),
                )
            models.append(model)
        devices.append(replace(device, models=tuple(models)))
    request = "Analyze this complex quality trade-off."
    profile = RuleBasedTaskClassifier().classify(request, preferred_mode="quality")
    plan = ExecutionPlanner().plan(request, profile, requested_execution_mode="layer_pipeline")

    with pytest.raises(ValueError, match="no compatible layer pipeline"):
        DeterministicRouter().route(plan, profile, devices)


def test_runtime_manifest_loads_all_eight_indexed_artifacts():
    registry = ArtifactRegistry.from_yaml(ROOT / "configs/model-artifacts.yaml")
    artifacts = [artifact for artifact in registry.all() if "qwen3-1.7b-s" in artifact.model_id]

    assert len(artifacts) == 8
    assert all(artifact.checksum.startswith("sha256:") for artifact in artifacts)
    assert all(len(artifact.checksum) == len("sha256:") + 64 for artifact in artifacts)
    assert {artifact.split_boundary.stage_index for artifact in artifacts} == {0, 1, 2, 3}
    s0 = registry.get("qwen3-1.7b-s0-s25").split_boundary
    assert s0 is not None
    assert s0.transformer_start_layer is None
    assert s0.transformer_end_layer is None
    assert s0.includes_embedding


def test_pipeline_session_lifecycle_and_cleanup():
    store = PipelineSessionStore()
    key = PipelineSessionKey("task-1", PIPELINE_ID, 1)
    store.begin_prefill(key)
    store.retain_outputs(key, {"past_key_0_out": b"key", "past_value_0_out": b"value"})
    store.complete_prefill(key)

    session = store.require_decode(key)
    assert session.kv_inputs == {"past_key_0_in": b"key", "past_value_0_in": b"value"}
    assert store.cleanup_task("task-1") == 1
    assert len(store) == 0
    with pytest.raises(RuntimeError, match="before prefill"):
        store.require_decode(key)


def test_public_grpc_path_runs_prefill_decode_and_cleans_sessions():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        config = AgentClientConfig(
            brain_target=f"127.0.0.1:{port}",
            heartbeat_interval_seconds=0.05,
            reconnect_initial_seconds=0.01,
            reconnect_max_seconds=0.05,
        )
        agents = [DeviceAgent(device, config) for device in _devices(2000)]
        tasks = [asyncio.create_task(agent.run_forever()) for agent in agents]
        try:
            await asyncio.wait_for(
                asyncio.gather(*(agent.registered.wait() for agent in agents)),
                timeout=3,
            )
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Analyze this complex quality trade-off.",
                        preferred_mode="quality",
                        execution_mode="layer_pipeline",
                        origin_device_id="phone-01",
                        timeout_ms=2_000,
                    )
                )
            assert response.success
            assert response.output_text == "xx"
            assert response.device_id == "phone-01"
            assert all(len(agent.pipeline_sessions) == 0 for agent in agents)
        finally:
            await asyncio.gather(*(agent.stop() for agent in agents))
            await asyncio.gather(*tasks, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())

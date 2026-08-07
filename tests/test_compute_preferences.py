from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import grpc

from dragon_nest.artifacts import ArtifactRegistry
from dragon_nest.models import (
    Device,
    ExecutionMode,
    HardwareInventory,
    HealthState,
    ModelCapability,
    ModelSegment,
)
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.proto import dragonnest_pb2_grpc as pb_grpc
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server


PIPELINE_ID = "qwen3-1.7b-w4a16-demo-v1"
STAGE_MEMORY_MB = (1024, 768, 768, 1152)


def _pipeline_models(target: str, compatibility: str) -> tuple[ModelCapability, ...]:
    layers = ((None, None), (0, 9), (10, 19), (20, 27))
    tensors = (
        ("input_ids", "embedding"),
        ("embedding", "add_21844"),
        ("add_21844", "add_42314"),
        ("add_42314", "logits"),
    )
    return tuple(
        ModelCapability(
            model_id=f"qwen3-1.7b-s{index}-{target}",
            model_family="qwen3-1.7b",
            role="pipeline_segment",
            task_classes=("chat_qa", "reasoning_analysis", "code_assistance"),
            max_context_tokens=512,
            warm=False,
            quality_score=0.80,
            model_version="qwen3-1.7b-demo-v1-unpinned-main",
            tokenizer_id="Qwen/Qwen3-1.7B",
            precision="w4a16-name-w8a16-compile-observed",
            boundary_format="qnn-raw-tensor-v1",
            segment=ModelSegment(
                pipeline_id=PIPELINE_ID,
                stage_index=index,
                stage_count=4,
                transformer_start_layer=layers[index][0],
                transformer_end_layer=layers[index][1],
                total_layers=28,
                includes_embedding=index == 0,
                includes_lm_head=index == 3,
                input_tensor=tensors[index][0],
                output_tensor=tensors[index][1],
                boundary_format="qnn-raw-tensor-v1",
            ),
            runtime_name="qnn",
            supported_accelerators=("htp",),
            min_memory_mb=STAGE_MEMORY_MB[index],
            supports_data_parallel=False,
            supports_layer_pipeline=True,
            target_compatibility_class=compatibility,
        )
        for index in range(4)
    )


def _devices() -> tuple[Device, Device]:
    phone_compat = "android-arm64-sm8750-v79-qairt-2.45"
    pc_compat = "windows-arm64-x1e-v73-qairt-2.45"
    phone = Device(
        "phone-01",
        "Galaxy S25 Ultra",
        "phone",
        "android",
        12_288,
        HealthState(available_memory_mb=6_000, thermal_level=0.2),
        (
            ModelCapability(
                "qwen3-0.6b-s25-base",
                "qwen3-0.6b",
                "small_chat",
                ("chat_qa", "reasoning_analysis", "code_assistance"),
                4096,
                True,
                0.60,
                min_memory_mb=128,
                runtime_name="mock",
            ),
            *_pipeline_models("s25", phone_compat),
        ),
        HardwareInventory(compatibility_key=phone_compat),
    )
    laptop = Device(
        "pc-01",
        "Snapdragon X Elite Laptop",
        "pc",
        "windows",
        32_768,
        HealthState(available_memory_mb=16_000, thermal_level=0.1),
        (
            ModelCapability(
                "qwen3-4b-genie",
                "qwen3",
                "large_reasoning",
                ("chat_qa", "reasoning_analysis", "code_assistance"),
                4096,
                False,
                0.95,
                min_memory_mb=4096,
                runtime_name="genie",
                runtime_version="QAIRT-2.48",
                supported_accelerators=("htp",),
                artifact_id="qwen3-4b-w4a16-xelite-v73-qairt248",
            ),
            *_pipeline_models("xelite", pc_compat),
        ),
        HardwareInventory(compatibility_key=pc_compat),
    )
    return phone, laptop


def _request(mode: str, text: str = "What is the capital of Japan?"):
    return pb.SubmitTaskRequest(
        request_text=text,
        preferred_mode=mode,
        execution_mode="auto",
        origin_device_id="phone-01",
        persona_id="balanced",
        timeout_ms=2_000,
    )


def test_explicit_and_auto_compute_preferences_are_deterministic():
    async def scenario() -> None:
        phone, laptop = _devices()
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        config = AgentClientConfig(
            brain_target=f"127.0.0.1:{port}",
            heartbeat_interval_seconds=60,
            reconnect_initial_seconds=0.01,
            reconnect_max_seconds=0.05,
        )
        artifacts = ArtifactRegistry({}, Path.cwd())
        agents = [
            DeviceAgent(device, config, artifacts=artifacts)
            for device in (phone, laptop)
        ]
        agent_tasks = [asyncio.create_task(agent.run_forever()) for agent in agents]

        def set_phone_memory(memory_mb: int) -> None:
            record = service.registry.get("phone-01")
            service.registry.heartbeat(
                "phone-01",
                replace(record.device.health, available_memory_mb=memory_mb),
                active_task_ids=record.active_task_ids,
                warm_model_ids=record.warm_model_ids,
                simulated_constraint=memory_mb != 6_000,
            )

        def set_pipeline_available(available: bool) -> None:
            for original in (phone, laptop):
                models = (
                    original.models
                    if available
                    else tuple(model for model in original.models if model.segment is None)
                )
                current = service.registry.get(original.device_id).device
                service.registry.register(replace(current, models=models))

        try:
            await asyncio.wait_for(
                asyncio.gather(*(agent.registered.wait() for agent in agents)),
                timeout=3,
            )
            async with grpc.aio.insecure_channel(config.brain_target) as channel:
                stub = pb_grpc.BrainControlStub(channel)

                local = await stub.SubmitTask(_request("local"))
                assert local.success and local.device_id == "phone-01"
                assert local.model_id == "qwen3-0.6b-s25-base"
                assert any("Local selected" in reason for reason in local.route_reasons)

                set_phone_memory(64)
                local_low = await stub.SubmitTask(_request("local"))
                assert not local_low.success
                assert local_low.error_code == "LOCAL_UNAVAILABLE"
                assert local_low.device_id == ""

                private_low = await stub.SubmitTask(_request("private"))
                assert not private_low.success
                assert private_low.error_code == "NO_ELIGIBLE_FALLBACK"

                for memory_mb, expected in (
                    (2_000, ["pc-01", "pc-01", "phone-01", "phone-01"]),
                    (1_300, ["pc-01", "pc-01", "pc-01", "phone-01"]),
                    (1_000, ["pc-01", "pc-01", "pc-01", "pc-01"]),
                ):
                    set_phone_memory(memory_mb)
                    elastic = await stub.SubmitTask(_request("elastic"))
                    assert elastic.success
                    plan = service.execution_plans[elastic.task_id]
                    assert plan.execution_mode == ExecutionMode.LAYER_PIPELINE
                    assert plan.pipeline_id == PIPELINE_ID
                    assert [stage.selected_device_id for stage in plan.stages] == expected

                set_pipeline_available(False)
                elastic_missing = await stub.SubmitTask(_request("elastic"))
                assert not elastic_missing.success
                assert elastic_missing.error_code == "ELASTIC_UNAVAILABLE"

                set_phone_memory(6_000)
                quality = await stub.SubmitTask(_request("quality"))
                assert quality.success and quality.device_id == "pc-01"
                assert quality.model_id == "qwen3-4b-genie"
                assert any("Quality selected" in reason for reason in quality.route_reasons)

                auto_local = await stub.SubmitTask(_request("auto"))
                assert auto_local.success and auto_local.device_id == "phone-01"
                assert any(
                    "Auto selected local execution" in reason
                    for reason in auto_local.route_reasons
                )

                set_phone_memory(64)
                auto_remote = await stub.SubmitTask(_request("auto"))
                assert auto_remote.success and auto_remote.device_id == "pc-01"
                assert auto_remote.model_id == "qwen3-4b-genie"
                assert any(
                    "Auto selected remote full model" in reason
                    for reason in auto_remote.route_reasons
                )

                set_phone_memory(6_000)
                complex_text = "Analyze and evaluate this architecture trade-off."
                auto_complex_fallback = await stub.SubmitTask(
                    _request("auto", complex_text)
                )
                assert auto_complex_fallback.success
                assert (
                    service.execution_plans[auto_complex_fallback.task_id].execution_mode
                    == ExecutionMode.SINGLE
                )
                assert any(
                    "Elastic was unavailable" in reason
                    for reason in auto_complex_fallback.route_reasons
                )

                set_pipeline_available(True)
                auto_complex = await stub.SubmitTask(_request("auto", complex_text))
                assert auto_complex.success
                assert (
                    service.execution_plans[auto_complex.task_id].execution_mode
                    == ExecutionMode.LAYER_PIPELINE
                )
                assert any(
                    "Auto selected elastic execution" in reason
                    for reason in auto_complex.route_reasons
                )
        finally:
            await asyncio.gather(*(agent.stop() for agent in agents))
            await asyncio.gather(*agent_tasks, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_quality_is_single_and_does_not_inflate_classifier_complexity():
    from dragon_nest.classifier import RuleBasedTaskClassifier
    from dragon_nest.planner import ExecutionPlanner

    request = "What is the capital of Japan?"
    profile = RuleBasedTaskClassifier().classify(request, preferred_mode="quality")
    plan = ExecutionPlanner().plan(request, profile, preferred_mode="quality")

    assert profile.complexity == "low"
    assert not profile.layer_parallel_candidate
    assert plan.execution_mode == ExecutionMode.SINGLE

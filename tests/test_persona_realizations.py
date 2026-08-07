from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import grpc

from dragon_nest.artifacts import ArtifactRegistry
from dragon_nest.models import Device, HealthState, ModelCapability
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.proto import dragonnest_pb2_grpc as pb_grpc
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server


def _model(model_id: str, profile_id: str = "") -> ModelCapability:
    return ModelCapability(
        model_id=model_id,
        model_family="qwen3",
        role="small_chat",
        task_classes=("chat_qa",),
        max_context_tokens=512,
        warm=False,
        quality_score=0.7,
        runtime_name="genie",
        runtime_version="QAIRT-2.45 / GenieX-0.3.5",
        supported_accelerators=("htp",),
        min_memory_mb=128,
        supports_steering=False,
        artifact_id=f"artifact-{model_id}",
        steering_modes=(("baked_profile",) if profile_id else ("none",)),
        behavior_profile_ids=((profile_id,) if profile_id else ()),
    )


def _request(persona_id: str) -> pb.SubmitTaskRequest:
    return pb.SubmitTaskRequest(
        request_text="What is the capital of Japan?",
        preferred_mode="local",
        execution_mode="auto",
        origin_device_id="phone-01",
        persona_id=persona_id,
        timeout_ms=2_000,
    )


def test_classic_grpc_routes_semantic_profiles_to_exact_baked_artifacts():
    async def scenario() -> None:
        models = (
            _model("qwen3-0.6b-s25-base"),
            _model("qwen3-0.6b-s25-concise", "concise"),
            _model("qwen3-0.6b-s25-detailed", "detailed"),
        )
        phone = Device(
            device_id="phone-01",
            display_name="Galaxy S25 Ultra",
            device_type="phone",
            platform="android",
            total_memory_mb=12_288,
            health=HealthState(available_memory_mb=6_000),
            models=models,
        )
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        config = AgentClientConfig(
            brain_target=f"127.0.0.1:{port}",
            heartbeat_interval_seconds=60,
            reconnect_initial_seconds=0.01,
            reconnect_max_seconds=0.05,
        )
        agent = DeviceAgent(phone, config, artifacts=ArtifactRegistry({}, Path.cwd()))
        agent_task = asyncio.create_task(agent.run_forever())
        try:
            await asyncio.wait_for(agent.registered.wait(), timeout=3)
            async with grpc.aio.insecure_channel(config.brain_target) as channel:
                stub = pb_grpc.BrainControlStub(channel)
                expected = {
                    "balanced": ("qwen3-0.6b-s25-base", "none"),
                    "concise": ("qwen3-0.6b-s25-concise", "baked_profile"),
                    "detailed": ("qwen3-0.6b-s25-detailed", "baked_profile"),
                }
                for persona_id, (model_id, realization) in expected.items():
                    response = await stub.SubmitTask(_request(persona_id))
                    assert response.success
                    assert response.model_id == model_id
                    assert response.device_id == "phone-01"
                    plan = service.execution_plans[response.task_id]
                    assert plan.behavior_profile_id == persona_id
                    assert plan.profile_realization == realization
                    assert not plan.steering.enabled
                    assert plan.steering.vector_id == ""
                    if realization == "baked_profile":
                        assert plan.steering.behavior_profile_id == persona_id
                        assert any(
                            f"profile {persona_id} realized by baked artifact"
                            in reason
                            for reason in response.route_reasons
                        )

                current = service.registry.get("phone-01").device
                service.registry.register(
                    replace(
                        current,
                        models=tuple(
                            model
                            for model in current.models
                            if model.model_id != "qwen3-0.6b-s25-detailed"
                        ),
                    )
                )
                missing = await stub.SubmitTask(_request("detailed"))
                assert not missing.success
                assert missing.error_code == "PROFILE_UNAVAILABLE"
                assert "no eligible device/model" in missing.error_message.lower()
        finally:
            await agent.stop()
            await asyncio.gather(agent_task, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import grpc

from dragon_nest.artifacts import ArtifactRegistry
from dragon_nest.models import Device, ExecutionMode, HealthState, ModelCapability
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.proto import dragonnest_pb2_grpc as pb_grpc
from dragon_nest.telemetry import TelemetrySnapshot
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server


class MutableTelemetry:
    def __init__(self, health: HealthState):
        self.available_memory_mb = health.available_memory_mb
        self._baseline_memory_mb = health.available_memory_mb
        self._health = health

    def sample(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            health=replace(
                self._health,
                available_memory_mb=self.available_memory_mb,
            ),
            simulated_constraint=self.available_memory_mb != self._baseline_memory_mb,
        )


def _demo_devices() -> tuple[Device, Device]:
    phone = Device(
        device_id="phone-01",
        display_name="Galaxy S25 Ultra",
        device_type="phone",
        platform="android",
        total_memory_mb=12_288,
        health=HealthState(
            available_memory_mb=6_000,
            thermal_level=0.2,
            battery_pct=80,
            reachable=True,
        ),
        models=(
            ModelCapability(
                model_id="android-mock-v1",
                model_family="mock",
                role="small_chat",
                task_classes=("chat_qa",),
                max_context_tokens=4_096,
                warm=True,
                quality_score=0.60,
                min_memory_mb=128,
                runtime_name="mock",
                supported_accelerators=("cpu",),
            ),
        ),
    )
    laptop = Device(
        device_id="pc-01",
        display_name="Snapdragon X Elite Laptop",
        device_type="pc",
        platform="windows",
        total_memory_mb=32_768,
        health=HealthState(
            available_memory_mb=16_000,
            thermal_level=0.1,
            battery_pct=100,
            charging=True,
            reachable=True,
        ),
        models=(
            ModelCapability(
                model_id="qwen3-4b-genie",
                model_family="qwen3",
                role="large_reasoning",
                task_classes=("chat_qa",),
                max_context_tokens=4_096,
                warm=False,
                quality_score=0.85,
                min_memory_mb=4_096,
                runtime_name="genie",
                runtime_version="QAIRT-2.48",
                supported_accelerators=("htp",),
                artifact_id="qwen3-4b-w4a16-xelite-v73-qairt248",
            ),
        ),
    )
    return phone, laptop


def _request(preferred_mode: str = "auto") -> pb.SubmitTaskRequest:
    return pb.SubmitTaskRequest(
        request_text="What is the capital of Japan?",
        preferred_mode=preferred_mode,
        execution_mode="auto",
        origin_device_id="phone-01",
        persona_id="balanced",
        use_profile_steering=True,
        timeout_ms=2_000,
    )


def test_personacare_grpc_request_reroutes_after_low_ram_heartbeat():
    async def scenario() -> None:
        phone, laptop = _demo_devices()
        phone_telemetry = MutableTelemetry(phone.health)
        laptop_telemetry = MutableTelemetry(laptop.health)
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        config = AgentClientConfig(
            brain_target=f"127.0.0.1:{port}",
            heartbeat_interval_seconds=60,
            reconnect_initial_seconds=0.01,
            reconnect_max_seconds=0.05,
        )
        # Keep the real routing identities while dispatching a harmless mock
        # result. No hardware runtime is invoked by this regression.
        artifacts = ArtifactRegistry({}, Path.cwd())
        phone_agent = DeviceAgent(
            phone, config, artifacts=artifacts, telemetry=phone_telemetry
        )
        laptop_agent = DeviceAgent(
            laptop, config, artifacts=artifacts, telemetry=laptop_telemetry
        )
        agents = [phone_agent, laptop_agent]
        agent_tasks = [asyncio.create_task(agent.run_forever()) for agent in agents]
        await asyncio.wait_for(
            asyncio.gather(*(agent.registered.wait() for agent in agents)),
            timeout=3,
        )

        try:
            async with grpc.aio.insecure_channel(config.brain_target) as channel:
                stub = pb_grpc.BrainControlStub(channel)

                normal = await stub.SubmitTask(_request())
                assert normal.success
                assert normal.device_id == "phone-01"
                assert normal.model_id == "android-mock-v1"
                assert not normal.steering.enabled
                assert (
                    service.execution_plans[normal.task_id].execution_mode
                    == ExecutionMode.SINGLE
                )
                assert any(
                    "compatible local capacity is available" in reason
                    for reason in normal.route_reasons
                )

                phone_telemetry.available_memory_mb = 64
                phone_agent.notify_network_changed()
                for _ in range(100):
                    if (
                        service.registry.get("phone-01").device.health.available_memory_mb
                        == 64
                    ):
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("Brain did not observe the 64 MB heartbeat")

                rerouted = await stub.SubmitTask(_request())
                assert rerouted.success
                assert rerouted.device_id == "pc-01"
                assert rerouted.model_id == "qwen3-4b-genie"
                assert not rerouted.steering.enabled
                assert (
                    service.execution_plans[rerouted.task_id].execution_mode
                    == ExecutionMode.SINGLE
                )
                assert any(
                    "Origin phone-01 has no eligible compatible local capacity"
                    in reason
                    for reason in rerouted.route_reasons
                )

                private = await stub.SubmitTask(_request(preferred_mode="private"))
                assert not private.success
                assert private.error_code == "NO_ELIGIBLE_FALLBACK"
                assert private.device_id == ""
                assert private.model_id == ""
                assert private.origin_device_id == "phone-01"
        finally:
            await asyncio.gather(*(agent.stop() for agent in agents))
            await asyncio.gather(*agent_tasks, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())

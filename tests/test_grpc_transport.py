from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import grpc

from dragon_nest.config import load_devices
from dragon_nest.models import HealthStatus
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.proto import dragonnest_pb2_grpc as pb_grpc
from dragon_nest.registry import DeviceRegistry, RegistryConfig
from dragon_nest.steering import SteeringRegistry
from dragon_nest.tasks import AttemptState
from dragon_nest.transport.agent import (
    AgentClientConfig,
    DeviceAgent,
    RegistrationRejectedError,
)
from dragon_nest.transport.brain import (
    BrainService,
    BrainServiceConfig,
    create_server,
    stop_server,
)


ROOT = Path(__file__).resolve().parents[1]


def _agents(target: str) -> list[DeviceAgent]:
    config = AgentClientConfig(
        brain_target=target,
        heartbeat_interval_seconds=0.05,
        reconnect_initial_seconds=0.01,
        reconnect_max_seconds=0.05,
    )
    return [
        DeviceAgent(device, config)
        for device in load_devices(ROOT / "configs/dev-fabric.yaml")
    ]


async def _start_agents(agents: list[DeviceAgent]) -> list[asyncio.Task[None]]:
    tasks = [asyncio.create_task(agent.run_forever()) for agent in agents]
    await asyncio.wait_for(
        asyncio.gather(*(agent.registered.wait() for agent in agents)), timeout=3
    )
    return tasks


async def _stop_agents(
    agents: list[DeviceAgent], tasks: list[asyncio.Task[None]]
) -> None:
    await asyncio.gather(*(agent.stop() for agent in agents))
    await asyncio.gather(*tasks, return_exceptions=True)


def test_grpc_agents_register_and_execute_routed_task():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = _agents(target)
        agent_tasks = await _start_agents(agents)
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                stub = pb_grpc.BrainControlStub(channel)
                response = await stub.SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Compare both options and recommend one.",
                        execution_mode="single",
                        timeout_ms=2_000,
                    )
                )
                fetched = await stub.GetTask(
                    pb.GetTaskRequest(task_id=response.task_id)
                )

            assert response.success
            assert response.state == "SUCCEEDED"
            assert response.device_id == "pc-01"
            assert response.model_id == "large-reasoning-v1"
            assert "Mock single result" in response.output_text
            assert fetched.output_text == response.output_text
            assert fetched.accepted_attempt_id == response.accepted_attempt_id
        finally:
            await _stop_agents(agents, agent_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_qr_registration_persists_client_profile_once():
    steering_registry = SteeringRegistry.from_yaml(
        ROOT / "configs/steering-vectors.yaml"
    )
    service = BrainService(steering_registry=steering_registry)
    session = service.enrollment.create(
        brain_host="192.168.1.20",
        brain_port=50051,
        use_tls=False,
    )
    registration = pb.RegisterDevice(
        device_id="android-profile-test",
        display_name="Test Phone",
        enrollment_token=session.bootstrap_credential,
        models=[pb.ModelCapability(model_id="android-mock")],
        personal_profile=pb.PersonalProfileRegistration(
            person_name="Alex",
            preferred_mode="private",
            steering_vector_id="concise-vs-verbose-layer-7",
            steering_alpha=-2,
            steering_positions="last",
        ),
    )

    error, credential = service._authorize_registration(registration)
    assert error == ""
    assert credential.startswith("dn_device_")
    profile = service.profiles.profile_for_device("android-profile-test")
    assert profile is not None
    assert profile.person_name == "Alex"
    assert profile.preferred_mode == "private"
    assert profile.steering_vector_id == "concise-vs-verbose-layer-7"
    first_profile_id = profile.profile_id

    # A duplicate bootstrap submission resolves to the existing claim rather
    # than creating a second profile during a reconnect race.
    error, duplicate_credential = service._authorize_registration(registration)
    assert error == ""
    assert duplicate_credential == credential
    assert service.profiles.profile_for_device("android-profile-test").profile_id == (
        first_profile_id
    )
    assert len(service.profiles.all()) == 1


def test_reconnect_with_device_credential_updates_client_profile():
    steering_registry = SteeringRegistry.from_yaml(
        ROOT / "configs/steering-vectors.yaml"
    )
    service = BrainService(steering_registry=steering_registry)
    session = service.enrollment.create(
        brain_host="192.168.1.20",
        brain_port=50051,
        use_tls=False,
    )

    def registration(enrollment_token: str, steering_alpha: float) -> pb.RegisterDevice:
        return pb.RegisterDevice(
            device_id="android-reconnect-test",
            display_name="Test Phone",
            enrollment_token=enrollment_token,
            models=[pb.ModelCapability(model_id="android-mock")],
            personal_profile=pb.PersonalProfileRegistration(
                person_name="Alex",
                preferred_mode="private",
                steering_vector_id="concise-vs-verbose-layer-7",
                steering_alpha=steering_alpha,
                steering_positions="last",
            ),
        )

    error, device_credential = service._authorize_registration(
        registration(session.bootstrap_credential, -2)
    )
    assert error == ""
    profile = service.profiles.profile_for_device("android-reconnect-test")
    assert profile.steering_alpha == -2

    # Reconnecting with the already-issued device credential (e.g. after the
    # user changes their answer style in settings) must update the existing
    # profile rather than silently dropping the new preference.
    error, credential = service._authorize_registration(
        registration(device_credential, 2)
    )
    assert error == ""
    assert credential == ""
    profile = service.profiles.profile_for_device("android-reconnect-test")
    assert profile.steering_alpha == 2
    assert len(service.profiles.all()) == 1


def test_grpc_stream_disconnect_retries_on_fallback_agent():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = _agents(target)
        agent_tasks = await _start_agents(agents)
        pc = next(agent for agent in agents if agent.device.device_id == "pc-01")
        pc.simulate_disconnect_on_next_task()
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Compare both options and recommend one.",
                        execution_mode="single",
                        timeout_ms=2_000,
                    )
                )

            task = service.tasks.get(response.task_id)
            assert response.success
            assert response.device_id == "phone-01"
            assert len(task.attempts) == 2
            assert task.attempts[0].device_id == "pc-01"
            assert task.attempts[0].state == AttemptState.DEVICE_OFFLINE
            assert task.attempts[1].device_id == "phone-01"
            assert task.attempts[1].state == AttemptState.SUCCEEDED
            assert task.attempts[0].attempt_id != task.attempts[1].attempt_id
        finally:
            await _stop_agents(agents, agent_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_agent_reconnects_after_unexpected_stream_offline():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        device = load_devices(ROOT / "configs/dev-fabric.yaml")[1]
        agent = DeviceAgent(
            device,
            AgentClientConfig(
                brain_target=target,
                heartbeat_interval_seconds=0.05,
                reconnect_initial_seconds=0.01,
                reconnect_max_seconds=0.02,
            ),
        )
        agent_task = asyncio.create_task(agent.run_forever())
        try:
            await asyncio.wait_for(agent.registered.wait(), timeout=3)
            original_session = await service.sessions.get(device.device_id)
            agent.simulate_disconnect_on_next_task()
            async with grpc.aio.insecure_channel(target) as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Compare both options and recommend one.",
                        execution_mode="single",
                        timeout_ms=1_000,
                    )
                )

            for _ in range(100):
                replacement = await service.sessions.get(device.device_id)
                record = service.registry.get(device.device_id)
                if (
                    replacement is not None
                    and replacement is not original_session
                    and not replacement.closed
                    and record.stream_connected
                    and record.status == HealthStatus.HEALTHY
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Agent did not reconnect after stream loss")

            assert not response.success
            assert response.error_code == "NO_ELIGIBLE_FALLBACK"
        finally:
            await agent.stop()
            await asyncio.gather(agent_task, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_live_brain_sweeper_expires_missed_agent_heartbeat():
    async def scenario() -> None:
        registry = DeviceRegistry(
            RegistryConfig(stale_after_seconds=0.05, offline_after_seconds=0.1)
        )
        service = BrainService(
            BrainServiceConfig(sweep_interval_seconds=0.01), registry=registry
        )
        server, port = await create_server(service, "127.0.0.1:0")
        device = load_devices(ROOT / "configs/dev-fabric.yaml")[0]
        agent = DeviceAgent(
            device,
            AgentClientConfig(
                brain_target=f"127.0.0.1:{port}",
                heartbeat_interval_seconds=60,
                reconnect_initial_seconds=1,
                reconnect_max_seconds=1,
            ),
        )
        agent_task = asyncio.create_task(agent.run_forever())
        try:
            await asyncio.wait_for(agent.registered.wait(), timeout=3)
            # Windows' asyncio/grpc timer resolution can make 10 ms sleeps
            # return early, so an iteration budget may elapse before the
            # registry's 100 ms monotonic deadline. Poll against a real
            # monotonic deadline without changing heartbeat/sweeper semantics.
            deadline = asyncio.get_running_loop().time() + 3
            while registry.get(device.device_id).status != HealthStatus.OFFLINE:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("missed heartbeat did not expire Agent")
                await asyncio.sleep(0.02)

            assert not registry.get(device.device_id).stream_connected
        finally:
            await agent.stop()
            await asyncio.gather(agent_task, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_graceful_agent_shutdown_excludes_device_from_new_routes():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = _agents(target)
        agent_tasks = await _start_agents(agents)
        pc_index = next(
            index
            for index, agent in enumerate(agents)
            if agent.device.device_id == "pc-01"
        )
        try:
            await agents[pc_index].stop(graceful=True)
            await asyncio.wait_for(agent_tasks[pc_index], timeout=2)
            async with grpc.aio.insecure_channel(target) as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Compare both options and recommend one.",
                        execution_mode="single",
                        timeout_ms=2_000,
                    )
                )
            assert response.success
            assert response.device_id == "phone-01"
        finally:
            remaining_agents = [
                agent for index, agent in enumerate(agents) if index != pc_index
            ]
            remaining_tasks = [
                task for index, task in enumerate(agent_tasks) if index != pc_index
            ]
            await _stop_agents(remaining_agents, remaining_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_private_mode_routes_only_to_declared_origin_device():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = _agents(target)
        agent_tasks = await _start_agents(agents)
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                stub = pb_grpc.BrainControlStub(channel)
                missing_origin = await stub.SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Compare both options and recommend one.",
                        preferred_mode="private",
                        execution_mode="single",
                    )
                )
                response = await stub.SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Compare both options and recommend one.",
                        preferred_mode="private",
                        execution_mode="single",
                        origin_device_id="phone-01",
                        timeout_ms=2_000,
                    )
                )

            assert missing_origin.error_code == "ORIGIN_DEVICE_REQUIRED"
            assert response.success
            assert response.device_id == "phone-01"
            assert response.origin_device_id == "phone-01"
            assert any(
                "remote devices were excluded" in reason
                for reason in response.route_reasons
            )
            task = service.tasks.get(response.task_id)
            assert {attempt.device_id for attempt in task.attempts} == {"phone-01"}
        finally:
            await _stop_agents(agents, agent_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_invalid_enrollment_token_is_rejected():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        device = load_devices(ROOT / "configs/dev-fabric.yaml")[0]
        agent = DeviceAgent(
            device,
            AgentClientConfig(
                brain_target=target,
                enrollment_token="wrong-token",
                heartbeat_interval_seconds=0.05,
            ),
        )
        try:
            try:
                await asyncio.wait_for(agent.run_forever(), timeout=2)
            except RegistrationRejectedError as exc:
                assert "invalid enrollment token" in str(exc)
            else:
                raise AssertionError("registration should have been rejected")
        finally:
            await agent.stop(graceful=False)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_grpc_data_parallel_fans_out_and_reduces_shards():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = _agents(target)
        agent_tasks = await _start_agents(agents)
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Summarize sections, then give key points.",
                        execution_mode="data_parallel",
                        timeout_ms=2_000,
                    )
                )

            assert response.success
            assert response.state == "SUCCEEDED"
            assert response.device_id == "brain"
            assert response.model_id == "parallel-reducer"
            assert "[shard-1]" in response.output_text
            assert "[shard-2]" in response.output_text
            assert "[shard-3]" in response.output_text
            child_devices = {
                service.tasks.get(f"{response.task_id}:shard-{index}")
                .attempts[-1]
                .device_id
                for index in range(1, 4)
            }
            assert child_devices == {"pc-01", "phone-01"}
        finally:
            await _stop_agents(agents, agent_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_grpc_replica_race_accepts_first_success_and_cancels_loser():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        devices = load_devices(ROOT / "configs/dev-fabric.yaml")
        agents = [
            DeviceAgent(
                device,
                AgentClientConfig(
                    brain_target=target,
                    heartbeat_interval_seconds=0.05,
                    reconnect_initial_seconds=0.01,
                    reconnect_max_seconds=0.05,
                    execution_delay_seconds=(
                        0.25 if device.device_id == "phone-01" else 0
                    ),
                ),
            )
            for device in devices
        ]
        agent_tasks = await _start_agents(agents)
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Compare both options and recommend one.",
                        execution_mode="data_parallel",
                        reducer="first_success",
                        timeout_ms=2_000,
                    )
                )

            child = service.tasks.get(f"{response.task_id}:shard-1")
            slow = next(
                agent for agent in agents if agent.device.device_id == "phone-01"
            )
            slow_attempt = next(
                attempt
                for attempt in child.attempts
                if attempt.device_id == "phone-01"
            )
            for _ in range(20):
                if slow_attempt.attempt_id in slow.cancelled_attempt_ids:
                    break
                await asyncio.sleep(0.01)

            assert response.success
            assert response.device_id == "pc-01"
            assert response.reducer == "first_success"
            assert len(child.attempts) == 2
            assert child.accepted_attempt_id
            assert {
                attempt.state for attempt in child.attempts
            } == {AttemptState.SUCCEEDED, AttemptState.CANCELLED}
            assert slow_attempt.attempt_id in slow.cancelled_attempt_ids
        finally:
            await _stop_agents(agents, agent_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_grpc_data_parallel_retries_shards_after_stream_loss():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = _agents(target)
        agent_tasks = await _start_agents(agents)
        pc = next(agent for agent in agents if agent.device.device_id == "pc-01")
        pc.simulate_disconnect_on_next_task()
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Summarize sections, then give key points.",
                        execution_mode="data_parallel",
                        timeout_ms=2_000,
                    )
                )

            assert response.success
            children = [
                service.tasks.get(f"{response.task_id}:shard-{index}")
                for index in range(1, 4)
            ]
            retried = [task for task in children if len(task.attempts) == 2]
            assert retried
            for task in retried:
                assert task.attempts[0].state == AttemptState.DEVICE_OFFLINE
                assert task.attempts[1].state == AttemptState.SUCCEEDED
                assert task.attempts[0].attempt_id != task.attempts[1].attempt_id
            assert all(task.attempts[-1].device_id == "phone-01" for task in retried)
        finally:
            await _stop_agents(agents, agent_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_grpc_layer_pipeline_transfers_boundary_between_agents():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = _agents(target)
        agent_tasks = await _start_agents(agents)
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Analyze this complex trade-off.",
                        preferred_mode="auto",
                        execution_mode="layer_pipeline",
                        timeout_ms=2_000,
                    )
                )

            assert response.success
            assert response.state == "SUCCEEDED"
            assert response.device_id == "pc-01"
            assert response.model_id == "qwen3-0.6b-split-14"
            assert "Mock remote layer-pipeline result" in response.output_text
            assert "boundary=sha256:" in response.output_text
            stage_1 = service.tasks.get(f"{response.task_id}:stage-1")
            stage_2 = service.tasks.get(f"{response.task_id}:stage-2")
            assert stage_1.attempts[-1].device_id == "phone-01"
            assert stage_2.attempts[-1].device_id == "pc-01"
        finally:
            await _stop_agents(agents, agent_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_grpc_layer_pipeline_retries_compatible_stage_after_disconnect():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        devices = load_devices(ROOT / "configs/dev-fabric.yaml")
        pc = next(device for device in devices if device.device_id == "pc-01")
        backup = replace(pc, device_id="pc-02", display_name="Backup PC")
        config = AgentClientConfig(
            brain_target=target,
            heartbeat_interval_seconds=0.05,
            reconnect_initial_seconds=0.01,
            reconnect_max_seconds=0.05,
        )
        agents = [DeviceAgent(device, config) for device in [*devices, backup]]
        agent_tasks = await _start_agents(agents)
        primary = next(agent for agent in agents if agent.device.device_id == "pc-01")
        primary.simulate_disconnect_on_next_task()
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Analyze this complex trade-off.",
                        preferred_mode="auto",
                        execution_mode="layer_pipeline",
                        timeout_ms=2_000,
                    )
                )

            assert response.success
            assert response.device_id == "pc-02"
            stage_2 = service.tasks.get(f"{response.task_id}:stage-2")
            assert len(stage_2.attempts) == 2
            assert stage_2.attempts[0].device_id == "pc-01"
            assert stage_2.attempts[0].state == AttemptState.DEVICE_OFFLINE
            assert stage_2.attempts[1].device_id == "pc-02"
            assert stage_2.attempts[1].state == AttemptState.SUCCEEDED
        finally:
            await _stop_agents(agents, agent_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_brain_rejects_corrupted_pipeline_boundary():
    service = BrainService()
    boundary = pb.BoundaryTensor(
        tensor_name="hidden",
        dtype="float32",
        shape=[1],
        data=b"corrupt",
        checksum=f"sha256:{'0' * 64}",
    )

    assert service._boundary_error(boundary) == "boundary checksum mismatch"


def test_grpc_steering_reaches_agent_and_response():
    async def scenario() -> None:
        steering_registry = SteeringRegistry.from_yaml(
            ROOT / "configs/steering-vectors.yaml"
        )
        service = BrainService(steering_registry=steering_registry)
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = _agents(target)
        agent_tasks = await _start_agents(agents)
        try:
            spec = steering_registry.default_spec("concise-vs-verbose-layer-7")
            async with grpc.aio.insecure_channel(target) as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Answer concisely: explain local AI.",
                        execution_mode="single",
                        steering=pb.SteeringSpec(
                            enabled=True,
                            vector_id=spec.vector_id,
                            model_family=spec.model_family,
                            target_layer=spec.target_layer,
                            alpha=spec.alpha,
                            positions=spec.positions,
                        ),
                    )
                )

            assert response.success
            assert response.steering.enabled
            assert response.steering.vector_id == spec.vector_id
            assert f"alpha={spec.alpha}" in response.output_text
            assert any("steering vector" in reason for reason in response.route_reasons)
        finally:
            await _stop_agents(agents, agent_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_grpc_pipeline_steering_is_assigned_to_owning_stage():
    async def scenario() -> None:
        steering_registry = SteeringRegistry.from_yaml(
            ROOT / "configs/steering-vectors.yaml"
        )
        service = BrainService(steering_registry=steering_registry)
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = _agents(target)
        agent_tasks = await _start_agents(agents)
        try:
            spec = steering_registry.default_spec("concise-vs-verbose-layer-7")
            async with grpc.aio.insecure_channel(target) as channel:
                response = await pb_grpc.BrainControlStub(channel).SubmitTask(
                    pb.SubmitTaskRequest(
                        request_text="Analyze this concisely.",
                        preferred_mode="auto",
                        execution_mode="layer_pipeline",
                        steering=pb.SteeringSpec(
                            enabled=True,
                            vector_id=spec.vector_id,
                            model_family=spec.model_family,
                            target_layer=7,
                            alpha=spec.alpha,
                            positions=spec.positions,
                        ),
                    )
                )

            assert response.success
            assert "Steering:" in response.output_text
            assert any("layer 7 on phone-01" in item for item in response.route_reasons)
        finally:
            await _stop_agents(agents, agent_tasks)
            await stop_server(server, service)

    asyncio.run(scenario())

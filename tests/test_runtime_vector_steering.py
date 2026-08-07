"""End-to-end routing tests for the S25 runtime activation-steering path.

These exercise the same gRPC surface PersonaCare uses, so they assert what the
device is actually told to do -- not just what the config file says. The
invariant under test throughout is that Balanced keeps resolving the stock Base
artifact on the stock runtime, whatever happens to steering.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import grpc

from dragon_nest.artifacts import ArtifactRegistry
from dragon_nest.behavior import BehaviorProfileRegistry
from dragon_nest.deployments import ArtifactCatalog
from dragon_nest.models import Device, HealthState, ModelCapability
from dragon_nest.steering import SteeringRegistry
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.proto import dragonnest_pb2_grpc as pb_grpc
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

VECTOR_ID = "concise-vs-verbose-layer-7"
STEERABLE = "qwen3-0.6b-s25-runtime-steerable"
BASE = "qwen3-0.6b-s25-base"


def _baked(model_id: str, profile_id: str = "") -> ModelCapability:
    """A stock-GenieX artifact: either plain Base or a baked profile."""
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


def _steerable(
    *,
    vector_ids: tuple[str, ...] = (VECTOR_ID,),
    layers: tuple[int, ...] = (7,),
    supports_steering: bool = True,
) -> ModelCapability:
    """The forked-runtime bundle that binds alpha/steering_vector at runtime."""
    return ModelCapability(
        model_id=STEERABLE,
        model_family="qwen3",
        role="small_chat",
        task_classes=("chat_qa",),
        max_context_tokens=512,
        warm=False,
        quality_score=0.7,
        runtime_name="genie_aux",
        runtime_version="GenieX-fork-aux-0.3.5 / QAIRT-2.45",
        supported_accelerators=("htp",),
        min_memory_mb=128,
        supports_steering=supports_steering,
        artifact_id=f"artifact-{STEERABLE}",
        steering_modes=("runtime_vector",),
        steering_vector_ids=vector_ids,
        supported_steering_layers=layers,
    )


def _phone(models: tuple[ModelCapability, ...]) -> Device:
    return Device(
        device_id="phone-01",
        display_name="Galaxy S25 Ultra",
        device_type="phone",
        platform="android",
        total_memory_mb=12_288,
        health=HealthState(available_memory_mb=6_000),
        models=models,
    )


def _request(persona_id: str) -> pb.SubmitTaskRequest:
    return pb.SubmitTaskRequest(
        request_text="Explain why the sky is blue.",
        preferred_mode="local",
        execution_mode="auto",
        origin_device_id="phone-01",
        persona_id=persona_id,
        timeout_ms=2_000,
    )


def _run(models: tuple[ModelCapability, ...], body) -> None:
    """Register a phone advertising `models`, then run `body(stub, service)`."""

    async def scenario() -> None:
        # Constructed exactly as scripts/run_brain.py does: the realization
        # ladder only exists when Brain actually holds these registries.
        service = BrainService(
            steering_registry=SteeringRegistry.from_yaml(
                CONFIGS / "steering-vectors.yaml"
            ),
            artifact_catalog=ArtifactCatalog.from_yaml(
                CONFIGS / "artifact-catalog.yaml"
            ),
            behavior_registry=BehaviorProfileRegistry.from_yaml(
                CONFIGS / "behavior-profiles.yaml"
            ),
        )
        server, port = await create_server(service, "127.0.0.1:0")
        config = AgentClientConfig(
            brain_target=f"127.0.0.1:{port}",
            heartbeat_interval_seconds=60,
            reconnect_initial_seconds=0.01,
            reconnect_max_seconds=0.05,
        )
        agent = DeviceAgent(
            _phone(models), config, artifacts=ArtifactRegistry({}, Path.cwd())
        )
        agent_task = asyncio.create_task(agent.run_forever())
        try:
            await asyncio.wait_for(agent.registered.wait(), timeout=3)
            async with grpc.aio.insecure_channel(config.brain_target) as channel:
                await body(pb_grpc.BrainControlStub(channel), service)
        finally:
            await agent.stop()
            await asyncio.gather(agent_task, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_concise_and_detailed_prefer_runtime_vector_balanced_stays_on_base():
    """The headline behavior: one steerable artifact, two directions, and a
    Balanced request that never leaves the physically accepted stock path."""
    models = (
        _baked(BASE),
        _baked("qwen3-0.6b-s25-concise", "concise"),
        _baked("qwen3-0.6b-s25-detailed", "detailed"),
        _steerable(),
    )

    async def body(stub, service) -> None:
        balanced = await stub.SubmitTask(_request("balanced"))
        assert balanced.success
        # Balanced must not drift onto the steering bundle even though it is
        # advertised, warm-eligible, and the same model family.
        assert balanced.model_id == BASE
        balanced_plan = service.execution_plans[balanced.task_id]
        assert balanced_plan.profile_realization == "none"
        assert not balanced_plan.steering.enabled
        assert balanced_plan.steering.vector_id == ""

        for persona_id, alpha_sign in (("concise", -1), ("detailed", +1)):
            response = await stub.SubmitTask(_request(persona_id))
            assert response.success, persona_id
            assert response.model_id == STEERABLE
            plan = service.execution_plans[response.task_id]
            assert plan.behavior_profile_id == persona_id
            assert plan.profile_realization == "runtime_vector"
            steering = plan.steering
            assert steering.enabled
            assert steering.mode == "runtime_vector"
            assert steering.vector_id == VECTOR_ID
            assert steering.target_layer == 7
            assert steering.behavior_profile_id == persona_id
            # Direction, not magnitude: the calibrated value may be retuned,
            # but Concise must never be positive nor Detailed negative.
            assert steering.alpha * alpha_sign > 0

    _run(models, body)


def test_runtime_vector_falls_back_to_baked_when_steerable_artifact_absent():
    models = (
        _baked(BASE),
        _baked("qwen3-0.6b-s25-concise", "concise"),
        _baked("qwen3-0.6b-s25-detailed", "detailed"),
    )

    async def body(stub, service) -> None:
        for persona_id in ("concise", "detailed"):
            response = await stub.SubmitTask(_request(persona_id))
            assert response.success, persona_id
            assert response.model_id == f"qwen3-0.6b-s25-{persona_id}"
            plan = service.execution_plans[response.task_id]
            assert plan.profile_realization == "baked_profile"
            # A baked artifact is an intervention compiled into the weights;
            # it must never be described as a runtime vector.
            assert not plan.steering.enabled
            assert plan.steering.vector_id == ""

    _run(models, body)


def test_device_advertising_a_different_vector_does_not_get_runtime_steering():
    """An unknown vector id must fail closed, not bind whatever is on disk."""
    models = (
        _baked(BASE),
        _baked("qwen3-0.6b-s25-concise", "concise"),
        _steerable(vector_ids=("some-other-vector",)),
    )

    async def body(stub, service) -> None:
        response = await stub.SubmitTask(_request("concise"))
        assert response.success
        assert response.model_id == "qwen3-0.6b-s25-concise"
        plan = service.execution_plans[response.task_id]
        assert plan.profile_realization == "baked_profile"

    _run(models, body)


def test_device_advertising_a_different_layer_does_not_get_runtime_steering():
    models = (
        _baked(BASE),
        _baked("qwen3-0.6b-s25-concise", "concise"),
        _steerable(layers=(21,)),
    )

    async def body(stub, service) -> None:
        response = await stub.SubmitTask(_request("concise"))
        assert response.success
        assert response.model_id == "qwen3-0.6b-s25-concise"
        plan = service.execution_plans[response.task_id]
        assert plan.profile_realization == "baked_profile"

    _run(models, body)


def test_steering_disabled_capability_does_not_get_runtime_steering():
    models = (
        _baked(BASE),
        _baked("qwen3-0.6b-s25-concise", "concise"),
        _steerable(supports_steering=False),
    )

    async def body(stub, service) -> None:
        response = await stub.SubmitTask(_request("concise"))
        assert response.success
        assert response.model_id == "qwen3-0.6b-s25-concise"
        plan = service.execution_plans[response.task_id]
        assert plan.profile_realization == "baked_profile"

    _run(models, body)


def test_base_answers_when_neither_realization_exists_but_is_never_called_concise():
    """Neither the steerable bundle nor the baked bake is present.

    Concise declares ``allow_unsteered``, so the request is answered from Base
    rather than failed -- a styled request must not become unanswerable just
    because the one device carrying the behavior went ineligible.

    The invariant that survives is the honesty one: Base may serve the
    request, but it is never *called* Concise. The realization is reported as
    "none" and no steering is sent to the device.
    """
    models = (_baked(BASE),)

    async def body(stub, service) -> None:
        response = await stub.SubmitTask(_request("concise"))
        assert response.success
        assert response.model_id == BASE
        assert response.steering.mode != "runtime_vector" or not response.steering.enabled
        plan = service.execution_plans[response.task_id]
        assert plan.profile_realization == "none"
        assert plan.steering.enabled is False

    _run(models, body)


def test_balanced_is_refused_rather_than_served_by_the_steering_bundle():
    """The A/B safety boundary. If only the steerable bundle is present, a
    Balanced request must fail rather than silently run on the forked runtime
    and be reported as the unsteered base model."""
    models = (_steerable(),)

    async def body(stub, service) -> None:
        response = await stub.SubmitTask(_request("balanced"))
        assert not response.success
        assert response.model_id != STEERABLE

    _run(models, body)


def test_removing_the_steerable_artifact_restores_the_baked_realization():
    """The ladder is re-evaluated per request, not cached from registration."""
    models = (
        _baked(BASE),
        _baked("qwen3-0.6b-s25-concise", "concise"),
        _steerable(),
    )

    async def body(stub, service) -> None:
        first = await stub.SubmitTask(_request("concise"))
        assert first.model_id == STEERABLE

        current = service.registry.get("phone-01").device
        service.registry.register(
            replace(
                current,
                models=tuple(
                    model
                    for model in current.models
                    if model.model_id != STEERABLE
                ),
            )
        )
        second = await stub.SubmitTask(_request("concise"))
        assert second.success
        assert second.model_id == "qwen3-0.6b-s25-concise"
        assert (
            service.execution_plans[second.task_id].profile_realization
            == "baked_profile"
        )

    _run(models, body)

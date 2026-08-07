from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from dragon_nest.behavior import BehaviorProfileRegistry
from dragon_nest.config import load_devices
from dragon_nest.dashboard import create_dashboard_app
from dragon_nest.deployments import ArtifactCatalog
from dragon_nest.steering import SteeringRegistry
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server

ROOT = Path(__file__).resolve().parents[1]


def _service() -> BrainService:
    return BrainService(
        steering_registry=SteeringRegistry.from_yaml(
            ROOT / "configs/steering-vectors.yaml"
        ),
        artifact_catalog=ArtifactCatalog.from_yaml(
            ROOT / "configs/artifact-catalog.yaml"
        ),
        behavior_registry=BehaviorProfileRegistry.from_yaml(
            ROOT / "configs/behavior-profiles.yaml"
        ),
    )


def _client(service: BrainService) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_dashboard_app(service)),
        base_url="http://test",
    )


def test_admin_page_serves_behavior_routing_and_provisioning_panels():
    async def scenario() -> None:
        service = _service()
        async with _client(service) as client:
            page = await client.get("/admin")
        assert page.status_code == 200
        for marker in (
            "Behavior Routing",
            "Provisioning",
            "behavior-candidates",
            "provisioning-jobs",
            "sim-artifacts",
            "Runtime steering supported",
        ):
            assert marker in page.text

    asyncio.run(scenario())


def test_route_plan_preview_lists_candidates_and_rejections():
    async def scenario() -> None:
        service = _service()
        for device in load_devices(ROOT / "configs/dev-fabric.yaml"):
            service.registry.register(device)
        async with _client(service) as client:
            profiles = (await client.get("/api/behavior-profiles")).json()
            catalog = (await client.get("/api/artifact-catalog")).json()
            plan = (
                await client.post(
                    "/api/route-plan",
                    json={
                        "base_model_family": "mock",
                        "behavior_profile_id": "creative",
                        "fallback_policy_override": "exact_only",
                    },
                )
            ).json()
            concise = (
                await client.post(
                    "/api/route-plan",
                    json={
                        "base_model_family": "mock",
                        "behavior_profile_id": "concise",
                    },
                )
            ).json()

        assert {p["profile_id"] for p in profiles} >= {"concise", "medical-safe"}
        assert any(a["artifact_id"] == "small-chat-v1" for a in catalog)
        assert plan["behavior_profile"] == "creative"
        assert plan["chosen"] is not None
        assert plan["chosen"]["realization_mode"] == "runtime_vector"
        rejected = [c for c in plan["candidates"] if not c["feasible"]]
        assert rejected and all(c["rejection_reasons"] for c in rejected)
        assert plan["explanation"]
        assert concise["chosen"] is None
        assert concise["error_code"] == "BEHAVIOR_UNAVAILABLE"
        assert concise["provisioning_hint"] == "concise"

    asyncio.run(scenario())


def test_artifact_and_steering_simulations_change_the_route():
    async def scenario() -> None:
        service = _service()
        for device in load_devices(ROOT / "configs/dev-fabric.yaml"):
            service.registry.register(device)
        async with _client(service) as client:
            baseline = (
                await client.post(
                    "/api/route-plan",
                    json={"base_model_family": "mock"},
                )
            ).json()
            assert baseline["chosen"] is not None
            chosen_device = baseline["chosen"]["device_id"]
            chosen_artifact = baseline["chosen"]["artifact_id"]

            # simulate the chosen artifact disappearing from that device
            simulate = await client.post(
                f"/api/devices/{chosen_device}/simulate",
                json={"artifact_states": {chosen_artifact: "absent"}},
            )
            assert simulate.status_code == 200
            rerouted = (
                await client.post(
                    "/api/route-plan",
                    json={"base_model_family": "mock"},
                )
            ).json()
            assert (
                rerouted["chosen"] is None
                or rerouted["chosen"]["device_id"] != chosen_device
                or rerouted["chosen"]["artifact_id"] != chosen_artifact
            )

            # Concise is exact baked-only and must reject when its artifact is absent.
            for device_id in ("phone-01", "pc-01"):
                await client.post(
                    f"/api/devices/{device_id}/simulate",
                    json={"runtime_steering_enabled": False},
                )
            steered = (
                await client.post(
                    "/api/route-plan",
                    json={
                        "base_model_family": "mock",
                        "behavior_profile_id": "concise",
                    },
                )
            ).json()
            if steered["chosen"] is not None:
                assert steered["chosen"]["realization_mode"] != "runtime_vector"
            else:
                assert steered["error_code"] == "BEHAVIOR_UNAVAILABLE"

            devices = (await client.get("/api/devices")).json()
            phone = next(d for d in devices if d["device_id"] == "phone-01")
            assert phone["runtime_steering_enabled"] is False
            assert "prompt_profile" in phone["steering_realization_modes"]
            assert phone["deployments"]

    asyncio.run(scenario())


def test_missing_profile_provisions_and_becomes_routable():
    async def scenario() -> None:
        service = _service()
        for device in load_devices(ROOT / "configs/dev-fabric.yaml"):
            service.registry.register(device)
        async with _client(service) as client:
            plan = (
                await client.post(
                    "/api/route-plan",
                    json={
                        "base_model_family": "mock",
                        "behavior_profile_id": "family-assistant",
                    },
                )
            ).json()
            assert plan["chosen"] is None
            assert plan["error_code"] == "BEHAVIOR_UNAVAILABLE"
            assert plan["provisioning_hint"] == "family-assistant"

            job = (
                await client.post(
                    "/api/provisioning",
                    json={
                        "profile_id": "family-assistant",
                        "device_id": "pc-01",
                        "artifact_id": "family-assistant-v0-baked",
                    },
                )
            ).json()
            assert job["state"] == "missing"

            states = []
            for _ in range(7):
                job = (
                    await client.post(f"/api/provisioning/{job['job_id']}/advance")
                ).json()
                states.append(job["state"])
            assert states[-1] == "warm"
            # the mock adapter must never claim a real compilation
            assert all(
                entry["detail"].startswith("[mock]")
                for entry in job["history"][1:]
            )

            provisioned = (
                await client.post(
                    "/api/route-plan",
                    json={
                        "base_model_family": "mock",
                        "behavior_profile_id": "family-assistant",
                    },
                )
            ).json()
            assert provisioned["chosen"] is not None
            assert provisioned["chosen"]["device_id"] == "pc-01"
            assert (
                provisioned["chosen"]["artifact_id"] == "family-assistant-v0-baked"
            )
            assert provisioned["chosen"]["realization_mode"] == "baked_profile"

    asyncio.run(scenario())


def test_behavior_task_executes_end_to_end_on_mock_agents():
    async def scenario() -> None:
        service = _service()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        devices = load_devices(ROOT / "configs/dev-fabric.yaml")
        agents = [
            DeviceAgent(
                device,
                AgentClientConfig(
                    brain_target=target, heartbeat_interval_seconds=0.1
                ),
            )
            for device in devices
        ]
        agent_tasks = [asyncio.create_task(agent.run_forever()) for agent in agents]
        try:
            await asyncio.wait_for(
                asyncio.gather(*(agent.registered.wait() for agent in agents)),
                timeout=5,
            )
            async with _client(service) as client:
                response = (
                    await client.post(
                        "/api/behavior-tasks",
                        json={
                            "request_text": "Explain why local AI routing matters.",
                            "base_model_family": "mock",
                            "behavior_profile_id": "creative",
                            "fallback_policy_override": "exact_only",
                            "timeout_ms": 5000,
                        },
                    )
                ).json()
                assert response["success"], response
                assert response["route_plan"]["chosen"] is not None
                assert response["steering"]["vector_id"] == (
                    "concise-vs-verbose-layer-7"
                )

                task = (
                    await client.get(f"/api/tasks/{response['task_id']}")
                ).json()
                assert task["route_plan"] is not None
                assert task["route_reasons"]
                assert task["state"] == "SUCCEEDED"

                listed = (await client.get("/api/tasks")).json()
                assert any(
                    item["task_id"] == response["task_id"] for item in listed
                )
        finally:
            await asyncio.gather(*(agent.stop() for agent in agents))
            await asyncio.gather(*agent_tasks, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())

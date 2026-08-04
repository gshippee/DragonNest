from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from dragon_nest.config import load_devices
from dragon_nest.dashboard import create_dashboard_app
from dragon_nest.steering import SteeringRegistry
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_serves_six_panel_ui_and_registry_api():
    async def scenario() -> None:
        steering = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
        service = BrainService(steering_registry=steering)
        for device in load_devices(ROOT / "configs/dev-fabric.yaml"):
            service.registry.register(device)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            page = await client.get("/")
            devices = (await client.get("/api/devices")).json()
            vectors = (await client.get("/api/steering-vectors")).json()

        assert page.status_code == 200
        for panel in (
            "Device Registry",
            "Task Submission",
            "Routing Trace",
            "Parallel Progress",
            "Result",
            "Live Event Log",
        ):
            assert panel in page.text
        assert {device["device_id"] for device in devices} == {
            "phone-01",
            "pc-01",
        }
        assert vectors[0]["vector_id"] == "concise-vs-verbose-layer-7"

    asyncio.run(scenario())


def test_dashboard_simulation_updates_registry_health():
    async def scenario() -> None:
        service = BrainService()
        service.registry.register(load_devices(ROOT / "configs/dev-fabric.yaml")[0])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/devices/phone-01/simulate",
                json={"thermal_level": 0.95, "network_rtt_ms": 180},
            )

        assert response.status_code == 200
        assert response.json()["status"] == "UNHEALTHY"
        assert response.json()["health"]["network_rtt_ms"] == 180

    asyncio.run(scenario())


def test_dashboard_simulation_persists_across_live_agent_heartbeats():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        device = load_devices(ROOT / "configs/dev-fabric.yaml")[0]
        agent = DeviceAgent(
            device,
            AgentClientConfig(
                brain_target=f"127.0.0.1:{port}",
                heartbeat_interval_seconds=0.02,
                reconnect_initial_seconds=0.01,
            ),
        )
        agent_task = asyncio.create_task(agent.run_forever())
        try:
            await asyncio.wait_for(agent.registered.wait(), timeout=3)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_dashboard_app(service)),
                base_url="http://test",
            ) as client:
                await client.post(
                    f"/api/devices/{device.device_id}/simulate",
                    json={"thermal_level": 0.95},
                )
                await asyncio.sleep(0.08)
                hot = (await client.get("/api/devices")).json()[0]

                await client.post(
                    f"/api/devices/{device.device_id}/simulate",
                    json={"offline": True},
                )
                await asyncio.sleep(0.05)
                offline = (await client.get("/api/devices")).json()[0]

                await client.post(
                    f"/api/devices/{device.device_id}/simulate",
                    json={"offline": False, "thermal_level": 0.2},
                )
                await asyncio.sleep(0.05)
                restored = (await client.get("/api/devices")).json()[0]

            assert hot["status"] == "UNHEALTHY"
            assert hot["health"]["thermal_level"] == 0.95
            assert hot["simulated_constraint"]
            assert offline["status"] == "OFFLINE"
            assert restored["status"] == "HEALTHY"
            assert restored["connected"]
        finally:
            await agent.stop()
            await asyncio.gather(agent_task, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_dashboard_http_submission_exposes_parallel_progress_and_events():
    async def scenario() -> None:
        steering = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
        service = BrainService(steering_registry=steering)
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        config = AgentClientConfig(
            brain_target=target,
            heartbeat_interval_seconds=0.05,
            reconnect_initial_seconds=0.01,
        )
        agents = [
            DeviceAgent(device, config)
            for device in load_devices(ROOT / "configs/dev-fabric.yaml")
        ]
        agent_tasks = [asyncio.create_task(agent.run_forever()) for agent in agents]
        await asyncio.wait_for(
            asyncio.gather(*(agent.registered.wait() for agent in agents)), timeout=3
        )
        app = create_dashboard_app(service)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                submitted = await client.post(
                    "/api/tasks",
                    json={
                        "request_text": "Summarize sections, then give key points.",
                        "execution_mode": "data_parallel",
                    },
                )
                task_id = submitted.json()["task_id"]
                task = (await client.get(f"/api/tasks/{task_id}")).json()
                events = (await client.get("/api/events")).json()

            assert submitted.status_code == 200
            assert submitted.json()["success"]
            assert task["execution_mode"] == "data_parallel"
            assert len(task["progress"]) == 3
            assert {item["device_id"] for item in task["progress"]} == {
                "phone-01",
                "pc-01",
            }
            assert any(event["type"] == "DISPATCHED" for event in events)
            assert any(event["type"] == "SUCCEEDED" for event in events)
        finally:
            await asyncio.gather(*(agent.stop() for agent in agents))
            await asyncio.gather(*agent_tasks, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_dashboard_private_submission_preserves_origin_and_reducer():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = [
            DeviceAgent(
                device,
                AgentClientConfig(
                    brain_target=target,
                    heartbeat_interval_seconds=0.05,
                    reconnect_initial_seconds=0.01,
                ),
            )
            for device in load_devices(ROOT / "configs/dev-fabric.yaml")
        ]
        agent_tasks = [asyncio.create_task(agent.run_forever()) for agent in agents]
        await asyncio.wait_for(
            asyncio.gather(*(agent.registered.wait() for agent in agents)), timeout=3
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_dashboard_app(service)),
                base_url="http://test",
            ) as client:
                submitted = await client.post(
                    "/api/tasks",
                    json={
                        "request_text": "Rewrite this private note.",
                        "preferred_mode": "private",
                        "execution_mode": "single",
                        "origin_device_id": "phone-01",
                        "reducer": "concat",
                    },
                )
                task = (
                    await client.get(f"/api/tasks/{submitted.json()['task_id']}")
                ).json()

            assert submitted.json()["success"]
            assert submitted.json()["origin_device_id"] == "phone-01"
            assert submitted.json()["reducer"] == "concat"
            assert task["origin_device_id"] == "phone-01"
            assert task["reducer"] == "concat"
            assert task["result"]["device_id"] == "phone-01"
        finally:
            await asyncio.gather(*(agent.stop() for agent in agents))
            await asyncio.gather(*agent_tasks, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())


def test_simulated_resource_constraints_reroute_and_explain_exclusion():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        target = f"127.0.0.1:{port}"
        agents = [
            DeviceAgent(
                device,
                AgentClientConfig(
                    brain_target=target,
                    heartbeat_interval_seconds=0.02,
                    reconnect_initial_seconds=0.01,
                ),
            )
            for device in load_devices(ROOT / "configs/dev-fabric.yaml")
        ]
        agent_tasks = [asyncio.create_task(agent.run_forever()) for agent in agents]
        await asyncio.wait_for(
            asyncio.gather(*(agent.registered.wait() for agent in agents)), timeout=3
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_dashboard_app(service)),
                base_url="http://test",
            ) as client:
                await client.post(
                    "/api/devices/pc-01/simulate",
                    json={
                        "thermal_level": 0.95,
                        "accelerator_utilization": 0.95,
                        "network_rtt_ms": 200,
                    },
                )
                submitted = await client.post(
                    "/api/tasks",
                    json={
                        "request_text": "Compare both options and recommend one.",
                        "execution_mode": "single",
                    },
                )

            payload = submitted.json()
            assert payload["success"]
            assert payload["device_id"] == "phone-01"
            assert any(
                "Excluded pc-01: health status UNHEALTHY" in reason
                for reason in payload["route_reasons"]
            )
        finally:
            await asyncio.gather(*(agent.stop() for agent in agents))
            await asyncio.gather(*agent_tasks, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())

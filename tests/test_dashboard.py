from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess

import httpx

from dragon_nest.config import load_devices
from dragon_nest.dashboard import create_dashboard_app
from dragon_nest.steering import SteeringRegistry
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server


ROOT = Path(__file__).resolve().parents[1]
SERVICE_WORKER = ROOT / "src" / "dragon_nest" / "web" / "sw.js"


def test_dashboard_serves_demo_control_room_and_registry_api():
    async def scenario() -> None:
        steering = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
        service = BrainService(steering_registry=steering)
        for device in load_devices(ROOT / "configs/dev-fabric.yaml"):
            service.registry.register(device)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            admin_page = await client.get("/admin")
            user_page = await client.get("/")
            manifest = await client.get("/manifest.webmanifest")
            service_worker = await client.get("/sw.js")
            devices = (await client.get("/api/devices")).json()
            vectors = (await client.get("/api/steering-vectors")).json()

        assert admin_page.status_code == 200
        for panel in (
            "DragonNest Fabric",
            "Device Registry",
            "Live Requests",
            "Selected Request",
            "Advanced",
            "Event log",
        ):
            assert panel in admin_page.text
        assert "index\">01" not in admin_page.text
        admin_script = (
            ROOT / "src" / "dragon_nest" / "web" / "admin" / "app.js"
        ).read_text(encoding="utf-8")
        assert "Compute preference" in admin_script
        assert "Classification" in admin_script
        assert "Model selected" in admin_script
        assert "Execution topology" in admin_script
        for preference in ("Auto", "Local", "Elastic", "Quality"):
            assert preference in admin_script
        assert user_page.status_code == 200
        assert "Personal AI" in user_page.text
        assert "Answer style" in user_page.text
        assert "Battery" not in user_page.text
        assert manifest.status_code == 200
        assert service_worker.status_code == 200
        assert {device["device_id"] for device in devices} == {
            "phone-01",
            "pc-01",
        }
        assert vectors[0]["vector_id"] == "concise-vs-verbose-layer-7"

    asyncio.run(scenario())


def test_dashboard_follows_latest_then_preserves_manual_pin():
    node = shutil.which("node")
    if node is None:
        raise AssertionError("Node.js is required for the dashboard state regression")
    helper = ROOT / "src" / "dragon_nest" / "web" / "admin" / "selection.js"
    request_a = {
        "task_id": "request-a",
        "created_at": 100,
        "origin_device_id": "phone-01",
        "state": "SUCCEEDED",
        "result": {"device_id": "phone-01"},
        "model_id": "android-mock-v1",
    }
    request_b = {
        "task_id": "request-b",
        "created_at": 200,
        "origin_device_id": "phone-01",
        "state": "SUCCEEDED",
        "result": {
            "device_id": "pc-01",
            "metrics": {
                "model_id": "qwen3-4b-genie",
                "runtime_name": "genie",
                "accelerator": "htp",
            },
        },
    }
    script = """
const helper = require(process.argv[1]);
const tasks = JSON.parse(process.argv[2]);
const followed = helper.reconcileSelection(tasks, '', true);
const pinned = helper.reconcileSelection(tasks, 'request-a', false);
const resumed = helper.reconcileSelection(tasks, pinned.selectedTaskId, true);
process.stdout.write(JSON.stringify({ followed, pinned, resumed }));
"""
    completed = subprocess.run(
        [node, "-e", script, str(helper), json.dumps([request_a, request_b])],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert [task["task_id"] for task in result["followed"]["tasks"]] == [
        "request-b",
        "request-a",
    ]
    assert result["followed"]["selectedTaskId"] == "request-b"
    assert result["pinned"]["selectedTaskId"] == "request-a"
    assert result["resumed"]["selectedTaskId"] == "request-b"


def test_service_worker_uses_network_first_shell_updates():
    source = SERVICE_WORKER.read_text()

    assert 'const CACHE_NAME = "dragonnest-shell-v2"' in source
    assert "event.respondWith(networkFirst(request));" in source
    assert "const response = await fetch(request);" in source
    assert "const cached = await caches.match(request);" in source


def test_dashboard_creates_qr_enrollment_without_exposing_secret():
    async def scenario() -> None:
        service = BrainService()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/api/enrollment-sessions",
                json={
                    "brain_host": "192.168.1.20",
                    "brain_port": 50051,
                    "use_tls": False,
                    "ttl_seconds": 60,
                    "person_name": "Alex",
                    "device_name": "Alex's Phone",
                },
            )
            body = created.json()
            qr = await client.get(body["qr_url"])
            status = await client.get(
                f"/api/enrollment-sessions/{body['session_id']}"
            )
            profiles = await client.get("/api/personal-profiles")
            cancelled = await client.delete(
                f"/api/enrollment-sessions/{body['session_id']}"
            )
            gone = await client.get(body["qr_url"])
            profiles_after_cancel = await client.get("/api/personal-profiles")

        assert created.status_code == 200
        assert body["status"] == "PENDING"
        assert "credential" not in body
        assert qr.status_code == 200
        assert qr.headers["content-type"].startswith("image/svg+xml")
        assert qr.headers["cache-control"] == "no-store"
        assert status.json()["status"] == "PENDING"
        assert profiles.json()[0]["person_name"] == "Alex"
        assert body["profile_id"] == profiles.json()[0]["profile_id"]
        assert cancelled.json()["status"] == "CANCELLED"
        assert gone.status_code == 410
        assert profiles_after_cancel.json() == []

    asyncio.run(scenario())


def test_dashboard_creates_profileless_qr_for_client_owned_enrollment():
    async def scenario() -> None:
        service = BrainService()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            created = await client.post(
                "/api/enrollment-sessions",
                json={
                    "brain_host": "192.168.1.20",
                    "brain_port": 50051,
                    "use_tls": False,
                    "ttl_seconds": 60,
                },
            )
            profiles = await client.get("/api/personal-profiles")

        assert created.status_code == 200
        assert created.json()["profile_id"] == ""
        assert profiles.json() == []

    asyncio.run(scenario())


def test_personal_profile_supplies_default_mode_and_steering():
    async def scenario() -> None:
        steering = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
        service = BrainService(steering_registry=steering)
        profile = service.profiles.create(
            person_name="Alex",
            preferred_mode="private",
            steering_vector_id="concise-vs-verbose-layer-7",
            steering_alpha=-1.5,
            steering_positions="last",
        )
        service.profiles.associate_device(
            "phone-01", profile.profile_id, "Alex's Phone"
        )
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
                submitted = await client.post(
                    "/api/tasks",
                    json={
                        "request_text": "Rewrite this note.",
                        "preferred_mode": "auto",
                        "execution_mode": "single",
                        "origin_device_id": "phone-01",
                        "use_profile_steering": True,
                    },
                )
                devices = (await client.get("/api/devices")).json()

            result = submitted.json()
            assert result["success"]
            assert result["device_id"] == "phone-01"
            assert result["steering"]["enabled"]
            assert result["steering"]["vector_id"] == (
                "concise-vs-verbose-layer-7"
            )
            assert result["steering"]["alpha"] == -1.5
            assert any(
                "Applied personal profile 'Alex'" in reason
                for reason in result["route_reasons"]
            )
            assert devices[0]["display_name"] == "Alex's Phone"
            assert devices[0]["personal_profile"]["person_name"] == "Alex"
        finally:
            await agent.stop()
            await asyncio.gather(agent_task, return_exceptions=True)
            await stop_server(server, service)

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
        assert set(response.json()["simulated_fields"]) == {
            "network_rtt_ms",
            "thermal_level",
        }

    asyncio.run(scenario())


def test_dashboard_can_remove_a_device_and_its_profile_association():
    async def scenario() -> None:
        service = BrainService()
        device = load_devices(ROOT / "configs/dev-fabric.yaml")[0]
        service.registry.register(device)
        profile = service.profiles.create(person_name="Alex")
        service.profiles.associate_device(
            device.device_id, profile.profile_id, device.display_name
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            removed = await client.delete(f"/api/devices/{device.device_id}")
            devices = (await client.get("/api/devices")).json()
            missing = await client.delete(f"/api/devices/{device.device_id}")

        assert removed.status_code == 200
        assert removed.json() == {
            "device_id": device.device_id,
            "status": "REMOVED",
        }
        assert devices == []
        assert service.profiles.association_for_device(device.device_id) is None
        assert device.device_id in service.removed_device_ids
        assert missing.status_code == 404

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
            assert task["preferred_mode"] == "private"
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

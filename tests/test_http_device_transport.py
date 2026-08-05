from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response as FastAPIResponse

from dragon_nest.dispatch import DeviceOfflineError
from dragon_nest.dashboard import create_dashboard_app
from dragon_nest.models import Device, HealthState, ModelCapability, SteeringSpec
from dragon_nest.transport.brain import BrainService
from dragon_nest.transport.http_device import (
    HttpDeviceConfig,
    HttpDeviceError,
    HttpDeviceSession,
)


def _endpoint_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"available_memory_mb": 4096, "reachable": True}

    @app.get("/info")
    async def info():
        return {
            "display_name": "Discovered Edge Box",
            "device_type": "endpoint",
            "platform": "linux",
            "total_memory_mb": 16384,
            "models": [
                {
                    "model_id": "discovered-model",
                    "model_family": "discovered",
                    "role": "general",
                    "task_classes": ["chat_qa"],
                    "max_context_tokens": 8192,
                    "warm": True,
                    "quality_score": 0.8,
                }
            ],
            "hardware": {"manufacturer": "Acme", "model": "EdgeBox 1"},
        }

    @app.post("/execute")
    async def execute(request: Request):
        body = await request.json()
        return {
            "success": True,
            "output_text": f"echo:{body['request_text']}",
            "metrics": {
                "model_id": body["model_id"],
                "model_version": "v1",
                "runtime_name": "http",
                "runtime_version": "1.0",
                "accelerator": "cpu",
                "execution_latency_ms": 12,
            },
        }

    @app.post("/cancel")
    async def cancel():
        return {"acknowledged": True}

    return app


def _broken_endpoint_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return FastAPIResponse(status_code=500)

    @app.post("/execute")
    async def execute():
        return FastAPIResponse(status_code=500)

    return app


def _device_payload(device_id: str) -> dict:
    return {
        "device_id": device_id,
        "base_url": "http://endpoint.local",
        "total_memory_mb": 8192,
        "models": [
            {
                "model_id": "remote-model",
                "model_family": "remote",
                "role": "general",
                "task_classes": ["chat_qa"],
                "max_context_tokens": 4096,
                "warm": True,
                "quality_score": 0.7,
            }
        ],
    }


def test_dashboard_registers_http_endpoint_device_and_dispatches_task():
    async def scenario() -> None:
        service = BrainService()
        # Route the brain's outbound HTTP client at the in-process fake
        # endpoint app instead of the real network.
        service._http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_endpoint_app()),
            base_url="http://endpoint.local",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            registered = await client.post(
                "/api/rest-devices", json=_device_payload("edge-01")
            )
            assert registered.status_code == 200
            body = registered.json()
            assert body["device_id"] == "edge-01"
            assert body["transport"] == "http_endpoint"
            assert body["base_url"] == "http://endpoint.local"

            devices = (await client.get("/api/devices")).json()
            assert any(d["device_id"] == "edge-01" for d in devices)

            submitted = await client.post(
                "/api/tasks",
                json={"request_text": "hello", "execution_mode": "single"},
            )
            response = submitted.json()
            assert response["success"], response
            assert response["device_id"] == "edge-01"
            assert response["output_text"] == "echo:hello"

            deregistered = await client.delete("/api/rest-devices/edge-01")
            assert deregistered.status_code == 200
            assert "edge-01" not in service.http_devices

            missing = await client.delete("/api/rest-devices/edge-01")
            assert missing.status_code == 404

        await service.stop()

    asyncio.run(scenario())


def test_register_http_device_validates_url_and_model_requirements():
    async def scenario() -> None:
        service = BrainService()
        try:
            try:
                await service.register_http_device(
                    Device(
                        device_id="bad-url",
                        display_name="bad-url",
                        device_type="endpoint",
                        platform="linux",
                        total_memory_mb=1024,
                        health=HealthState(),
                        models=(
                            ModelCapability(
                                model_id="m",
                                model_family="m",
                                role="general",
                                task_classes=(),
                                max_context_tokens=1024,
                                warm=True,
                                quality_score=0.5,
                            ),
                        ),
                    ),
                    HttpDeviceConfig(device_id="bad-url", base_url="ftp://nope"),
                )
                assert False, "expected HttpDeviceError"
            except HttpDeviceError as exc:
                assert "http(s)" in str(exc)

            try:
                await service.register_http_device(
                    Device(
                        device_id="no-models",
                        display_name="no-models",
                        device_type="endpoint",
                        platform="linux",
                        total_memory_mb=1024,
                        health=HealthState(),
                        models=(),
                    ),
                    HttpDeviceConfig(
                        device_id="no-models", base_url="http://endpoint.local"
                    ),
                )
                assert False, "expected HttpDeviceError"
            except HttpDeviceError as exc:
                assert "model capability" in str(exc)
        finally:
            await service.stop()

    asyncio.run(scenario())


def test_discover_route_returns_probed_capabilities():
    async def scenario() -> None:
        service = BrainService()
        service._http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_endpoint_app()),
            base_url="http://endpoint.local",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            probed = await client.post(
                "/api/rest-devices/discover",
                json={"base_url": "http://endpoint.local"},
            )
            assert probed.status_code == 200
            body = probed.json()
            assert body["display_name"] == "Discovered Edge Box"
            assert body["total_memory_mb"] == 16384
            assert [m["model_id"] for m in body["models"]] == ["discovered-model"]
            assert body["hardware"]["manufacturer"] == "Acme"

        await service.stop()

    asyncio.run(scenario())


def test_discover_route_reports_502_when_endpoint_has_no_info_route():
    async def scenario() -> None:
        service = BrainService()
        service._http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_broken_endpoint_app()),
            base_url="http://endpoint.local",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            probed = await client.post(
                "/api/rest-devices/discover",
                json={"base_url": "http://endpoint.local"},
            )
            assert probed.status_code == 502

        await service.stop()

    asyncio.run(scenario())


def test_register_with_empty_models_auto_discovers_from_endpoint():
    async def scenario() -> None:
        service = BrainService()
        service._http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_endpoint_app()),
            base_url="http://endpoint.local",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            registered = await client.post(
                "/api/rest-devices",
                json={"device_id": "auto-01", "base_url": "http://endpoint.local"},
            )
            assert registered.status_code == 200
            body = registered.json()
            assert body["display_name"] == "Discovered Edge Box"
            assert body["platform"] == "linux"
            assert body["health"]["available_memory_mb"] == 16384
            assert [m["model_id"] for m in body["models"]] == ["discovered-model"]
            assert body["hardware"]["manufacturer"] == "Acme"

            submitted = await client.post(
                "/api/tasks",
                json={"request_text": "hello", "execution_mode": "single"},
            )
            response = submitted.json()
            assert response["success"], response
            assert response["device_id"] == "auto-01"

        await service.stop()

    asyncio.run(scenario())


def test_register_with_empty_models_and_unreachable_endpoint_fails_cleanly():
    async def scenario() -> None:
        service = BrainService()
        service._http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_broken_endpoint_app()),
            base_url="http://endpoint.local",
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            registered = await client.post(
                "/api/rest-devices",
                json={"device_id": "broken-01", "base_url": "http://endpoint.local"},
            )
            assert registered.status_code == 502
            assert "broken-01" not in service.http_devices

        await service.stop()

    asyncio.run(scenario())


def test_http_device_session_raises_offline_on_transport_failure():
    async def scenario() -> None:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_broken_endpoint_app()),
            base_url="http://endpoint.local",
        )
        session = HttpDeviceSession(
            HttpDeviceConfig(device_id="flaky", base_url="http://endpoint.local"),
            client,
        )
        try:
            raised = False
            try:
                await session.fetch_health()
            except DeviceOfflineError:
                raised = True
            assert raised

            raised = False
            try:
                await session.execute(
                    "task-1", "attempt-1", "hi", "remote-model", 1000, SteeringSpec()
                )
            except DeviceOfflineError:
                raised = True
            assert raised
        finally:
            await client.aclose()

    asyncio.run(scenario())

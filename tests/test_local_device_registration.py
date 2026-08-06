from __future__ import annotations

import asyncio

import httpx

from dragon_nest import local_hardware
from dragon_nest.dashboard import create_dashboard_app
from dragon_nest.transport.brain import BrainService


_FAKE_PROBE = {
    "hardware": {
        "manufacturer": "Qualcomm",
        "model": "Snapdragon X Elite Dev Box",
        "device": "bench-01",
        "os_version": "Windows 11 Pro",
        "api_level": 0,
        "soc_manufacturer": "Qualcomm",
        "soc_model": "Snapdragon X Elite X1E80100",
        "cpu_abis": ["arm64-v8a"],
        "cpu_core_count": 12,
        "total_storage_mb": 512_000,
        "available_storage_mb": 128_000,
        "npu_status": "available",
        "npu_name": "Hexagon NPU (v73)",
        "qnn_runtime_version": "2.31.0",
    },
    "platform": "windows",
    "total_memory_mb": 32_768,
    "display_name": "bench-01",
}


def test_local_probe_route_returns_hardware_inventory_shape(monkeypatch):
    monkeypatch.setattr(local_hardware, "probe_local_hardware", lambda: _FAKE_PROBE)

    async def scenario() -> None:
        service = BrainService()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            probed = await client.get("/api/local-devices/probe")
            assert probed.status_code == 200
            assert probed.json() == _FAKE_PROBE
        await service.stop()

    asyncio.run(scenario())


def test_registering_a_local_device_fills_hardware_without_a_network_round_trip(monkeypatch):
    monkeypatch.setattr(local_hardware, "probe_local_hardware", lambda: _FAKE_PROBE)

    async def scenario() -> None:
        service = BrainService()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            registered = await client.post(
                "/api/rest-devices",
                json={
                    "device_id": "bench-01",
                    "base_url": "http://127.0.0.1:9999",
                    "probe_local": True,
                    "models": [
                        {
                            "model_id": "local-model",
                            "model_family": "local",
                            "role": "general",
                            "task_classes": ["chat_qa"],
                            "max_context_tokens": 4096,
                            "warm": True,
                            "quality_score": 0.7,
                        }
                    ],
                },
            )
            assert registered.status_code == 200
            body = registered.json()
            assert body["device_id"] == "bench-01"
            assert body["display_name"] == "bench-01"
            assert body["platform"] == "windows"
            assert body["hardware"] == _FAKE_PROBE["hardware"]
            assert body["health"]["available_memory_mb"] == 32_768
        await service.stop()

    asyncio.run(scenario())


def test_explicit_hardware_wins_over_probe_local():
    async def scenario() -> None:
        service = BrainService()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            registered = await client.post(
                "/api/rest-devices",
                json={
                    "device_id": "bench-02",
                    "base_url": "http://127.0.0.1:9999",
                    "probe_local": True,
                    "hardware": {"manufacturer": "Explicit Corp"},
                    "models": [
                        {
                            "model_id": "local-model",
                            "model_family": "local",
                            "role": "general",
                            "task_classes": ["chat_qa"],
                            "max_context_tokens": 4096,
                            "warm": True,
                            "quality_score": 0.7,
                        }
                    ],
                },
            )
            assert registered.status_code == 200
            body = registered.json()
            assert body["hardware"]["manufacturer"] == "Explicit Corp"
        await service.stop()

    asyncio.run(scenario())

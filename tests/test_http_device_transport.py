from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

from dragon_nest.dashboard import create_dashboard_app
from dragon_nest.endpoints import EndpointError, HttpEndpoint, HttpEndpointStore
from dragon_nest.models import Device, HealthState, ModelCapability
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.transport.brain import AgentSession, BrainService, BrainServiceConfig
from dragon_nest.transport.sessions import SessionConflictError


ADMIN_HEADERS = {"Authorization": "Bearer endpoint-admin"}


def _model() -> ModelCapability:
    return ModelCapability(
        model_id="remote-model",
        model_family="remote",
        role="general",
        task_classes=("chat_qa",),
        max_context_tokens=4096,
        warm=True,
        quality_score=0.7,
    )


def _endpoint(
    device_id: str = "edge-01",
    *,
    credential_env: str = "",
    allow_profile_context: bool = False,
) -> HttpEndpoint:
    return HttpEndpoint(
        device=Device(
            device_id=device_id,
            display_name="Edge Box",
            device_type="endpoint",
            platform="linux",
            total_memory_mb=8192,
            health=HealthState(available_memory_mb=8192),
            models=(_model(),),
        ),
        base_url="http://endpoint.local",
        credential_env=credential_env,
        poll_interval_seconds=300,
        allow_profile_context=allow_profile_context,
    )


def _service(*, enabled: bool = True, state_db_path: str = ":memory:") -> BrainService:
    return BrainService(
        BrainServiceConfig(
            state_db_path=state_db_path,
            http_endpoint_registration_enabled=enabled,
            http_endpoint_admin_token="endpoint-admin",
            http_endpoint_allowed_hosts=("endpoint.local",),
        )
    )


def _endpoint_app(requests: list[dict], expected_token: str = "") -> FastAPI:
    app = FastAPI()

    def authorized(request: Request) -> bool:
        return not expected_token or request.headers.get("authorization") == (
            f"Bearer {expected_token}"
        )

    @app.get("/health")
    async def health(request: Request):
        if not authorized(request):
            return Response(status_code=401)
        return {"available_memory_mb": 4096, "reachable": True}

    @app.get("/info")
    async def info(request: Request):
        if not authorized(request):
            return Response(status_code=401)
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

    async def result(request: Request):
        if not authorized(request):
            return Response(status_code=401)
        body = await request.json()
        requests.append(body)
        return {
            "success": True,
            "output_text": f"echo:{body['request_text']}",
            "metrics": {
                "model_id": body["model_id"],
                "runtime_name": "http",
                "execution_latency_ms": 12,
            },
        }

    app.post("/execute")(result)
    app.post("/execute_shard")(result)
    app.post("/execute_pipeline_stage")(result)

    @app.post("/cancel")
    async def cancel():
        return {"acknowledged": True}

    return app


def _install_endpoint_client(
    service: BrainService, requests: list[dict], expected_token: str = ""
) -> None:
    service._http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=_endpoint_app(requests, expected_token=expected_token)
        ),
        base_url="http://endpoint.local",
    )


def _registration_payload(device_id: str = "edge-01") -> dict:
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


def test_endpoint_admin_api_is_disabled_and_authenticated():
    async def scenario() -> None:
        disabled = _service(enabled=False)
        enabled = _service()
        assert "endpoint-admin" not in repr(enabled.config)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_dashboard_app(disabled)),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/api/rest-devices", json=_registration_payload()
                )
                assert response.status_code == 404

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_dashboard_app(enabled)),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/api/rest-devices", json=_registration_payload()
                )
                assert response.status_code == 401
        finally:
            await disabled.stop()
            await enabled.stop()

    asyncio.run(scenario())


def test_dashboard_registers_dispatches_and_deregisters_endpoint():
    async def scenario() -> None:
        requests: list[dict] = []
        service = _service()
        _install_endpoint_client(service, requests)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            registered = await client.post(
                "/api/rest-devices",
                json=_registration_payload(),
                headers=ADMIN_HEADERS,
            )
            assert registered.status_code == 200, registered.text
            assert registered.json()["base_url"] == "http://endpoint.local"

            public_device = (await client.get("/api/devices")).json()[0]
            assert public_device["transport"] == "http_endpoint"
            assert public_device["base_url"] == ""

            submitted = await client.post(
                "/api/tasks",
                json={"request_text": "hello", "execution_mode": "single"},
            )
            assert submitted.json()["output_text"] == "echo:hello"
            assert requests[-1]["request_text"] == "hello"

            removed = await client.delete(
                "/api/rest-devices/edge-01", headers=ADMIN_HEADERS
            )
            assert removed.status_code == 200
            assert await service.sessions.get("edge-01") is None
            assert (await client.get("/api/devices")).json() == []

        await service.stop()

    asyncio.run(scenario())


def test_endpoint_url_policy_and_device_id_collisions_are_rejected():
    async def scenario() -> None:
        service = _service()
        try:
            outside = _endpoint()
            outside = HttpEndpoint(
                device=outside.device,
                base_url="http://10.20.30.40",
            )
            try:
                await service.register_http_device(outside)
            except EndpointError as exc:
                assert "allowlist" in str(exc)
            else:
                raise AssertionError("outside URL should be rejected")

            await service.sessions.register(AgentSession("edge-01"))
            try:
                await service.register_http_device(_endpoint())
            except SessionConflictError as exc:
                assert "grpc_stream" in str(exc)
            else:
                raise AssertionError("transport collision should be rejected")
        finally:
            await service.stop()

    asyncio.run(scenario())


def test_endpoint_credentials_are_resolved_from_environment(monkeypatch):
    async def scenario() -> None:
        monkeypatch.setenv("EDGE_API_TOKEN", "secret-value")
        requests: list[dict] = []
        service = _service()
        _install_endpoint_client(service, requests, expected_token="secret-value")
        try:
            await service.register_http_device(
                _endpoint(credential_env="EDGE_API_TOKEN")
            )
            response = await service.SubmitTask(
                pb.SubmitTaskRequest(
                    request_text="authenticated", execution_mode="single"
                ),
                None,
            )
            assert response.success
            assert requests[-1]["request_text"] == "authenticated"
        finally:
            await service.stop()

    asyncio.run(scenario())


def test_profile_context_requires_endpoint_trust():
    async def scenario() -> None:
        requests: list[dict] = []
        service = _service()
        _install_endpoint_client(service, requests)
        profile = service.profiles.create(
            person_name="Alex", notes="Prefers implementation details"
        )
        service.profiles.associate_device("phone-01", profile.profile_id, "Phone")
        try:
            await service.register_http_device(_endpoint())
            first = await service.SubmitTask(
                pb.SubmitTaskRequest(
                    request_text="Plan this",
                    execution_mode="single",
                    origin_device_id="phone-01",
                ),
                None,
            )
            assert first.success
            assert requests[-1]["request_text"] == "Plan this"

            await service.register_http_device(_endpoint(allow_profile_context=True))
            second = await service.SubmitTask(
                pb.SubmitTaskRequest(
                    request_text="Plan this",
                    execution_mode="single",
                    origin_device_id="phone-01",
                ),
                None,
            )
            assert second.success
            assert requests[-1]["request_text"] == (
                "About the user:\nPrefers implementation details\n\nRequest:\nPlan this"
            )
        finally:
            await service.stop()

    asyncio.run(scenario())


def test_endpoint_configuration_persists_without_secret_value(tmp_path):
    database = tmp_path / "state.sqlite3"
    store = HttpEndpointStore(database)
    store.put(_endpoint(credential_env="EDGE_API_TOKEN", allow_profile_context=True))
    store.close()

    reopened = HttpEndpointStore(database)
    loaded = reopened.get("edge-01")
    assert loaded.credential_env == "EDGE_API_TOKEN"
    assert loaded.allow_profile_context is True
    assert loaded.device.models[0].model_id == "remote-model"
    assert "secret-value" not in database.read_text(errors="ignore")
    reopened.close()


def test_service_restores_persisted_endpoints_on_start(tmp_path):
    async def scenario() -> None:
        database = tmp_path / "state.sqlite3"
        store = HttpEndpointStore(database)
        store.put(_endpoint())
        store.close()

        service = _service(state_db_path=str(database))
        _install_endpoint_client(service, [])
        try:
            await service.start()
            session = await service.sessions.get("edge-01")
            assert session is not None
            assert session.transport == "http_endpoint"
            assert service.registry.get("edge-01").stream_connected
        finally:
            await service.stop()

    asyncio.run(scenario())


def test_discovery_validates_untrusted_endpoint_metadata():
    async def scenario() -> None:
        service = _service()
        _install_endpoint_client(service, [])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_dashboard_app(service)),
            base_url="http://test",
        ) as client:
            discovered = await client.post(
                "/api/rest-devices/discover",
                json={"base_url": "http://endpoint.local"},
                headers=ADMIN_HEADERS,
            )
            assert discovered.status_code == 200
            assert discovered.json()["models"][0]["model_id"] == "discovered-model"
        await service.stop()

    asyncio.run(scenario())

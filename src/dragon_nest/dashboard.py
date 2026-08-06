from __future__ import annotations

from io import BytesIO
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .dispatch import DeviceOfflineError
from .models import (
    Device,
    HardwareInventory,
    HealthState,
    HealthStatus,
    ModelCapability,
    TaskResult,
)
from .enrollment import EnrollmentError, EnrollmentStatus
from .profiles import PersonalProfile, ProfileError
from .regimes import build_regime_report
from .proto import dragonnest_pb2 as pb
from .tasks import AttemptState, TaskRecord
from .transport.brain import BrainService
from .transport.http_device import HttpDeviceConfig, HttpDeviceError


WEB_ROOT = Path(__file__).with_name("web")


class SteeringRequest(BaseModel):
    enabled: bool = False
    vector_id: str = ""
    model_family: str = ""
    target_layer: int = 0
    alpha: float = 0
    positions: str = "last"
    allow_remote_vector: bool = False


class TaskSubmission(BaseModel):
    request_text: str = Field(min_length=1)
    preferred_mode: str = "auto"
    execution_mode: str = "auto"
    origin_device_id: str = ""
    reducer: str = "mock_synthesis"
    timeout_ms: int = Field(default=30_000, ge=100, le=300_000)
    steering: SteeringRequest = Field(default_factory=SteeringRequest)
    use_profile_steering: bool = True


class SimulationRequest(BaseModel):
    offline: bool | None = None
    reachable: bool | None = None
    thermal_level: float | None = Field(default=None, ge=0, le=1)
    battery_pct: float | None = Field(default=None, ge=0, le=100)
    charging: bool | None = None
    available_memory_mb: int | None = Field(default=None, ge=0)
    cpu_utilization: float | None = Field(default=None, ge=0, le=1)
    accelerator_utilization: float | None = Field(default=None, ge=0, le=1)
    network_rtt_ms: float | None = Field(default=None, ge=0)


class EnrollmentSessionRequest(BaseModel):
    brain_host: str = Field(min_length=1, max_length=253)
    brain_port: int = Field(default=50051, ge=1, le=65535)
    use_tls: bool = False
    ttl_seconds: int = Field(default=300, ge=30, le=900)
    # Dashboard enrollment remains supported, but mobile clients now collect
    # profile preferences locally and send them on their first registration.
    person_name: str = Field(default="", max_length=120)
    device_name: str = Field(default="", max_length=120)
    preferred_mode: str = "auto"
    steering_vector_id: str = ""
    steering_alpha: float = 0
    steering_positions: str = "last"
    allow_remote_vector: bool = False
    notes: str = Field(default="", max_length=500)


class ModelCapabilityPayload(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    model_family: str = Field(min_length=1, max_length=120)
    role: str = Field(min_length=1, max_length=60)
    task_classes: list[str] = Field(default_factory=list)
    max_context_tokens: int = Field(default=0, ge=0)
    warm: bool = False
    quality_score: float = Field(default=0.5, ge=0, le=1)
    model_version: str = ""
    tokenizer_id: str = ""
    precision: str = ""
    boundary_format: str = ""
    steering_vector_ids: list[str] = Field(default_factory=list)
    supported_steering_layers: list[int] = Field(default_factory=list)
    runtime_name: str = "http"
    runtime_version: str = ""
    supported_accelerators: list[str] = Field(default_factory=lambda: ["cpu"])
    min_memory_mb: int = Field(default=0, ge=0)
    supports_steering: bool = False
    supports_data_parallel: bool = True
    supports_layer_pipeline: bool = False


class HardwareInventoryPayload(BaseModel):
    manufacturer: str = ""
    model: str = ""
    device: str = ""
    os_version: str = ""
    api_level: int = 0
    soc_manufacturer: str = ""
    soc_model: str = ""
    cpu_abis: list[str] = Field(default_factory=list)
    cpu_core_count: int = 0
    total_storage_mb: int = 0
    available_storage_mb: int = 0
    npu_status: str = "not_probed"
    npu_name: str = ""
    qnn_runtime_version: str = ""


class RestDeviceRegistration(BaseModel):
    """Registers a device the brain reaches over plain HTTP instead of the
    gRPC agent stream: any endpoint API server (e.g. a standalone inference
    server) or a local network device exposing the HttpDeviceSession
    contract (see transport/http_device.py).

    Leave `models` empty to have the brain auto-discover the endpoint's
    metadata and capabilities by calling GET {base_url}/info itself."""

    device_id: str = Field(min_length=1, max_length=200)
    display_name: str = Field(default="", max_length=120)
    device_type: str = Field(default="endpoint", max_length=40)
    platform: str = Field(default="", max_length=40)
    total_memory_mb: int = Field(default=0, ge=0)
    base_url: str = Field(min_length=1, max_length=500)
    api_key: str = Field(default="", max_length=500)
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    poll_interval_seconds: float = Field(default=5.0, ge=1, le=300)
    models: list[ModelCapabilityPayload] = Field(default_factory=list)
    hardware: HardwareInventoryPayload = Field(default_factory=HardwareInventoryPayload)


class HttpEndpointProbeRequest(BaseModel):
    base_url: str = Field(min_length=1, max_length=500)
    api_key: str = Field(default="", max_length=500)


class EndpointInfoPayload(BaseModel):
    """Shape expected back from an endpoint's GET /info route."""

    display_name: str = ""
    device_type: str = ""
    platform: str = ""
    total_memory_mb: int = Field(default=0, ge=0)
    models: list[ModelCapabilityPayload] = Field(default_factory=list)
    hardware: HardwareInventoryPayload = Field(default_factory=HardwareInventoryPayload)


class PersonalProfileUpdate(BaseModel):
    person_name: str = Field(min_length=1, max_length=120)
    preferred_mode: str = "auto"
    steering_vector_id: str = ""
    steering_alpha: float = 0
    steering_positions: str = "last"
    allow_remote_vector: bool = False
    notes: str = Field(default="", max_length=500)


def create_dashboard_app(service: BrainService) -> FastAPI:
    app = FastAPI(title="DragonNest Brain API", version="0.1.0")
    app.state.brain = service
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

    @app.get("/", include_in_schema=False)
    async def user_page():
        return FileResponse(WEB_ROOT / "app" / "index.html")

    @app.get("/admin", include_in_schema=False)
    async def admin_page():
        return FileResponse(WEB_ROOT / "admin" / "index.html")

    @app.get("/regimes", include_in_schema=False)
    async def regimes_page():
        return FileResponse(WEB_ROOT / "regimes" / "index.html")

    @app.get("/manifest.webmanifest", include_in_schema=False)
    async def manifest():
        return FileResponse(
            WEB_ROOT / "manifest.webmanifest", media_type="application/manifest+json"
        )

    @app.get("/sw.js", include_in_schema=False)
    async def service_worker():
        return FileResponse(
            WEB_ROOT / "sw.js",
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/health")
    async def api_health():
        return {
            "status": "ok",
            "brain_id": service.config.brain_id,
            "device_count": len(service.registry.records()),
            "task_count": len(service.tasks.records()),
        }

    @app.get("/api/devices")
    async def api_devices():
        return [_device_dict(service, record) for record in service.registry.records()]

    @app.get("/api/regimes")
    async def api_regimes():
        devices = [record.device for record in service.registry.records()]
        return build_regime_report(devices)

    @app.post("/api/rest-devices/discover")
    async def api_discover_rest_device(request: HttpEndpointProbeRequest):
        info = await _probe_http_endpoint(service, request.base_url, request.api_key)
        return info.model_dump()

    @app.post("/api/rest-devices")
    async def api_register_rest_device(request: RestDeviceRegistration):
        models = request.models
        display_name = request.display_name
        device_type = request.device_type
        platform = request.platform
        total_memory_mb = request.total_memory_mb
        hardware = request.hardware
        if not models:
            info = await _probe_http_endpoint(
                service, request.base_url, request.api_key
            )
            models = info.models
            display_name = display_name or info.display_name
            platform = platform or info.platform
            total_memory_mb = total_memory_mb or info.total_memory_mb
            if info.device_type:
                device_type = info.device_type
            if info.hardware != HardwareInventoryPayload():
                hardware = info.hardware
            if not models:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "endpoint reported no models via /info; supply "
                        "`models` explicitly instead"
                    ),
                )
        device = Device(
            device_id=request.device_id,
            display_name=display_name or request.device_id,
            device_type=device_type,
            platform=platform,
            total_memory_mb=total_memory_mb,
            health=HealthState(available_memory_mb=total_memory_mb, reachable=True),
            models=tuple(
                ModelCapability(
                    model_id=model.model_id,
                    model_family=model.model_family,
                    role=model.role,
                    task_classes=tuple(model.task_classes),
                    max_context_tokens=model.max_context_tokens,
                    warm=model.warm,
                    quality_score=model.quality_score,
                    model_version=model.model_version,
                    tokenizer_id=model.tokenizer_id,
                    precision=model.precision,
                    boundary_format=model.boundary_format,
                    steering_vector_ids=tuple(model.steering_vector_ids),
                    supported_steering_layers=tuple(model.supported_steering_layers),
                    runtime_name=model.runtime_name,
                    runtime_version=model.runtime_version,
                    supported_accelerators=tuple(model.supported_accelerators),
                    min_memory_mb=model.min_memory_mb,
                    supports_steering=model.supports_steering,
                    supports_data_parallel=model.supports_data_parallel,
                    supports_layer_pipeline=model.supports_layer_pipeline,
                )
                for model in models
            ),
            hardware=HardwareInventory(**hardware.model_dump()),
        )
        endpoint = HttpDeviceConfig(
            device_id=request.device_id,
            base_url=request.base_url,
            api_key=request.api_key,
            request_timeout_seconds=request.request_timeout_seconds,
            poll_interval_seconds=request.poll_interval_seconds,
        )
        try:
            await service.register_http_device(device, endpoint)
        except HttpDeviceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _device_dict(service, service.registry.get(request.device_id))

    @app.delete("/api/rest-devices/{device_id}")
    async def api_deregister_rest_device(device_id: str):
        if device_id not in service.http_devices:
            raise HTTPException(
                status_code=404,
                detail="device not found or is not an HTTP endpoint device",
            )
        await service.deregister_http_device(device_id)
        return {"device_id": device_id, "status": "deregistered"}

    @app.post("/api/enrollment-sessions")
    async def api_create_enrollment(request: EnrollmentSessionRequest):
        if not service.config.dev_mode:
            raise HTTPException(
                status_code=409,
                detail="QR token enrollment is available only in dev mode",
            )
        enrollment_values = request.model_dump(
            include={"brain_host", "brain_port", "use_tls", "ttl_seconds"}
        )
        profile_values = request.model_dump(
            include={
                "person_name",
                "preferred_mode",
                "steering_vector_id",
                "steering_alpha",
                "steering_positions",
                "allow_remote_vector",
                "notes",
            }
        )
        session = None
        try:
            session = service.enrollment.create(**enrollment_values)
            if profile_values["person_name"].strip():
                _validate_profile_steering(service, profile_values)
                profile = service.profiles.create(**profile_values)
                session.profile_id = profile.profile_id
                session.device_name = request.device_name.strip()
            elif any(
                profile_values[key]
                for key in (
                    "steering_vector_id",
                    "steering_alpha",
                    "allow_remote_vector",
                    "notes",
                )
            ) or profile_values["preferred_mode"] != "auto":
                raise ProfileError("person_name is required when creating a profile")
        except (EnrollmentError, ProfileError) as exc:
            if session is not None:
                service.enrollment.cancel(session.session_id)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **session.public_dict(),
            "qr_url": f"/api/enrollment-sessions/{session.session_id}/qr.svg",
        }

    @app.get("/api/enrollment-sessions/{session_id}")
    async def api_enrollment_status(session_id: str):
        try:
            session = service.enrollment.get(session_id)
            if session.status == EnrollmentStatus.EXPIRED and session.profile_id:
                service.profiles.delete_if_unassociated(session.profile_id)
            return session.public_dict()
        except EnrollmentError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/enrollment-sessions/{session_id}/qr.svg")
    async def api_enrollment_qr(session_id: str):
        try:
            session = service.enrollment.get(session_id)
        except EnrollmentError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if session.status != EnrollmentStatus.PENDING:
            raise HTTPException(
                status_code=410,
                detail=f"enrollment session is {session.status.value.lower()}",
            )
        import qrcode
        import qrcode.image.svg

        image = qrcode.make(
            session.qr_payload(), image_factory=qrcode.image.svg.SvgPathImage
        )
        output = BytesIO()
        image.save(output)
        return Response(
            content=output.getvalue(),
            media_type="image/svg+xml",
            headers={"Cache-Control": "no-store"},
        )

    @app.delete("/api/enrollment-sessions/{session_id}")
    async def api_cancel_enrollment(session_id: str):
        try:
            session = service.enrollment.cancel(session_id)
            if session.status == EnrollmentStatus.CANCELLED and session.profile_id:
                service.profiles.delete_if_unassociated(session.profile_id)
            return session.public_dict()
        except EnrollmentError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/personal-profiles")
    async def api_personal_profiles():
        return [_profile_dict(profile) for profile in service.profiles.all()]

    @app.put("/api/personal-profiles/{profile_id}")
    async def api_update_personal_profile(
        profile_id: str, request: PersonalProfileUpdate
    ):
        values = request.model_dump()
        try:
            _validate_profile_steering(service, values)
            return _profile_dict(service.profiles.update(profile_id, **values))
        except ProfileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/devices/{device_id}/simulate")
    async def api_simulate_device(device_id: str, request: SimulationRequest):
        try:
            record = service.registry.get(device_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="device not found") from exc
        changes = request.model_dump(exclude_none=True)
        if request.offline is True:
            service.set_device_simulation(device_id, changes)
            service.dispatch.handle_device_offline(device_id)
            return _device_dict(service, service.registry.get(device_id))
        if request.offline is False:
            changes.pop("offline", None)
            changes.setdefault("reachable", True)
        service.set_device_simulation(device_id, changes)
        health = replace(
            record.device.health,
            status=HealthStatus.HEALTHY,
            **changes,
        )
        service.registry.heartbeat(
            device_id,
            health,
            simulated_constraint=bool(changes),
        )
        return _device_dict(service, service.registry.get(device_id))

    @app.get("/api/tasks")
    async def api_tasks(limit: int = 50):
        records = [
            task
            for task in service.tasks.records()
            if task.task_id in service.task_profiles
        ][-max(1, min(limit, 200)) :]
        return [_task_dict(service, task) for task in reversed(records)]

    @app.get("/api/tasks/{task_id:path}")
    async def api_task(task_id: str):
        try:
            task = service.tasks.get(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        return _task_dict(service, task)

    @app.post("/api/tasks")
    async def api_submit_task(request: TaskSubmission):
        response = await service.SubmitTask(
            pb.SubmitTaskRequest(
                request_text=request.request_text,
                preferred_mode=request.preferred_mode,
                execution_mode=request.execution_mode,
                origin_device_id=request.origin_device_id,
                reducer=request.reducer,
                timeout_ms=request.timeout_ms,
                steering=pb.SteeringSpec(**request.steering.model_dump()),
                use_profile_steering=request.use_profile_steering,
            ),
            None,
        )
        return _response_dict(response)

    @app.get("/api/events")
    async def api_events(limit: int = 100):
        events: list[dict[str, Any]] = []
        for event in service.registry.events():
            events.append(
                {
                    "timestamp": event.timestamp,
                    "source": "registry",
                    "type": event.status.value,
                    "subject": event.device_id,
                    "message": event.reason,
                }
            )
        for event in service.tasks.events():
            events.append(
                {
                    "timestamp": event.timestamp,
                    "source": "task",
                    "type": event.event_type,
                    "subject": event.task_id,
                    "message": event.message,
                    "attempt_id": event.attempt_id,
                    "device_id": event.device_id,
                }
            )
        events.sort(key=lambda item: item["timestamp"], reverse=True)
        return events[: max(1, min(limit, 500))]

    @app.get("/api/steering-vectors")
    async def api_steering_vectors():
        if service.steering_registry is None:
            return []
        return [asdict(vector) for vector in service.steering_registry.vectors()]

    return app


def _device_dict(service: BrainService, record) -> dict[str, Any]:
    device = record.device
    personal_profile = service.profiles.profile_for_device(device.device_id)
    active_task_ids = sorted(
        {
            *record.active_task_ids,
            *{
            task.task_id
            for task in service.tasks.records()
            for attempt in task.attempts
            if attempt.device_id == device.device_id
            and attempt.state in {AttemptState.DISPATCHED, AttemptState.RUNNING}
            },
        }
    )
    http_session = service.http_devices.get(device.device_id)
    return {
        "device_id": device.device_id,
        "display_name": device.display_name,
        "device_type": device.device_type,
        "platform": device.platform,
        "status": record.status.value,
        "connected": record.stream_connected,
        "transport": "http_endpoint" if http_session else "grpc_stream",
        "base_url": http_session.config.base_url if http_session else "",
        "last_heartbeat": record.last_heartbeat,
        "health": {
            "battery_pct": device.health.battery_pct,
            "charging": device.health.charging,
            "thermal_level": device.health.thermal_level,
            "cpu_utilization": device.health.cpu_utilization,
            "accelerator_utilization": device.health.accelerator_utilization,
            "available_memory_mb": device.health.available_memory_mb,
            "network_rtt_ms": device.health.network_rtt_ms,
            "reachable": device.health.reachable,
        },
        "models": [
            {
                "model_id": model.model_id,
                "role": model.role,
                "family": model.model_family,
                "version": model.model_version,
                "runtime": model.runtime_name,
                "runtime_version": model.runtime_version,
                "accelerators": model.supported_accelerators,
                "min_memory_mb": model.min_memory_mb,
                "warm": model.warm,
                "steering_vectors": model.steering_vector_ids,
                "segment": asdict(model.segment) if model.segment else None,
            }
            for model in device.models
        ],
        "hardware": asdict(device.hardware),
        "active_tasks": active_task_ids,
        "simulated_constraint": record.simulated_constraint,
        "personal_profile": (
            _profile_dict(personal_profile) if personal_profile else None
        ),
    }


async def _probe_http_endpoint(
    service: BrainService, base_url: str, api_key: str
) -> EndpointInfoPayload:
    try:
        raw = await service.fetch_http_endpoint_info(base_url, api_key)
    except HttpDeviceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DeviceOfflineError as exc:
        raise HTTPException(
            status_code=502, detail=f"could not reach endpoint: {exc}"
        ) from exc
    try:
        return EndpointInfoPayload.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"endpoint returned an invalid /info response: {exc}",
        ) from exc


def _task_dict(service: BrainService, task: TaskRecord) -> dict[str, Any]:
    result = task.result if isinstance(task.result, TaskResult) else None
    profile = service.task_profiles.get(task.task_id)
    plan = service.execution_plans.get(task.task_id)
    steering = service._steering_specs.get(task.task_id)
    return {
        "task_id": task.task_id,
        "state": task.state.value,
        "request": task.request,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "accepted_attempt_id": task.accepted_attempt_id,
        "error_code": task.error_code,
        "error_message": task.error_message,
        "profile": asdict(profile) if profile else None,
        "execution_mode": plan.execution_mode.value if plan else "internal",
        "origin_device_id": plan.origin_device_id if plan else "",
        "reducer": plan.reducer if plan else "",
        "route_reasons": service._route_reasons.get(task.task_id, ()),
        "steering": asdict(steering) if steering else None,
        "attempts": [
            {
                "attempt_id": attempt.attempt_id,
                "device_id": attempt.device_id,
                "state": attempt.state.value,
                "error_code": attempt.error_code,
                "error_message": attempt.error_message,
                "created_at": attempt.created_at,
                "updated_at": attempt.updated_at,
            }
            for attempt in task.attempts
        ],
        "progress": _progress_dict(plan, service, task.task_id),
        "result": _result_dict(result),
        "stale_result_count": len(task.stale_results),
    }


def _progress_dict(plan, service: BrainService, parent_task_id: str):
    if plan is None:
        return []
    items = []
    planned = plan.tasks if plan.tasks else plan.stages
    for item in planned:
        item_id = getattr(item, "shard_id", None) or item.stage_id
        child_id = f"{parent_task_id}:{item_id}"
        try:
            child = service.tasks.get(child_id)
        except KeyError:
            continue
        latest = child.attempts[-1] if child.attempts else None
        child_result = child.result
        if plan.reducer == "first_success":
            for replica_index, attempt in enumerate(child.attempts, start=1):
                items.append(
                    {
                        "id": f"{item_id} / replica-{replica_index}",
                        "state": attempt.state.value,
                        "device_id": attempt.device_id,
                        "model_id": service._attempt_models.get(
                            attempt.attempt_id, ""
                        ),
                        "retry_count": 0,
                        "latency_ms": (
                            attempt.result.metrics.execution_latency_ms
                            if isinstance(attempt.result, TaskResult)
                            and attempt.result.metrics
                            else 0
                        ),
                        "winner": attempt.attempt_id == child.accepted_attempt_id,
                    }
                )
            continue
        items.append(
            {
                "id": item_id,
                "state": child.state.value,
                "device_id": latest.device_id if latest else "",
                "model_id": (
                    service._attempt_models.get(latest.attempt_id, "") if latest else ""
                ),
                "retry_count": max(0, len(child.attempts) - 1),
                "winner": False,
                "latency_ms": (
                    child_result.metrics.execution_latency_ms
                    if isinstance(child_result, TaskResult) and child_result.metrics
                    else 0
                ),
            }
        )
    return items


def _result_dict(result: TaskResult | None):
    if result is None:
        return None
    return {
        "success": result.success,
        "output_text": result.output_text,
        "device_id": result.device_id,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "latency_ms": result.latency_ms,
        "metrics": asdict(result.metrics) if result.metrics else None,
    }


def _response_dict(response: pb.SubmitTaskResponse):
    return {
        "task_id": response.task_id,
        "state": response.state,
        "success": response.success,
        "output_text": response.output_text,
        "error_code": response.error_code,
        "error_message": response.error_message,
        "accepted_attempt_id": response.accepted_attempt_id,
        "device_id": response.device_id,
        "model_id": response.model_id,
        "route_reasons": list(response.route_reasons),
        "origin_device_id": response.origin_device_id,
        "reducer": response.reducer,
        "steering": {
            "enabled": response.steering.enabled,
            "vector_id": response.steering.vector_id,
            "target_layer": response.steering.target_layer,
            "alpha": response.steering.alpha,
            "positions": response.steering.positions,
        },
    }


def _profile_dict(profile: PersonalProfile) -> dict[str, Any]:
    return asdict(profile)


def _validate_profile_steering(
    service: BrainService, values: dict[str, Any]
) -> None:
    vector_id = str(values.get("steering_vector_id", ""))
    if not vector_id:
        return
    if service.steering_registry is None:
        raise ProfileError("Brain has no steering registry configured")
    vectors = {
        vector.vector_id: vector for vector in service.steering_registry.vectors()
    }
    vector = vectors.get(vector_id)
    if vector is None:
        raise ProfileError(f"unknown steering vector {vector_id}")
    alpha = float(values.get("steering_alpha", 0))
    positions = str(values.get("steering_positions", "last"))
    if not vector.alpha_min <= alpha <= vector.alpha_max:
        raise ProfileError(
            f"steering alpha must be between {vector.alpha_min} and {vector.alpha_max}"
        )
    if positions not in vector.positions:
        raise ProfileError(f"steering positions {positions!r} are not supported")

from __future__ import annotations

import hmac
import socket
from io import BytesIO
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .deployments import ArtifactState
from .dispatch import DeviceOfflineError
from .endpoints import EndpointError, HttpEndpoint
from .scheduler import RequestSpec, RoutePlan
from .models import (
    Device,
    HardwareInventory,
    HealthState,
    HealthStatus,
    ModelCapability,
    ModelSegment,
    TaskResult,
)
from .enrollment import EnrollmentError, EnrollmentStatus
from .profiles import PersonalProfile, ProfileError
from .regimes import build_regime_report
from .proto import dragonnest_pb2 as pb
from .tasks import AttemptState, TaskRecord
from .transport.brain import BrainService
from .transport.sessions import SessionConflictError


WEB_ROOT = Path(__file__).with_name("web")


def _detect_lan_addresses() -> list[str]:
    """All LAN-reachable IPv4 addresses bound to this host's interfaces.

    A host can have more than one live network (e.g. a workshop Wi-Fi that
    isolates clients from each other, plus a laptop-hosted Mobile Hotspot
    the phone actually joined), so callers must not assume a single
    "the" address is correct without knowing which network the other
    device is on.
    """
    addresses: list[str] = []
    try:
        _, _, host_addresses = socket.gethostbyname_ex(socket.gethostname())
        for address in host_addresses:
            if not address.startswith("127.") and address not in addresses:
                addresses.append(address)
    except OSError:
        pass
    if not addresses:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                address = sock.getsockname()[0]
            if address and not address.startswith("127."):
                addresses.append(address)
        except OSError:
            pass
    return addresses


class SteeringRequest(BaseModel):
    enabled: bool = False
    vector_id: str = ""
    model_family: str = ""
    target_layer: int = 0
    alpha: float = 0
    positions: str = "last"
    allow_remote_vector: bool = False
    mode: str = "runtime_vector"
    behavior_profile_id: str = ""


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
    gpu_utilization: float | None = Field(default=None, ge=0, le=1)
    npu_utilization: float | None = Field(default=None, ge=0, le=1)
    network_rtt_ms: float | None = Field(default=None, ge=0)
    # Deployment/steering overlays (not health fields):
    artifact_states: dict[str, str] | None = None  # artifact_id -> ArtifactState
    runtime_steering_enabled: bool | None = None


class RoutePlanRequest(BaseModel):
    request_text: str = Field(default="", max_length=8000)
    base_model_family: str = "mock"
    behavior_profile_id: str = ""
    estimated_input_tokens: int = Field(default=256, ge=1, le=1_000_000)
    estimated_output_tokens: int = Field(default=128, ge=1, le=1_000_000)
    privacy: str = "trusted_fabric"
    latency_preference: str = "interactive"
    origin_device_id: str = ""
    fallback_policy_override: str = ""

    def to_spec(self) -> RequestSpec:
        return RequestSpec(**self.model_dump())


class BehaviorTaskSubmission(RoutePlanRequest):
    request_text: str = Field(min_length=1, max_length=8000)
    timeout_ms: int = Field(default=30_000, ge=100, le=300_000)

    def to_spec(self) -> RequestSpec:
        values = self.model_dump()
        values.pop("timeout_ms", None)
        return RequestSpec(**values)


class ProvisioningRequest(BaseModel):
    profile_id: str = Field(min_length=1, max_length=200)
    device_id: str = Field(min_length=1, max_length=200)
    artifact_id: str = Field(min_length=1, max_length=200)


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


class ModelSegmentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_id: str = Field(min_length=1, max_length=200)
    start_layer: int | None = Field(default=None, ge=0)
    end_layer: int | None = Field(default=None, ge=0)
    total_layers: int = Field(default=0, ge=0)
    includes_embedding: bool = False
    includes_lm_head: bool = False
    stage_index: int = Field(default=-1, ge=-1)
    stage_count: int = Field(default=0, ge=0)
    transformer_start_layer: int | None = Field(default=None, ge=0)
    transformer_end_layer: int | None = Field(default=None, ge=0)
    input_tensor: str = ""
    output_tensor: str = ""
    boundary_format: str = ""


class ModelCapabilityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    segment: ModelSegmentPayload | None = None
    runtime_name: str = "http"
    runtime_version: str = ""
    supported_accelerators: list[str] = Field(default_factory=lambda: ["cpu"])
    min_memory_mb: int = Field(default=0, ge=0)
    supports_steering: bool = False
    supports_data_parallel: bool = True
    supports_layer_pipeline: bool = False
    artifact_id: str = ""
    steering_modes: list[str] = Field(default_factory=lambda: ["none"])
    behavior_profile_ids: list[str] = Field(default_factory=list)
    target_compatibility_class: str = ""


class HardwareInventoryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    compatibility_key: str = ""


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
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="dragonnest", pattern=r"^(dragonnest|openai_chat)$")
    credential_env: str = Field(
        default="", max_length=120, pattern=r"^$|^[A-Za-z_][A-Za-z0-9_]*$"
    )
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    health_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    poll_interval_seconds: float = Field(default=5.0, ge=1, le=300)
    allow_profile_context: bool = False
    models: list[ModelCapabilityPayload] = Field(default_factory=list)
    hardware: HardwareInventoryPayload = Field(default_factory=HardwareInventoryPayload)


class HttpEndpointProbeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=1, max_length=500)
    credential_env: str = Field(
        default="", max_length=120, pattern=r"^$|^[A-Za-z_][A-Za-z0-9_]*$"
    )


class EndpointInfoPayload(BaseModel):
    """Shape expected back from an endpoint's GET /info route."""

    model_config = ConfigDict(extra="forbid")

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

    @app.get("/api/server-info")
    async def api_server_info():
        return {"lan_addresses": _detect_lan_addresses()}

    @app.get("/api/devices")
    async def api_devices():
        return [_device_dict(service, record) for record in service.registry.records()]

    @app.delete("/api/devices/{device_id:path}")
    async def api_remove_device(device_id: str):
        try:
            await service.remove_device(device_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="device not found") from exc
        return {"device_id": device_id, "status": "REMOVED"}

    @app.get("/api/regimes")
    async def api_regimes():
        devices = [record.device for record in service.registry.records()]
        return build_regime_report(devices)

    @app.post("/api/rest-devices/discover")
    async def api_discover_rest_device(
        request: HttpEndpointProbeRequest, admin_request: Request
    ):
        _require_endpoint_admin(service, admin_request)
        info = await _probe_http_endpoint(
            service, request.base_url, request.credential_env
        )
        return info.model_dump()

    @app.post("/api/rest-devices")
    async def api_register_rest_device(
        request: RestDeviceRegistration, admin_request: Request
    ):
        _require_endpoint_admin(service, admin_request)
        models = request.models
        display_name = request.display_name
        device_type = request.device_type
        platform = request.platform
        total_memory_mb = request.total_memory_mb
        hardware = request.hardware
        if not models and request.provider == "openai_chat":
            raise HTTPException(
                status_code=400,
                detail=(
                    "OpenAI-compatible endpoints don't expose DragonNest's /info "
                    "route; supply `models` explicitly (one entry per model id "
                    "you want to route to)"
                ),
            )
        if not models:
            info = await _probe_http_endpoint(
                service, request.base_url, request.credential_env
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
                    segment=(
                        ModelSegment(**model.segment.model_dump())
                        if model.segment is not None
                        else None
                    ),
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
            hardware=HardwareInventory(
                **{
                    **hardware.model_dump(),
                    "cpu_abis": tuple(hardware.cpu_abis),
                }
            ),
        )
        try:
            endpoint = HttpEndpoint(
                device=device,
                base_url=request.base_url,
                credential_env=request.credential_env,
                request_timeout_seconds=request.request_timeout_seconds,
                health_timeout_seconds=request.health_timeout_seconds,
                poll_interval_seconds=request.poll_interval_seconds,
                allow_profile_context=request.allow_profile_context,
                provider=request.provider,
            )
            await service.register_http_device(endpoint)
        except (EndpointError, SessionConflictError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _device_dict(
            service,
            service.registry.get(request.device_id),
            include_endpoint_details=True,
        )

    @app.delete("/api/rest-devices/{device_id}")
    async def api_deregister_rest_device(device_id: str, admin_request: Request):
        _require_endpoint_admin(service, admin_request)
        try:
            await service.deregister_http_device(device_id)
        except EndpointError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
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
        artifact_states = changes.pop("artifact_states", None)
        runtime_steering_enabled = changes.pop("runtime_steering_enabled", None)
        if artifact_states is not None:
            try:
                parsed_states = {
                    artifact_id: ArtifactState(state)
                    for artifact_id, state in artifact_states.items()
                }
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=f"invalid artifact state: {exc}"
                ) from exc
            service.set_deployment_simulation(device_id, parsed_states)
        if runtime_steering_enabled is not None:
            service.set_runtime_steering_enabled(
                device_id, runtime_steering_enabled
            )
        if not changes:
            return _device_dict(service, service.registry.get(device_id))
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

    @app.get("/api/behavior-profiles")
    async def api_behavior_profiles():
        if service.behavior_registry is None:
            return []
        return [
            {
                "profile_id": profile.profile_id,
                "display_name": profile.display_name,
                "description": profile.description,
                "base_model_family": profile.base_model_family,
                "version": profile.version,
                "policy_tags": profile.policy_tags,
                "fallback_policy": profile.fallback_policy.value,
                "provenance": profile.provenance,
                "evaluation_status": profile.evaluation_status,
                "realizations": [
                    {
                        "mode": realization.mode.value,
                        "vector_id": realization.vector_id,
                        "alpha": realization.alpha,
                        "injection_layer": realization.injection_layer,
                        "baked_artifact_id": realization.baked_artifact_id,
                        "verification_status": realization.verification_status,
                        "description": realization.describe(),
                    }
                    for realization in profile.realizations
                ],
            }
            for profile in service.behavior_registry.all()
        ]

    @app.get("/api/artifact-catalog")
    async def api_artifact_catalog():
        if service.artifact_catalog is None:
            return []
        return [asdict(artifact) for artifact in service.artifact_catalog.all()]

    @app.get("/api/deployments")
    async def api_deployments():
        if service.artifact_catalog is None:
            return []
        index = service.current_deployment_index()
        return [
            {
                "device_id": record.device.device_id,
                "steering_realization_modes": (
                    service.device_steering_realization_modes(
                        record.device.device_id
                    )
                ),
                "artifacts": [
                    {
                        "artifact_id": state.artifact_id,
                        "state": state.state.value,
                        "resident_bytes": state.resident_bytes,
                    }
                    for state in index.for_device(record.device.device_id)
                ],
            }
            for record in service.registry.records()
        ]

    @app.post("/api/route-plan")
    async def api_route_plan(request: RoutePlanRequest):
        _require_scheduler(service)
        try:
            plan = service.build_route_plan(request.to_spec())
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _route_plan_dict(plan)

    @app.post("/api/behavior-tasks")
    async def api_submit_behavior_task(request: BehaviorTaskSubmission):
        _require_scheduler(service)
        try:
            plan, response = await service.submit_behavior_task(
                request.to_spec(), timeout_ms=request.timeout_ms
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **_response_dict(response),
            "route_plan": _route_plan_dict(plan),
        }

    @app.get("/api/provisioning")
    async def api_provisioning_jobs():
        return [_provisioning_dict(job) for job in service.provisioning.jobs()]

    @app.post("/api/provisioning")
    async def api_start_provisioning(request: ProvisioningRequest):
        if service.artifact_catalog is not None:
            if service.artifact_catalog.maybe_get(request.artifact_id) is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown artifact {request.artifact_id}",
                )
        job = service.provisioning.start(
            request.profile_id, request.device_id, request.artifact_id
        )
        return _provisioning_dict(job)

    @app.post("/api/provisioning/{job_id}/advance")
    async def api_advance_provisioning(job_id: str):
        try:
            job = service.provisioning.advance(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _provisioning_dict(job)

    return app


def _require_scheduler(service: BrainService) -> None:
    if service.scheduler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "behavior-aware scheduling is not configured; start the Brain "
                "with an artifact catalog and behavior profiles"
            ),
        )


def _route_plan_dict(plan: RoutePlan) -> dict[str, Any]:
    def candidate_dict(candidate) -> dict[str, Any]:
        return {
            "device_id": candidate.device_id,
            "artifact_id": candidate.artifact.artifact_id,
            "artifact_behavior_profile": candidate.artifact.behavior_profile_id,
            "runtime": candidate.artifact.runtime,
            "quantization": candidate.artifact.quantization,
            "realization_mode": candidate.realization_mode,
            "realization": (
                candidate.realization.describe()
                if candidate.realization is not None
                else ""
            ),
            "deployment_state": candidate.deployment.state.value,
            "feasible": candidate.feasible,
            "rejection_reasons": candidate.rejection_reasons,
            "memory": asdict(candidate.memory) if candidate.memory else None,
            "cost": asdict(candidate.cost) if candidate.cost else None,
        }

    return {
        "request": asdict(plan.request),
        "behavior_profile": plan.profile.profile_id if plan.profile else "",
        "fallback_policy": plan.fallback_policy,
        "candidates": [candidate_dict(c) for c in plan.candidates],
        "chosen": (
            candidate_dict(plan.chosen) if plan.chosen is not None else None
        ),
        "steering": asdict(plan.steering),
        "prompt_prefix": plan.prompt_prefix,
        "explanation": plan.explanation,
        "error_code": plan.error_code,
        "provisioning_hint": plan.provisioning_hint,
    }


def _provisioning_dict(job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "profile_id": job.profile_id,
        "device_id": job.target_device_id,
        "artifact_id": job.artifact_id,
        "state": job.state.value,
        "detail": job.detail,
        "adapter": job.adapter_name,
        "history": [
            {"state": state, "detail": detail} for state, detail in job.history
        ],
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _device_dict(
    service: BrainService, record, *, include_endpoint_details: bool = False
) -> dict[str, Any]:
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
    try:
        endpoint = service.endpoints.get(device.device_id)
    except EndpointError:
        endpoint = None
    return {
        "device_id": device.device_id,
        "display_name": device.display_name,
        "device_type": device.device_type,
        "platform": device.platform,
        "status": record.status.value,
        "connected": record.stream_connected,
        "transport": "http_endpoint" if endpoint else "grpc_stream",
        "endpoint_provider": endpoint.provider if endpoint else "",
        "base_url": (
            endpoint.base_url if endpoint and include_endpoint_details else ""
        ),
        "allow_profile_context": (
            endpoint.allow_profile_context
            if endpoint and include_endpoint_details
            else False
        ),
        "last_heartbeat": record.last_heartbeat,
        "health": {
            "battery_pct": device.health.battery_pct,
            "charging": device.health.charging,
            "thermal_level": device.health.thermal_level,
            "cpu_utilization": device.health.cpu_utilization,
            "accelerator_utilization": device.health.accelerator_utilization,
            "gpu_utilization": device.health.gpu_utilization,
            "npu_utilization": device.health.npu_utilization,
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
        "simulated_fields": sorted(
            service._device_simulations.get(device.device_id, {}).keys()
        ),
        "personal_profile": (
            _profile_dict(personal_profile) if personal_profile else None
        ),
        "steering_realization_modes": (
            service.device_steering_realization_modes(device.device_id)
            if service.artifact_catalog is not None
            else ()
        ),
        "runtime_steering_enabled": (
            device.device_id not in service.runtime_steering_disabled
        ),
        "deployments": (
            [
                {
                    "artifact_id": state.artifact_id,
                    "state": state.state.value,
                }
                for state in service.current_deployment_index().for_device(
                    device.device_id
                )
            ]
            if service.artifact_catalog is not None
            else []
        ),
    }


async def _probe_http_endpoint(
    service: BrainService, base_url: str, credential_env: str
) -> EndpointInfoPayload:
    try:
        raw = await service.fetch_http_endpoint_info(base_url, credential_env)
    except EndpointError as exc:
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


def _require_endpoint_admin(service: BrainService, request: Request) -> None:
    if not service.config.http_endpoint_registration_enabled:
        raise HTTPException(status_code=404, detail="HTTP endpoint API is disabled")
    expected = service.config.http_endpoint_admin_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="HTTP endpoint admin token is not configured",
        )
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail="valid endpoint admin bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )


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
        "preferred_mode": plan.preferred_mode if plan else "auto",
        "pipeline_id": plan.pipeline_id if plan else "",
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
        "route_plan": (
            _route_plan_dict(service.route_plans[task.task_id])
            if task.task_id in service.route_plans
            else None
        ),
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

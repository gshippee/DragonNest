from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .models import HealthStatus, TaskResult
from .proto import dragonnest_pb2 as pb
from .tasks import AttemptState, TaskRecord
from .transport.brain import BrainService


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


def create_dashboard_app(service: BrainService) -> FastAPI:
    app = FastAPI(title="DragonNest Brain API", version="0.1.0")
    app.state.brain = service
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

    @app.get("/", include_in_schema=False)
    async def dashboard():
        return FileResponse(WEB_ROOT / "index.html")

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
    return {
        "device_id": device.device_id,
        "display_name": device.display_name,
        "device_type": device.device_type,
        "platform": device.platform,
        "status": record.status.value,
        "connected": record.stream_connected,
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
        "active_tasks": active_task_ids,
        "simulated_constraint": record.simulated_constraint,
    }


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

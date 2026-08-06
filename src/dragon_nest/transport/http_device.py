from __future__ import annotations

import base64
import binascii
import contextlib
import os
from typing import Any

import httpx

from ..dispatch import DeviceOfflineError
from ..endpoints import HttpEndpoint
from ..models import HealthState
from ..proto import dragonnest_pb2 as pb


class HttpDeviceSession:
    """Adapts the endpoint JSON API to the common protobuf session contract."""

    transport = "http_endpoint"

    def __init__(self, endpoint: HttpEndpoint, client: httpx.AsyncClient):
        self.device_id = endpoint.device.device_id
        self.endpoint = endpoint
        self.allow_profile_context = endpoint.allow_profile_context
        self._client = client
        self.closed = False

    def _url(self, path: str) -> str:
        return f"{self.endpoint.base_url.rstrip('/')}{path}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.endpoint.credential_env:
            credential = os.environ.get(self.endpoint.credential_env, "")
            if not credential:
                raise DeviceOfflineError(
                    f"credential environment variable "
                    f"{self.endpoint.credential_env!r} is not set"
                )
            headers["Authorization"] = f"Bearer {credential}"
        return headers

    async def _post(
        self, path: str, payload: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        if self.closed:
            raise DeviceOfflineError(f"device {self.device_id} endpoint is closed")
        try:
            response = await self._client.post(
                self._url(path),
                json=payload,
                headers=self._headers(),
                timeout=min(timeout_seconds, self.endpoint.request_timeout_seconds),
                follow_redirects=False,
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("response body must be a JSON object")
            return body
        except (httpx.HTTPError, ValueError) as exc:
            raise DeviceOfflineError(
                f"HTTP device {self.device_id} request to {path} failed: {exc}"
            ) from exc

    async def fetch_health(self) -> HealthState:
        body = await self._get("/health", self.endpoint.health_timeout_seconds)
        return _health_from_json(body)

    async def fetch_info(self) -> dict[str, Any]:
        return await self._get("/info", self.endpoint.health_timeout_seconds)

    async def _get(self, path: str, timeout_seconds: float) -> dict[str, Any]:
        if self.closed:
            raise DeviceOfflineError(f"device {self.device_id} endpoint is closed")
        try:
            response = await self._client.get(
                self._url(path),
                headers=self._headers(),
                timeout=timeout_seconds,
                follow_redirects=False,
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("response body must be a JSON object")
            return body
        except (httpx.HTTPError, ValueError) as exc:
            raise DeviceOfflineError(
                f"HTTP device {self.device_id} request to {path} failed: {exc}"
            ) from exc

    async def execute(
        self, command: pb.ExecuteTask, timeout_seconds: float
    ) -> pb.TaskResult:
        body = await self._post("/execute", _execute_json(command), timeout_seconds)
        return _task_result(body, command.task_id, command.attempt_id, self.device_id)

    async def execute_shard(
        self, command: pb.ExecuteShard, timeout_seconds: float
    ) -> pb.PartialTaskResult:
        body = await self._post(
            "/execute_shard",
            {
                **_execute_json(command),
                "shard_id": command.shard_id,
            },
            timeout_seconds,
        )
        result = _task_result(body, command.task_id, command.attempt_id, self.device_id)
        return pb.PartialTaskResult(
            task_id=result.task_id,
            attempt_id=result.attempt_id,
            shard_id=command.shard_id,
            device_id=result.device_id,
            success=result.success,
            output_text=result.output_text,
            error_code=result.error_code,
            error_message=result.error_message,
            metrics=result.metrics,
        )

    async def execute_pipeline_stage(
        self, command: pb.ExecutePipelineStage, timeout_seconds: float
    ) -> pb.PipelineStageResult:
        body = await self._post(
            "/execute_pipeline_stage",
            {
                **_execute_json(command),
                "stage_id": command.stage_id,
                "stage_index": command.stage_index,
                "input_boundary": _boundary_to_json(
                    command.input_boundary
                    if command.HasField("input_boundary")
                    else None
                ),
                "final_stage": command.final_stage,
            },
            timeout_seconds,
        )
        result = _task_result(body, command.task_id, command.attempt_id, self.device_id)
        output_boundary = _boundary_from_json(body.get("output_boundary"))
        return pb.PipelineStageResult(
            task_id=result.task_id,
            attempt_id=result.attempt_id,
            stage_id=command.stage_id,
            device_id=result.device_id,
            success=result.success,
            output_text=result.output_text,
            error_code=result.error_code,
            error_message=result.error_message,
            metrics=result.metrics,
            output_boundary=output_boundary,
        )

    async def cancel(self, task_id: str, attempt_id: str, reason: str) -> None:
        if self.closed:
            return
        with contextlib.suppress(Exception):
            await self._post(
                "/cancel",
                {"task_id": task_id, "attempt_id": attempt_id, "reason": reason},
                5.0,
            )

    async def close(self, graceful: bool = False) -> None:
        del graceful
        self.closed = True


async def fetch_endpoint_info(
    client: httpx.AsyncClient,
    base_url: str,
    credential_env: str = "",
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if credential_env:
        credential = os.environ.get(credential_env, "")
        if not credential:
            raise DeviceOfflineError(
                f"credential environment variable {credential_env!r} is not set"
            )
        headers["Authorization"] = f"Bearer {credential}"
    try:
        response = await client.get(
            f"{base_url.rstrip('/')}/info",
            headers=headers,
            timeout=timeout_seconds,
            follow_redirects=False,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("response body must be a JSON object")
        return body
    except (httpx.HTTPError, ValueError) as exc:
        raise DeviceOfflineError(f"HTTP endpoint info request failed: {exc}") from exc


def _execute_json(
    command: pb.ExecuteTask | pb.ExecuteShard | pb.ExecutePipelineStage,
) -> dict[str, Any]:
    return {
        "task_id": command.task_id,
        "attempt_id": command.attempt_id,
        "request_text": command.request_text,
        "model_id": command.model_id,
        "timeout_ms": command.timeout_ms,
        "steering": _steering_to_json(command.steering),
    }


def _steering_to_json(spec: pb.SteeringSpec) -> dict[str, Any]:
    return {
        "enabled": spec.enabled,
        "vector_id": spec.vector_id,
        "model_family": spec.model_family,
        "target_layer": spec.target_layer,
        "alpha": spec.alpha,
        "positions": spec.positions,
        "allow_remote_vector": spec.allow_remote_vector,
    }


def _boundary_to_json(boundary: pb.BoundaryTensor | None) -> dict[str, Any] | None:
    if boundary is None or not boundary.data:
        return None
    return {
        "tensor_name": boundary.tensor_name,
        "dtype": boundary.dtype,
        "shape": list(boundary.shape),
        "checksum": boundary.checksum,
        "data_base64": base64.b64encode(boundary.data).decode("ascii"),
    }


def _boundary_from_json(value: Any) -> pb.BoundaryTensor | None:
    if not value:
        return None
    if not isinstance(value, dict):
        raise DeviceOfflineError("output_boundary must be a JSON object")
    try:
        data = base64.b64decode(value.get("data_base64", ""), validate=True)
        shape = [int(dim) for dim in value.get("shape", [])]
    except (binascii.Error, TypeError, ValueError) as exc:
        raise DeviceOfflineError("output_boundary is malformed") from exc
    return pb.BoundaryTensor(
        tensor_name=str(value.get("tensor_name", "")),
        dtype=str(value.get("dtype", "")),
        shape=shape,
        checksum=str(value.get("checksum", "")),
        data=data,
    )


def _health_from_json(body: dict[str, Any]) -> HealthState:
    return HealthState(
        battery_pct=float(body.get("battery_pct", -1)),
        charging=bool(body.get("charging", False)),
        thermal_level=float(body.get("thermal_level", -1)),
        cpu_utilization=float(body.get("cpu_utilization", -1)),
        accelerator_utilization=float(body.get("accelerator_utilization", -1)),
        available_memory_mb=int(body.get("available_memory_mb", 0)),
        network_rtt_ms=float(body.get("network_rtt_ms", -1)),
        reachable=bool(body.get("reachable", True)),
    )


def _task_result(
    body: dict[str, Any], task_id: str, attempt_id: str, device_id: str
) -> pb.TaskResult:
    metrics_body = body.get("metrics") or {}
    metrics = None
    if not isinstance(metrics_body, dict):
        raise DeviceOfflineError("metrics must be a JSON object")
    if metrics_body.get("model_id"):
        metrics = pb.ExecutionMetrics(
            model_id=str(metrics_body.get("model_id", "")),
            model_version=str(metrics_body.get("model_version", "")),
            runtime_name=str(metrics_body.get("runtime_name", "")),
            runtime_version=str(metrics_body.get("runtime_version", "")),
            accelerator=str(metrics_body.get("accelerator", "")),
            execution_latency_ms=int(metrics_body.get("execution_latency_ms", 0)),
            error_code=str(metrics_body.get("error_code", "")),
            error_message=str(metrics_body.get("error_message", "")),
            observed_memory_delta_mb=int(
                metrics_body.get("observed_memory_delta_mb", 0) or 0
            ),
            observed_thermal_delta=float(
                metrics_body.get("observed_thermal_delta", 0) or 0
            ),
        )
    return pb.TaskResult(
        task_id=task_id,
        attempt_id=attempt_id,
        success=bool(body.get("success", False)),
        output_text=str(body.get("output_text", "")),
        device_id=device_id,
        error_code=str(body.get("error_code", "")),
        error_message=str(body.get("error_message", "")),
        metrics=metrics,
    )

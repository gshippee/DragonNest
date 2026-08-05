from __future__ import annotations

import base64
import contextlib
from dataclasses import dataclass
from typing import Any

import httpx

from ..dispatch import DeviceOfflineError
from ..models import ExecutionMetrics, HealthState, SteeringSpec, TaskResult
from ..proto import dragonnest_pb2 as pb


class HttpDeviceError(ValueError):
    pass


@dataclass(frozen=True)
class HttpDeviceConfig:
    device_id: str
    base_url: str
    api_key: str = ""
    request_timeout_seconds: float = 30.0
    health_timeout_seconds: float = 5.0
    poll_interval_seconds: float = 5.0


class HttpDeviceSession:
    """Adapts a remote HTTP endpoint into the same execute/execute_shard/
    execute_pipeline_stage/cancel surface AgentSession exposes for gRPC
    agents, so the brain can dispatch tasks to it without a persistent
    stream. The remote endpoint must implement:

      GET  {base_url}/health                -> device health snapshot
      GET  {base_url}/info                  -> device metadata + declared
                                                model capabilities (optional;
                                                used for capability discovery)
      POST {base_url}/execute               -> single-task execution
      POST {base_url}/execute_shard         -> data-parallel shard execution
      POST {base_url}/execute_pipeline_stage-> layer-pipeline stage execution
      POST {base_url}/cancel                -> best-effort cancellation

    All POST bodies and responses are JSON; see the _*_json helpers below
    for the exact shape.
    """

    def __init__(self, config: HttpDeviceConfig, client: httpx.AsyncClient):
        self.device_id = config.device_id
        self.config = config
        self._client = client
        self.closed = False

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{path}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
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
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DeviceOfflineError(
                f"HTTP device {self.device_id} request to {path} failed: {exc}"
            ) from exc

    async def fetch_health(self) -> HealthState:
        if self.closed:
            raise DeviceOfflineError(f"device {self.device_id} endpoint is closed")
        try:
            response = await self._client.get(
                self._url("/health"),
                headers=self._headers(),
                timeout=self.config.health_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DeviceOfflineError(
                f"HTTP device {self.device_id} health check failed: {exc}"
            ) from exc
        return _health_from_json(body)

    async def fetch_info(self) -> dict[str, Any]:
        """Fetch the endpoint's self-reported metadata and model
        capabilities, used to auto-fill a device registration instead of
        requiring them to be typed in by hand."""
        if self.closed:
            raise DeviceOfflineError(f"device {self.device_id} endpoint is closed")
        try:
            response = await self._client.get(
                self._url("/info"),
                headers=self._headers(),
                timeout=self.config.health_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DeviceOfflineError(
                f"HTTP device {self.device_id} info request failed: {exc}"
            ) from exc

    async def execute(
        self,
        task_id: str,
        attempt_id: str,
        request_text: str,
        model_id: str,
        timeout_ms: int,
        steering: SteeringSpec,
    ) -> TaskResult:
        body = await self._post(
            "/execute",
            {
                "task_id": task_id,
                "attempt_id": attempt_id,
                "request_text": request_text,
                "model_id": model_id,
                "timeout_ms": timeout_ms,
                "steering": _steering_to_json(steering),
            },
            timeout_ms / 1000,
        )
        return _task_result_from_json(body, task_id, attempt_id, self.device_id)

    async def execute_shard(
        self,
        task_id: str,
        attempt_id: str,
        shard_id: str,
        request_text: str,
        model_id: str,
        timeout_ms: int,
        steering: SteeringSpec,
    ) -> TaskResult:
        body = await self._post(
            "/execute_shard",
            {
                "task_id": task_id,
                "attempt_id": attempt_id,
                "shard_id": shard_id,
                "request_text": request_text,
                "model_id": model_id,
                "timeout_ms": timeout_ms,
                "steering": _steering_to_json(steering),
            },
            timeout_ms / 1000,
        )
        return _task_result_from_json(body, task_id, attempt_id, self.device_id)

    async def execute_pipeline_stage(
        self,
        task_id: str,
        attempt_id: str,
        stage_id: str,
        stage_index: int,
        request_text: str,
        model_id: str,
        input_boundary: pb.BoundaryTensor | None,
        final_stage: bool,
        timeout_ms: int,
        steering: SteeringSpec,
    ) -> tuple[TaskResult, pb.BoundaryTensor | None]:
        body = await self._post(
            "/execute_pipeline_stage",
            {
                "task_id": task_id,
                "attempt_id": attempt_id,
                "stage_id": stage_id,
                "stage_index": stage_index,
                "request_text": request_text,
                "model_id": model_id,
                "input_boundary": _boundary_to_json(input_boundary),
                "final_stage": final_stage,
                "timeout_ms": timeout_ms,
                "steering": _steering_to_json(steering),
            },
            timeout_ms / 1000,
        )
        result = _task_result_from_json(body, task_id, attempt_id, self.device_id)
        return result, _boundary_from_json(body.get("output_boundary"))

    async def cancel(self, task_id: str, attempt_id: str, reason: str) -> None:
        if self.closed:
            return
        with contextlib.suppress(Exception):
            await self._post(
                "/cancel",
                {"task_id": task_id, "attempt_id": attempt_id, "reason": reason},
                5.0,
            )

    async def close(self) -> None:
        self.closed = True


def _steering_to_json(spec: SteeringSpec) -> dict[str, Any]:
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
        "dtype": boundary.dtype,
        "shape": list(boundary.shape),
        "checksum": boundary.checksum,
        "data_base64": base64.b64encode(boundary.data).decode("ascii"),
    }


def _boundary_from_json(value: Any) -> pb.BoundaryTensor | None:
    if not value:
        return None
    try:
        data = base64.b64decode(value.get("data_base64", ""))
    except Exception:
        data = b""
    return pb.BoundaryTensor(
        dtype=str(value.get("dtype", "")),
        shape=[int(dim) for dim in value.get("shape", [])],
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


def _task_result_from_json(
    body: dict[str, Any], task_id: str, attempt_id: str, device_id: str
) -> TaskResult:
    metrics_body = body.get("metrics") or {}
    metrics = None
    if metrics_body.get("model_id"):
        metrics = ExecutionMetrics(
            model_id=str(metrics_body.get("model_id", "")),
            model_version=str(metrics_body.get("model_version", "")),
            runtime_name=str(metrics_body.get("runtime_name", "")),
            runtime_version=str(metrics_body.get("runtime_version", "")),
            accelerator=str(metrics_body.get("accelerator", "")),
            execution_latency_ms=int(metrics_body.get("execution_latency_ms", 0)),
            error_code=str(metrics_body.get("error_code", "")),
            error_message=str(metrics_body.get("error_message", "")),
            observed_memory_delta_mb=metrics_body.get("observed_memory_delta_mb"),
            observed_thermal_delta=metrics_body.get("observed_thermal_delta"),
        )
    return TaskResult(
        task_id=task_id,
        attempt_id=attempt_id,
        success=bool(body.get("success", False)),
        output_text=str(body.get("output_text", "")),
        device_id=device_id,
        error_code=str(body.get("error_code", "")),
        error_message=str(body.get("error_message", "")),
        latency_ms=(
            metrics.execution_latency_ms
            if metrics
            else int(body.get("latency_ms", 0))
        ),
        metrics=metrics,
    )

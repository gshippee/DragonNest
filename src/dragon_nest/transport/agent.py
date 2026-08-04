from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ssl
import time
from dataclasses import dataclass, replace

import grpc
import numpy as np

from ..artifacts import ArtifactNotFoundError, ArtifactRegistry
from ..executors import ExecutorDispatcher
from ..models import (
    Device,
    ExecutionMode,
    ExecutionPlan,
    PlannedTask,
    RuntimeName,
    TaskResult,
)
from ..proto import dragonnest_pb2 as pb
from ..proto import dragonnest_pb2_grpc as pb_grpc
from ..telemetry import PlatformTelemetry, SystemTelemetry
from .conversion import (
    health_to_proto,
    partial_result_to_proto,
    registration_from_device,
    steering_from_proto,
    task_result_to_proto,
)


class RegistrationRejectedError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentClientConfig:
    brain_target: str = "127.0.0.1:50051"
    enrollment_token: str = "dev-token"
    agent_version: str = "0.1.0"
    heartbeat_interval_seconds: float = 2.0
    reconnect_initial_seconds: float = 0.25
    reconnect_max_seconds: float = 5.0
    execution_delay_seconds: float = 0.0
    tls_ca_path: str = ""
    tls_client_certificate_path: str = ""
    tls_client_key_path: str = ""


class DeviceAgent:
    def __init__(
        self,
        device: Device,
        config: AgentClientConfig | None = None,
        artifacts: ArtifactRegistry | None = None,
        executor: ExecutorDispatcher | None = None,
        telemetry: PlatformTelemetry | None = None,
    ):
        self.config = config or AgentClientConfig()
        self.artifacts = artifacts
        self.device = self._available_device(device)
        self.executor = executor or ExecutorDispatcher(artifacts)
        self.telemetry = telemetry or SystemTelemetry(self.device)
        self.registered = asyncio.Event()
        self._stop = asyncio.Event()
        self._outbound: asyncio.Queue[pb.DeviceToBrain] = asyncio.Queue()
        self._call: grpc.aio.StreamStreamCall | None = None
        self._execution_tasks: dict[str, asyncio.Task[None]] = {}
        self._active_task_ids: dict[str, int] = {}
        self._disconnect_next_task = False
        self._simulated_disconnect_in_progress = False
        self.cancelled_attempt_ids: set[str] = set()
        self._network_changed = asyncio.Event()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._last_heartbeat_sent_at: float | None = None
        self._network_rtt_ms = -1.0

    async def run_forever(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        backoff = self.config.reconnect_initial_seconds
        while not self._stop.is_set():
            self.registered.clear()
            try:
                await self._run_connection()
                backoff = self.config.reconnect_initial_seconds
            except RegistrationRejectedError:
                raise
            except (grpc.aio.AioRpcError, ConnectionError, asyncio.TimeoutError):
                pass
            finally:
                self._call = None
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.config.reconnect_max_seconds)

    async def stop(self, graceful: bool = True) -> None:
        self._stop.set()
        if graceful and self._call is not None:
            await self._outbound.put(
                pb.DeviceToBrain(
                    shutdown=pb.ShutdownEvent(
                        device_id=self.device.device_id,
                        reason="agent_shutdown",
                    )
                )
            )
            await asyncio.sleep(0)
        elif self._call is not None:
            self._call.cancel()
        for task in tuple(self._execution_tasks.values()):
            task.cancel()
        for task in tuple(self._execution_tasks.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def simulate_disconnect_on_next_task(self) -> None:
        self._disconnect_next_task = True

    def notify_network_changed(self) -> None:
        """Request an immediate telemetry refresh after a platform network event."""
        if self._event_loop is not None and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._network_changed.set)
        else:
            self._network_changed.set()

    async def _run_connection(self) -> None:
        options = (
            ("grpc.keepalive_time_ms", 10_000),
            ("grpc.keepalive_timeout_ms", 5_000),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        )
        channel_factory = (
            grpc.aio.secure_channel
            if self.config.tls_ca_path
            else grpc.aio.insecure_channel
        )
        channel_credentials = self._channel_credentials()
        channel_args = (
            (self.config.brain_target, channel_credentials)
            if channel_credentials is not None
            else (self.config.brain_target,)
        )
        async with channel_factory(*channel_args, options=options) as channel:
            stub = pb_grpc.BrainControlStub(channel)
            heartbeat = asyncio.create_task(self._heartbeat_loop())
            try:
                call = stub.Connect(self._request_messages())
                self._call = call
                try:
                    async for message in call:
                        kind = message.WhichOneof("payload")
                        if kind == "registration_accepted":
                            self.registered.set()
                        elif kind == "registration_rejected":
                            raise RegistrationRejectedError(
                                message.registration_rejected.reason
                            )
                        elif kind == "execute_task":
                            if self._disconnect_next_task:
                                self._disconnect_next_task = False
                                self._simulated_disconnect_in_progress = True
                                call.cancel()
                                continue
                            task = asyncio.create_task(
                                self._execute_assignment(message.execute_task)
                            )
                            attempt_id = message.execute_task.attempt_id
                            self._execution_tasks[attempt_id] = task
                            task.add_done_callback(
                                lambda completed, key=attempt_id: (
                                    self._execution_tasks.pop(key, None)
                                )
                            )
                        elif kind == "execute_shard":
                            if self._disconnect_next_task:
                                self._disconnect_next_task = False
                                self._simulated_disconnect_in_progress = True
                                call.cancel()
                                continue
                            task = asyncio.create_task(
                                self._execute_shard(message.execute_shard)
                            )
                            attempt_id = message.execute_shard.attempt_id
                            self._execution_tasks[attempt_id] = task
                            task.add_done_callback(
                                lambda completed, key=attempt_id: (
                                    self._execution_tasks.pop(key, None)
                                )
                            )
                        elif kind == "execute_pipeline_stage":
                            if self._disconnect_next_task:
                                self._disconnect_next_task = False
                                self._simulated_disconnect_in_progress = True
                                call.cancel()
                                continue
                            task = asyncio.create_task(
                                self._execute_pipeline_stage(
                                    message.execute_pipeline_stage
                                )
                            )
                            attempt_id = message.execute_pipeline_stage.attempt_id
                            self._execution_tasks[attempt_id] = task
                            task.add_done_callback(
                                lambda completed, key=attempt_id: (
                                    self._execution_tasks.pop(key, None)
                                )
                            )
                        elif kind == "cancel_task":
                            self._cancel_assignment(message.cancel_task)
                        elif kind == "heartbeat_ack":
                            if self._last_heartbeat_sent_at is not None:
                                self._network_rtt_ms = max(
                                    0.0,
                                    (time.monotonic() - self._last_heartbeat_sent_at)
                                    * 1000,
                                )
                except asyncio.CancelledError:
                    if self._simulated_disconnect_in_progress:
                        self._simulated_disconnect_in_progress = False
                        return
                    raise
            finally:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat

    async def _request_messages(self):
        yield pb.DeviceToBrain(
            register_device=registration_from_device(
                self.device,
                self.config.enrollment_token,
                self.config.agent_version,
                self._certificate_fingerprint(),
            )
        )
        while not self._stop.is_set() or not self._outbound.empty():
            message = await self._outbound.get()
            yield message
            if message.WhichOneof("payload") == "shutdown":
                return

    def _channel_credentials(self) -> grpc.ChannelCredentials | None:
        if not self.config.tls_ca_path:
            return None
        root = _read_bytes(self.config.tls_ca_path)
        certificate = (
            _read_bytes(self.config.tls_client_certificate_path)
            if self.config.tls_client_certificate_path
            else None
        )
        key = (
            _read_bytes(self.config.tls_client_key_path)
            if self.config.tls_client_key_path
            else None
        )
        if (certificate is None) != (key is None):
            raise ValueError("both TLS client certificate and key are required")
        return grpc.ssl_channel_credentials(
            root_certificates=root,
            private_key=key,
            certificate_chain=certificate,
        )

    def _certificate_fingerprint(self) -> str:
        path = self.config.tls_client_certificate_path
        return certificate_fingerprint(_read_bytes(path)) if path else ""

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            self._network_changed.clear()
            snapshot = self.telemetry.sample()
            sampled_health = replace(
                snapshot.health,
                network_rtt_ms=(
                    self._network_rtt_ms
                    if self._network_rtt_ms >= 0
                    else snapshot.health.network_rtt_ms
                ),
            )
            health = health_to_proto(
                replace(self.device, health=sampled_health), int(time.time() * 1000)
            )
            health.active_task_ids.extend(
                sorted({*snapshot.active_task_ids, *self._active_task_ids})
            )
            health.warm_model_ids.extend(snapshot.warm_model_ids)
            health.simulated_constraint = snapshot.simulated_constraint
            self._last_heartbeat_sent_at = time.monotonic()
            await self._outbound.put(pb.DeviceToBrain(health_update=health))
            try:
                await asyncio.wait_for(
                    self._network_changed.wait(),
                    timeout=self.config.heartbeat_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def _execute_assignment(self, command: pb.ExecuteTask) -> None:
        self._start_task(command.task_id)
        try:
            plan = ExecutionPlan(
                task_id=command.task_id,
                execution_mode=ExecutionMode.SINGLE,
                request_text=command.request_text,
                tasks=(
                    PlannedTask(
                        shard_id="shard-1",
                        request_text=command.request_text,
                        selected_device_id=self.device.device_id,
                        selected_model_id=command.model_id,
                    ),
                ),
                steering=steering_from_proto(command.steering),
            )
            try:
                if self.config.execution_delay_seconds:
                    await asyncio.sleep(self.config.execution_delay_seconds)
                result = await asyncio.wait_for(
                    self.executor.execute(plan), timeout=command.timeout_ms / 1000
                )
                result = replace(
                    result,
                    task_id=command.task_id,
                    attempt_id=command.attempt_id,
                    device_id=self.device.device_id,
                )
            except Exception as exc:
                result = TaskResult(
                    task_id=command.task_id,
                    attempt_id=command.attempt_id,
                    success=False,
                    output_text="",
                    device_id=self.device.device_id,
                    error_code="EXECUTION_FAILED",
                    error_message=str(exc),
                )
            await self._outbound.put(
                pb.DeviceToBrain(task_result=task_result_to_proto(result))
            )
        finally:
            self._finish_task(command.task_id)

    async def _execute_shard(self, command: pb.ExecuteShard) -> None:
        self._start_task(command.task_id)
        try:
            plan = ExecutionPlan(
                task_id=command.task_id,
                execution_mode=ExecutionMode.SINGLE,
                request_text=command.request_text,
                tasks=(
                    PlannedTask(
                        shard_id=command.shard_id,
                        request_text=command.request_text,
                        selected_device_id=self.device.device_id,
                        selected_model_id=command.model_id,
                    ),
                ),
                steering=steering_from_proto(command.steering),
            )
            try:
                if self.config.execution_delay_seconds:
                    await asyncio.sleep(self.config.execution_delay_seconds)
                result = await asyncio.wait_for(
                    self.executor.execute(plan), timeout=command.timeout_ms / 1000
                )
                result = replace(
                    result,
                    task_id=command.task_id,
                    attempt_id=command.attempt_id,
                    device_id=self.device.device_id,
                )
            except Exception as exc:
                result = TaskResult(
                    task_id=command.task_id,
                    attempt_id=command.attempt_id,
                    success=False,
                    output_text="",
                    device_id=self.device.device_id,
                    error_code="EXECUTION_FAILED",
                    error_message=str(exc),
                )
            await self._outbound.put(
                pb.DeviceToBrain(
                    partial_task_result=partial_result_to_proto(
                        result, command.shard_id
                    )
                )
            )
        finally:
            self._finish_task(command.task_id)

    async def _execute_pipeline_stage(self, command: pb.ExecutePipelineStage) -> None:
        self._start_task(command.task_id)
        try:
            try:
                result = await asyncio.wait_for(
                    self._run_pipeline_stage(command),
                    timeout=command.timeout_ms / 1000,
                )
            except Exception as exc:
                result = pb.PipelineStageResult(
                    task_id=command.task_id,
                    attempt_id=command.attempt_id,
                    stage_id=command.stage_id,
                    device_id=self.device.device_id,
                    success=False,
                    error_code="PIPELINE_STAGE_FAILED",
                    error_message=str(exc),
                )
            await self._outbound.put(pb.DeviceToBrain(pipeline_stage_result=result))
        finally:
            self._finish_task(command.task_id)

    async def _run_pipeline_stage(
        self, command: pb.ExecutePipelineStage
    ) -> pb.PipelineStageResult:
        artifact = None
        if self.artifacts is not None:
            try:
                artifact = self.artifacts.get(command.model_id)
            except ArtifactNotFoundError:
                artifact = None
        if artifact is not None and artifact.runtime == RuntimeName.QNN:
            if command.steering.enabled:
                raise RuntimeError(
                    "runtime steering inputs are not configured for this QNN stage"
                )
            return await self._run_qnn_pipeline_stage(command, artifact)
        return self._run_mock_pipeline_stage(command)

    async def _run_qnn_pipeline_stage(self, command, artifact):
        qnn = self.executor.qnn
        if qnn is None:
            raise RuntimeError("QNN executor is not configured")
        if command.HasField("input_boundary") and command.input_boundary.data:
            boundary = _array_from_boundary(command.input_boundary)
            split = artifact.split_boundary
            if split is None:
                raise RuntimeError(
                    f"{artifact.model_id} has no split boundary metadata"
                )
            inputs = {split.input_tensor: boundary}
        else:
            if qnn.input_builder is None:
                raise RuntimeError(
                    f"{artifact.model_id} requires a text-to-tensor input adapter"
                )
            inputs = qnn.input_builder(command.request_text, artifact)
        graph = await qnn.execute_graph(
            artifact.model_id, inputs, attempt_id=command.attempt_id
        )
        if not graph.success:
            return pb.PipelineStageResult(
                task_id=command.task_id,
                attempt_id=command.attempt_id,
                stage_id=command.stage_id,
                device_id=self.device.device_id,
                success=False,
                error_code=graph.error_code,
                error_message=graph.error_message,
            )
        metric_holder = TaskResult(
            task_id=command.task_id,
            attempt_id=command.attempt_id,
            success=True,
            output_text="",
            metrics=graph.metrics,
        )
        metrics = task_result_to_proto(metric_holder).metrics
        if command.final_stage:
            if qnn.output_formatter:
                output = qnn.output_formatter(graph.outputs, artifact)
            else:
                output = ", ".join(
                    f"{name}:shape={tuple(value.shape)}"
                    for name, value in graph.outputs.items()
                )
            return pb.PipelineStageResult(
                task_id=command.task_id,
                attempt_id=command.attempt_id,
                stage_id=command.stage_id,
                device_id=self.device.device_id,
                success=True,
                output_text=output,
                metrics=metrics,
            )
        split = artifact.split_boundary
        if split is None or split.output_tensor not in graph.outputs:
            raise RuntimeError(f"{artifact.model_id} did not emit its boundary tensor")
        return pb.PipelineStageResult(
            task_id=command.task_id,
            attempt_id=command.attempt_id,
            stage_id=command.stage_id,
            device_id=self.device.device_id,
            success=True,
            output_boundary=_boundary_from_array(
                split.output_tensor, graph.outputs[split.output_tensor]
            ),
            metrics=metrics,
        )

    def _run_mock_pipeline_stage(
        self, command: pb.ExecutePipelineStage
    ) -> pb.PipelineStageResult:
        metrics = pb.ExecutionMetrics(
            model_id=command.model_id,
            model_version="mock-v1",
            runtime_name=RuntimeName.MOCK.value,
            runtime_version="dragon-nest-0.1.0",
            accelerator="cpu",
            execution_latency_ms=1,
        )
        if command.final_stage:
            boundary_note = (
                command.input_boundary.checksum[:20]
                if command.HasField("input_boundary")
                else "none"
            )
            return pb.PipelineStageResult(
                task_id=command.task_id,
                attempt_id=command.attempt_id,
                stage_id=command.stage_id,
                device_id=self.device.device_id,
                success=True,
                output_text=(
                    f"[Mock remote layer-pipeline result from {self.device.device_id}/"
                    f"{command.model_id}] boundary={boundary_note} "
                    f"steering={_steering_text(command.steering)}"
                ),
                metrics=metrics,
            )
        payload = (f"{command.task_id}:{command.stage_id}:{command.model_id}").encode(
            "utf-8"
        )
        return pb.PipelineStageResult(
            task_id=command.task_id,
            attempt_id=command.attempt_id,
            stage_id=command.stage_id,
            device_id=self.device.device_id,
            success=True,
            output_boundary=_boundary_from_array(
                "hidden", np.frombuffer(payload, dtype=np.uint8)
            ),
            metrics=metrics,
        )

    def _cancel_assignment(self, command: pb.CancelTask) -> None:
        self.cancelled_attempt_ids.add(command.attempt_id)
        task = self._execution_tasks.get(command.attempt_id)
        if task is not None and not task.done():
            task.cancel()

    def _start_task(self, task_id: str) -> None:
        self._active_task_ids[task_id] = self._active_task_ids.get(task_id, 0) + 1

    def _finish_task(self, task_id: str) -> None:
        remaining = self._active_task_ids.get(task_id, 0) - 1
        if remaining > 0:
            self._active_task_ids[task_id] = remaining
        else:
            self._active_task_ids.pop(task_id, None)

    def _available_device(self, device: Device) -> Device:
        if self.artifacts is None:
            return replace(
                device,
                models=tuple(
                    model
                    for model in device.models
                    if model.model_family == "mock" or model.segment is not None
                ),
            )
        models = []
        for model in device.models:
            try:
                self.artifacts.get(model.model_id)
            except ArtifactNotFoundError:
                models.append(model)
                continue
            if self.artifacts.is_available(model.model_id):
                artifact = self.artifacts.get(model.model_id)
                models.append(
                    replace(
                        model,
                        model_version=artifact.model_version,
                        tokenizer_id=artifact.tokenizer_id,
                        precision=artifact.precision,
                        runtime_name=artifact.runtime.value,
                        runtime_version=str(
                            artifact.runtime_options.get("runtime_version", "")
                        ),
                        supported_accelerators=artifact.supported_accelerators,
                        min_memory_mb=artifact.min_memory_mb,
                        max_context_tokens=artifact.max_context_tokens,
                        supports_steering=artifact.supports_steering,
                        supports_data_parallel=artifact.supports_data_parallel,
                        supports_layer_pipeline=artifact.supports_layer_pipeline,
                        boundary_format=(
                            artifact.split_boundary.boundary_format
                            if artifact.split_boundary
                            else model.boundary_format
                        ),
                    )
                )
        return replace(device, models=tuple(models))


def _boundary_from_array(name: str, value: np.ndarray) -> pb.BoundaryTensor:
    array = np.ascontiguousarray(value)
    data = array.tobytes()
    return pb.BoundaryTensor(
        tensor_name=name,
        dtype=str(array.dtype),
        shape=array.shape,
        data=data,
        checksum=f"sha256:{hashlib.sha256(data).hexdigest()}",
    )


def certificate_fingerprint(certificate: bytes) -> str:
    try:
        pem = certificate.decode("ascii")
        der = ssl.PEM_cert_to_DER_cert(pem)
        encoded = der
    except (UnicodeDecodeError, ValueError):
        encoded = certificate
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _array_from_boundary(boundary: pb.BoundaryTensor) -> np.ndarray:
    expected = f"sha256:{hashlib.sha256(boundary.data).hexdigest()}"
    if boundary.checksum != expected:
        raise ValueError("boundary tensor checksum mismatch")
    array = np.frombuffer(boundary.data, dtype=np.dtype(boundary.dtype))
    expected_values = int(np.prod(boundary.shape))
    if array.size != expected_values:
        raise ValueError("boundary tensor shape does not match payload")
    return array.reshape(tuple(boundary.shape)).copy()


def _steering_text(spec: pb.SteeringSpec) -> str:
    if not spec.enabled:
        return "disabled"
    return (
        f"{spec.vector_id}:alpha={spec.alpha}:layer={spec.target_layer}:"
        f"positions={spec.positions}"
    )

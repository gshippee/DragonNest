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
    ModelSegment,
    PlannedTask,
    RuntimeName,
    TaskResult,
)
from ..proto import dragonnest_pb2 as pb
from ..proto import dragonnest_pb2_grpc as pb_grpc
from ..pipeline_sessions import PipelineSessionKey, PipelineSessionStore
from ..runtime.qwen17_provider import PIPELINE_ID as QWEN17_PIPELINE_ID
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
        pipeline_provider=None,
    ):
        self.config = config or AgentClientConfig()
        self.artifacts = artifacts
        self.executor = executor or ExecutorDispatcher(artifacts)
        self.device = self._available_device(device)
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
        self.pipeline_sessions = PipelineSessionStore()
        self.pipeline_provider = pipeline_provider
        self._network_changed = asyncio.Event()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._last_heartbeat_sent_at: float | None = None
        self._network_rtt_ms = -1.0
        self._enrollment_token = self.config.enrollment_token

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
        self.pipeline_sessions.clear()
        if self.pipeline_provider is not None:
            self.pipeline_provider.clear()

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
                            if message.registration_accepted.device_credential:
                                self._enrollment_token = (
                                    message.registration_accepted.device_credential
                                )
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
                # A disconnected Brain cannot complete/reset an in-flight
                # generation. Never carry stage-local KV into a new stream.
                self.pipeline_sessions.clear()
                if self.pipeline_provider is not None:
                    self.pipeline_provider.clear()

    async def _request_messages(self):
        yield pb.DeviceToBrain(
            register_device=registration_from_device(
                self.device,
                self._enrollment_token,
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
                self.pipeline_sessions.cleanup_task(command.task_id)
                if self.pipeline_provider is not None:
                    self.pipeline_provider.cleanup_task(command.task_id)
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
        if command.operation in {pb.PIPELINE_RESET, pb.PIPELINE_CANCEL}:
            if self.pipeline_provider is not None:
                self.pipeline_provider.release(
                    command.task_id,
                    command.pipeline_id,
                    command.stage_index,
                    cancelled=command.operation == pb.PIPELINE_CANCEL,
                )
            if command.pipeline_id:
                self.pipeline_sessions.release(
                    PipelineSessionKey(
                        command.task_id, command.pipeline_id, command.stage_index
                    )
                )
            else:
                self.pipeline_sessions.cleanup_task(command.task_id)
            return pb.PipelineStageResult(
                task_id=command.task_id,
                attempt_id=command.attempt_id,
                stage_id=command.stage_id,
                device_id=self.device.device_id,
                success=True,
                operation=command.operation,
            )
        artifact = None
        if self.artifacts is not None:
            try:
                artifact = self.artifacts.get(command.model_id)
            except ArtifactNotFoundError:
                artifact = None
        if artifact is None:
            # Plain ExecutorDispatcher agents are the explicit control-plane
            # simulation used by tests/demo scenarios. HardwareRuntimeAdapter
            # exposes capabilities(); on that path missing bytes are fatal.
            if self.artifacts is None or not hasattr(self.executor, "capabilities"):
                return self._run_mock_pipeline_stage(command)
            raise RuntimeError(
                f"physical pipeline artifact {command.model_id!r} is unavailable; "
                "mock fallback is forbidden"
            )
        if artifact.runtime == RuntimeName.QNN:
            if command.steering.enabled:
                raise RuntimeError(
                    "runtime steering inputs are not configured for this QNN stage"
                )
            split = artifact.split_boundary
            if split is not None and split.pipeline_id == QWEN17_PIPELINE_ID:
                if self.pipeline_provider is None:
                    raise RuntimeError(
                        "Qwen3-1.7B physical provider is not configured; "
                        "refusing generic replacement-KV or mock execution"
                    )
                return await self._run_qwen17_pipeline_stage(command, artifact)
            return await self._run_qnn_pipeline_stage(command, artifact)
        raise RuntimeError(
            f"pipeline artifact {artifact.model_id} uses unsupported runtime "
            f"{artifact.runtime.value}; mock fallback is forbidden"
        )

    async def _run_qwen17_pipeline_stage(self, command, artifact):
        boundary = (
            _array_from_boundary(command.input_boundary)
            if command.HasField("input_boundary") and command.input_boundary.data
            else None
        )
        physical = await asyncio.to_thread(
            self.pipeline_provider.execute, command, artifact, boundary
        )
        metrics = pb.ExecutionMetrics(
            model_id=artifact.model_id,
            model_version=artifact.model_version,
            runtime_name=RuntimeName.QNN.value,
            runtime_version=str(
                artifact.runtime_options.get("runtime_version", "QAIRT-2.45")
            ),
            accelerator="htp",
            execution_latency_ms=physical.latency_ms,
        )
        result = pb.PipelineStageResult(
            task_id=command.task_id,
            attempt_id=command.attempt_id,
            stage_id=command.stage_id,
            device_id=self.device.device_id,
            success=True,
            metrics=metrics,
            operation=command.operation,
            eos=physical.eos,
            token_text=physical.token_text,
        )
        if physical.boundary is not None:
            result.output_boundary.CopyFrom(
                _boundary_from_array(
                    artifact.split_boundary.output_tensor, physical.boundary
                )
            )
        if physical.next_token_id is not None:
            result.next_token_id = physical.next_token_id
        return result

    async def _run_qnn_pipeline_stage(self, command, artifact):
        qnn = self.executor.qnn
        if qnn is None:
            raise RuntimeError("QNN executor is not configured")
        split = artifact.split_boundary
        if split is None:
            raise RuntimeError(f"{artifact.model_id} has no split boundary metadata")
        pipeline_id = command.pipeline_id or split.pipeline_id
        key = PipelineSessionKey(command.task_id, pipeline_id, command.stage_index)
        session = None
        if command.operation == pb.PIPELINE_PREFILL:
            session = self.pipeline_sessions.begin_prefill(key)
        elif command.operation == pb.PIPELINE_DECODE:
            session = self.pipeline_sessions.require_decode(key)

        if command.HasField("input_boundary") and command.input_boundary.data:
            boundary = _array_from_boundary(command.input_boundary)
            inputs = {split.input_tensor: boundary}
        elif command.operation == pb.PIPELINE_DECODE and split.includes_embedding:
            inputs = {split.input_tensor: np.asarray([[command.token_id]], dtype=np.int32)}
        else:
            if qnn.input_builder is None:
                raise RuntimeError(
                    f"{artifact.model_id} requires a text-to-tensor input adapter"
                )
            inputs = qnn.input_builder(command.request_text, artifact)
        if session is not None:
            inputs.update(session.kv_inputs)
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
                operation=command.operation,
            )
        if session is not None:
            self.pipeline_sessions.retain_outputs(key, graph.outputs)
            if command.operation == pb.PIPELINE_PREFILL:
                self.pipeline_sessions.complete_prefill(key)
        metric_holder = TaskResult(
            task_id=command.task_id,
            attempt_id=command.attempt_id,
            success=True,
            output_text="",
            metrics=graph.metrics,
        )
        metrics = task_result_to_proto(metric_holder).metrics
        if command.final_stage:
            if command.operation in {pb.PIPELINE_PREFILL, pb.PIPELINE_DECODE}:
                if split.output_tensor not in graph.outputs:
                    raise RuntimeError(
                        f"{artifact.model_id} did not emit {split.output_tensor}"
                    )
                logits = np.asarray(graph.outputs[split.output_tensor])
                if logits.size == 0:
                    raise RuntimeError(f"{artifact.model_id} emitted empty logits")
                next_token_id = int(np.argmax(logits.reshape(-1, logits.shape[-1])[-1]))
                eos_ids = {
                    int(value)
                    for value in artifact.runtime_options.get("eos_token_ids", ())
                }
                token_decoder = getattr(qnn, "token_decoder", None)
                token_text = (
                    str(token_decoder(next_token_id, artifact))
                    if token_decoder is not None
                    else ""
                )
                return pb.PipelineStageResult(
                    task_id=command.task_id,
                    attempt_id=command.attempt_id,
                    stage_id=command.stage_id,
                    device_id=self.device.device_id,
                    success=True,
                    metrics=metrics,
                    next_token_id=next_token_id,
                    eos=next_token_id in eos_ids,
                    token_text=token_text,
                    operation=command.operation,
                )
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
                operation=command.operation,
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
            operation=command.operation,
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
        pipeline_id = command.pipeline_id or "legacy-mock-pipeline"
        key = PipelineSessionKey(command.task_id, pipeline_id, command.stage_index)
        if command.operation == pb.PIPELINE_PREFILL:
            self.pipeline_sessions.begin_prefill(key)
            self.pipeline_sessions.complete_prefill(key)
        elif command.operation == pb.PIPELINE_DECODE:
            self.pipeline_sessions.require_decode(key)
        if command.final_stage:
            if command.operation in {pb.PIPELINE_PREFILL, pb.PIPELINE_DECODE}:
                return pb.PipelineStageResult(
                    task_id=command.task_id,
                    attempt_id=command.attempt_id,
                    stage_id=command.stage_id,
                    device_id=self.device.device_id,
                    success=True,
                    next_token_id=1,
                    eos=(command.operation == pb.PIPELINE_DECODE),
                    token_text="x",
                    operation=command.operation,
                    metrics=metrics,
                )
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
                operation=command.operation,
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
            operation=command.operation,
        )

    def _cancel_assignment(self, command: pb.CancelTask) -> None:
        self.cancelled_attempt_ids.add(command.attempt_id)
        self.pipeline_sessions.cleanup_task(command.task_id)
        if self.pipeline_provider is not None:
            self.pipeline_provider.cleanup_task(command.task_id, cancelled=True)
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
        adapter_capabilities = (
            self.executor.capabilities()
            if hasattr(self.executor, "capabilities")
            else None
        )
        installed_ids = (
            set(adapter_capabilities.installed_artifact_ids)
            if adapter_capabilities is not None
            else None
        )
        warm_ids = (
            set(adapter_capabilities.warm_artifact_ids)
            if adapter_capabilities is not None
            else set()
        )
        for model in device.models:
            try:
                self.artifacts.get(model.model_id)
            except ArtifactNotFoundError:
                models.append(model)
                continue
            if self.artifacts.is_available(model.model_id):
                artifact = self.artifacts.get(model.model_id)
                if installed_ids is not None and artifact.artifact_id not in installed_ids:
                    continue
                models.append(
                    replace(
                        model,
                        warm=(
                            artifact.artifact_id in warm_ids
                            if installed_ids is not None
                            else model.warm
                        ),
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
                        artifact_id=artifact.artifact_id,
                        steering_modes=(artifact.steering_mode.value,),
                        behavior_profile_ids=(
                            (artifact.behavior_profile_id,)
                            if artifact.behavior_profile_id
                            else ()
                        ),
                        target_compatibility_class=(
                            artifact.target_compatibility_class
                        ),
                        boundary_format=(
                            artifact.split_boundary.boundary_format
                            if artifact.split_boundary
                            else model.boundary_format
                        ),
                        segment=(
                            ModelSegment(
                                pipeline_id=artifact.split_boundary.pipeline_id,
                                start_layer=artifact.split_boundary.start_layer,
                                end_layer=artifact.split_boundary.end_layer,
                                total_layers=artifact.split_boundary.total_layers,
                                includes_embedding=(
                                    artifact.split_boundary.includes_embedding
                                ),
                                includes_lm_head=(
                                    artifact.split_boundary.includes_lm_head
                                ),
                                stage_index=artifact.split_boundary.stage_index,
                                stage_count=artifact.split_boundary.stage_count,
                                transformer_start_layer=(
                                    artifact.split_boundary.transformer_start_layer
                                ),
                                transformer_end_layer=(
                                    artifact.split_boundary.transformer_end_layer
                                ),
                                input_tensor=artifact.split_boundary.input_tensor,
                                output_tensor=artifact.split_boundary.output_tensor,
                                boundary_format=(
                                    artifact.split_boundary.boundary_format
                                ),
                            )
                            if artifact.split_boundary
                            else model.segment
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

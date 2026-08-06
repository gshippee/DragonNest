from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import ssl
import time
from dataclasses import dataclass, field, replace
from typing import AsyncIterator
from urllib.parse import urlsplit

import grpc
import httpx

from ..classifier import RuleBasedTaskClassifier
from ..dispatch import DeviceOfflineError, DispatchManager
from ..endpoints import EndpointError, HttpEndpoint, HttpEndpointStore
from ..enrollment import EnrollmentError, EnrollmentManager
from ..models import (
    Device,
    ExecutionMetrics,
    ExecutionMode,
    ExecutionPlan,
    HealthStatus,
    PlannedTask,
    PipelineStage,
    ReducerMode,
    RuntimeName,
    SteeringSpec,
    TaskProfile,
    TaskResult,
)
from ..planner import ExecutionPlanner
from ..profiles import PersonalProfile, ProfileError, ProfileStore
from ..proto import dragonnest_pb2 as pb
from ..proto import dragonnest_pb2_grpc as pb_grpc
from ..registry import DeviceRecord, DeviceRegistry
from ..router import DeterministicRouter
from ..steering import SteeringRegistry
from ..tasks import AttemptState, TaskRecord, TaskStore
from .http_device import HttpDeviceSession, fetch_endpoint_info
from .sessions import DeviceSession, SessionConflictError, SessionRegistry
from .conversion import (
    device_from_registration,
    health_from_proto,
    partial_result_from_proto,
    pipeline_result_from_proto,
    steering_from_proto,
    steering_to_proto,
    task_result_from_proto,
)


PERSONA_STEERING: dict[str, float | None] = {
    "balanced": None,
    "concise": -2.0,
    "detailed": 2.0,
}
PERSONA_VECTOR_ID = "concise-vs-verbose-layer-7"


@dataclass(frozen=True)
class BrainServiceConfig:
    brain_id: str = "dragon-nest-brain"
    enrollment_token: str = "dev-token"
    heartbeat_interval_ms: int = 2000
    default_task_timeout_ms: int = 30_000
    sweep_interval_seconds: float = 1.0
    max_boundary_bytes: int = 32 * 1024 * 1024
    dev_mode: bool = True
    tls_server_certificate_path: str = ""
    tls_server_key_path: str = ""
    tls_client_ca_path: str = ""
    require_client_certificate: bool = True
    enrollment_session_ttl_seconds: int = 300
    state_db_path: str = ":memory:"
    http_endpoint_registration_enabled: bool = False
    http_endpoint_admin_token: str = field(default="", repr=False)
    http_endpoint_allowed_cidrs: tuple[str, ...] = ("127.0.0.0/8", "::1/128")
    http_endpoint_allowed_hosts: tuple[str, ...] = ("localhost",)


@dataclass(frozen=True)
class PipelineAttemptResult:
    success: bool
    output_text: str
    device_id: str
    boundary: pb.BoundaryTensor | None = None
    error_code: str = ""
    error_message: str = ""
    metrics: ExecutionMetrics | None = None


class AgentSession:
    transport = "grpc_stream"
    allow_profile_context = True

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.outbound: asyncio.Queue[pb.BrainToDevice | None] = asyncio.Queue()
        self.pending: dict[
            str,
            tuple[
                str,
                asyncio.Future[
                    pb.TaskResult | pb.PartialTaskResult | pb.PipelineStageResult
                ],
            ],
        ] = {}
        self.closed = False
        self.graceful = False

    async def execute(
        self, command: pb.ExecuteTask, timeout_seconds: float
    ) -> pb.TaskResult:
        result = await self._dispatch(
            command.task_id,
            command.attempt_id,
            pb.BrainToDevice(execute_task=command),
            timeout_seconds,
        )
        if not isinstance(result, pb.TaskResult):
            raise RuntimeError("agent returned wrong result type for task")
        return result

    async def execute_shard(
        self, command: pb.ExecuteShard, timeout_seconds: float
    ) -> pb.PartialTaskResult:
        result = await self._dispatch(
            command.task_id,
            command.attempt_id,
            pb.BrainToDevice(execute_shard=command),
            timeout_seconds,
        )
        if not isinstance(result, pb.PartialTaskResult):
            raise RuntimeError("agent returned wrong result type for shard")
        return result

    async def execute_pipeline_stage(
        self, command: pb.ExecutePipelineStage, timeout_seconds: float
    ) -> pb.PipelineStageResult:
        result = await self._dispatch(
            command.task_id,
            command.attempt_id,
            pb.BrainToDevice(execute_pipeline_stage=command),
            timeout_seconds,
        )
        if not isinstance(result, pb.PipelineStageResult):
            raise RuntimeError("agent returned wrong result type for pipeline stage")
        return result

    async def cancel(self, task_id: str, attempt_id: str, reason: str) -> None:
        if self.closed:
            return
        await self.outbound.put(
            pb.BrainToDevice(
                cancel_task=pb.CancelTask(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    reason=reason,
                )
            )
        )

    async def _dispatch(
        self,
        task_id: str,
        attempt_id: str,
        message: pb.BrainToDevice,
        timeout_seconds: float,
    ) -> pb.TaskResult | pb.PartialTaskResult | pb.PipelineStageResult:
        if self.closed:
            raise DeviceOfflineError(f"device {self.device_id} stream is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[
            pb.TaskResult | pb.PartialTaskResult | pb.PipelineStageResult
        ] = loop.create_future()
        self.pending[attempt_id] = (task_id, future)
        await self.outbound.put(message)
        try:
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        finally:
            self.pending.pop(attempt_id, None)

    def resolve(
        self, result: pb.TaskResult | pb.PartialTaskResult | pb.PipelineStageResult
    ) -> bool:
        pending = self.pending.get(result.attempt_id)
        if pending is None:
            return False
        task_id, future = pending
        if task_id != result.task_id or future.done():
            return False
        future.set_result(result)
        return True

    async def close(self, graceful: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        self.graceful = graceful
        error = DeviceOfflineError(f"device {self.device_id} stream closed")
        for _, future in tuple(self.pending.values()):
            if not future.done():
                future.set_exception(error)
        await self.outbound.put(None)


class BrainService(pb_grpc.BrainControlServicer):
    def __init__(
        self,
        config: BrainServiceConfig | None = None,
        registry: DeviceRegistry | None = None,
        tasks: TaskStore | None = None,
        steering_registry: SteeringRegistry | None = None,
    ):
        self.config = config or BrainServiceConfig()
        self.registry = registry or DeviceRegistry()
        self.tasks = tasks or TaskStore()
        self.dispatch = DispatchManager(self.registry, self.tasks)
        self.sessions = SessionRegistry()
        self.classifier = RuleBasedTaskClassifier()
        self.planner = ExecutionPlanner()
        self.steering_registry = steering_registry
        self.router = DeterministicRouter(steering_registry)
        self._route_reasons: dict[str, tuple[str, ...]] = {}
        self._attempt_models: dict[str, str] = {}
        self._steering_specs: dict[str, SteeringSpec] = {}
        self.task_profiles: dict[str, TaskProfile] = {}
        self.execution_plans: dict[str, ExecutionPlan] = {}
        self._device_simulations: dict[str, dict[str, float | bool | int]] = {}
        self.certificate_fingerprints: dict[str, str] = {}
        self.revoked_certificate_fingerprints: set[str] = set()
        self.enrollment = EnrollmentManager(
            default_ttl_seconds=self.config.enrollment_session_ttl_seconds
        )
        self.profiles = ProfileStore(self.config.state_db_path)
        self.endpoints = HttpEndpointStore(self.config.state_db_path)
        self._sweeper: asyncio.Task[None] | None = None
        self._http_client = httpx.AsyncClient()
        self._http_poll_tasks: dict[str, asyncio.Task[None]] = {}

    def set_device_simulation(
        self, device_id: str, changes: dict[str, float | bool | int]
    ) -> None:
        if changes:
            self._device_simulations[device_id] = changes
        else:
            self._device_simulations.pop(device_id, None)

    async def start(self) -> None:
        if self._sweeper is not None:
            return
        try:
            if self.config.http_endpoint_registration_enabled:
                for endpoint in self.endpoints.all():
                    await self.register_http_device(endpoint, persist=False)
        except Exception:
            for device_id in tuple(self._http_poll_tasks):
                await self._cancel_http_poll(device_id)
            await self.sessions.close_all()
            raise
        self._sweeper = asyncio.create_task(self._sweep_loop())

    async def stop(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None
        for device_id in tuple(self._http_poll_tasks):
            await self._cancel_http_poll(device_id)
        await self.sessions.close_all()
        await self._http_client.aclose()

    async def register_http_device(
        self, endpoint: HttpEndpoint, *, persist: bool = True
    ) -> DeviceRecord:
        if not self.config.http_endpoint_registration_enabled:
            raise EndpointError("HTTP endpoint registration is disabled")
        normalized_url = await self.validate_http_endpoint_url(endpoint.base_url)
        endpoint = replace(endpoint, base_url=normalized_url)
        existing = await self.sessions.get(endpoint.device.device_id)
        if existing is not None and existing.transport != "http_endpoint":
            raise SessionConflictError(
                f"device {endpoint.device.device_id!r} is already connected via "
                f"{existing.transport}"
            )
        await self._cancel_http_poll(endpoint.device.device_id)
        session = HttpDeviceSession(endpoint, self._http_client)
        await self.sessions.register(session, replace_same_transport=True)
        record = self.registry.register(endpoint.device)
        if persist:
            self.endpoints.put(endpoint)
        self._http_poll_tasks[endpoint.device.device_id] = asyncio.create_task(
            self._poll_http_device(session)
        )
        return record

    async def deregister_http_device(self, device_id: str) -> None:
        session = await self.sessions.get(device_id)
        if session is None or session.transport != "http_endpoint":
            raise EndpointError("HTTP endpoint not found")
        await self._cancel_http_poll(device_id)
        await self.sessions.close_device(device_id)
        self.endpoints.delete(device_id)
        with contextlib.suppress(KeyError):
            self.dispatch.handle_device_offline(device_id)
            self.registry.deregister(device_id)

    async def fetch_http_endpoint_info(
        self, base_url: str, credential_env: str = ""
    ) -> dict[str, object]:
        normalized_url = await self.validate_http_endpoint_url(base_url)
        return await fetch_endpoint_info(
            self._http_client,
            normalized_url,
            credential_env,
        )

    async def validate_http_endpoint_url(self, base_url: str) -> str:
        parsed = urlsplit(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise EndpointError("base_url must be an absolute http(s) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise EndpointError(
                "base_url cannot contain credentials, a query, or a fragment"
            )
        try:
            parsed.port
        except ValueError as exc:
            raise EndpointError("base_url port is invalid") from exc
        hostname = parsed.hostname.rstrip(".").lower()
        allowed_hosts = {
            value.rstrip(".").lower()
            for value in self.config.http_endpoint_allowed_hosts
        }
        if hostname not in allowed_hosts:
            try:
                address = ipaddress.ip_address(hostname)
            except ValueError as exc:
                raise EndpointError(
                    "base_url hostname is not explicitly allowed"
                ) from exc
            try:
                networks = tuple(
                    ipaddress.ip_network(value, strict=False)
                    for value in self.config.http_endpoint_allowed_cidrs
                )
            except ValueError as exc:
                raise EndpointError("configured endpoint CIDR is invalid") from exc
            if not any(address in network for network in networks):
                raise EndpointError("base_url is outside the endpoint allowlist")
        return base_url.strip().rstrip("/")

    async def _cancel_http_poll(self, device_id: str) -> None:
        poll_task = self._http_poll_tasks.pop(device_id, None)
        if poll_task is not None:
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task

    async def _poll_http_device(self, session: HttpDeviceSession) -> None:
        device_id = session.device_id
        while True:
            current = await self.sessions.get(device_id)
            if current is not session or session.closed:
                return
            await asyncio.sleep(session.endpoint.poll_interval_seconds)
            current = await self.sessions.get(device_id)
            if current is not session or session.closed:
                return
            try:
                health = await session.fetch_health()
            except DeviceOfflineError:
                continue
            with contextlib.suppress(KeyError):
                self.registry.heartbeat(device_id, health)

    async def _resolve_session(self, device_id: str) -> DeviceSession:
        session = await self.sessions.get(device_id)
        if session is not None and not session.closed:
            return session
        raise DeviceOfflineError(f"device {device_id} is disconnected")

    async def Connect(
        self,
        request_iterator: AsyncIterator[pb.DeviceToBrain],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.BrainToDevice]:
        try:
            first = await anext(request_iterator)
        except StopAsyncIteration:
            return
        if first.WhichOneof("payload") != "register_device":
            yield pb.BrainToDevice(
                registration_rejected=pb.RegistrationRejected(
                    reason="first stream message must be RegisterDevice"
                )
            )
            return
        registration = first.register_device
        peer_fingerprint = self._peer_certificate_fingerprint(context)
        rejection, issued_credential = self._authorize_registration(
            registration, peer_fingerprint
        )
        if rejection:
            yield pb.BrainToDevice(
                registration_rejected=pb.RegistrationRejected(reason=rejection)
            )
            return
        existing_session = await self.sessions.get(registration.device_id)
        if (
            existing_session is not None
            and existing_session.transport != AgentSession.transport
        ):
            yield pb.BrainToDevice(
                registration_rejected=pb.RegistrationRejected(
                    reason=(
                        f"device ID is already registered via "
                        f"{existing_session.transport}"
                    )
                )
            )
            return

        device = device_from_registration(registration)
        association = self.profiles.association_for_device(device.device_id)
        if association:
            device = replace(device, display_name=association.device_name)
        session = AgentSession(device.device_id)
        await self.sessions.register(session, replace_same_transport=True)
        self.registry.register(device)
        if peer_fingerprint:
            self.certificate_fingerprints[device.device_id] = peer_fingerprint
        yield pb.BrainToDevice(
            registration_accepted=pb.RegistrationAccepted(
                brain_id=self.config.brain_id,
                heartbeat_interval_ms=self.config.heartbeat_interval_ms,
                device_credential=issued_credential,
            )
        )

        consumer = asyncio.create_task(self._consume_agent(session, request_iterator))
        try:
            while True:
                outbound = asyncio.create_task(session.outbound.get())
                done, _ = await asyncio.wait(
                    {outbound, consumer}, return_when=asyncio.FIRST_COMPLETED
                )
                if consumer in done:
                    outbound.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await outbound
                    break
                message = outbound.result()
                if message is None:
                    break
                yield message
        finally:
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer
            removed = await self.sessions.remove(session)
            if removed:
                if session.graceful:
                    self.registry.mark_offline(
                        session.device_id, reason="graceful_shutdown"
                    )
                else:
                    self.registry.stream_closed(session.device_id, unexpected=True)
                    self.dispatch.handle_device_offline(session.device_id)
            await session.close(graceful=session.graceful)

    async def SubmitTask(
        self,
        request: pb.SubmitTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.SubmitTaskResponse:
        if not request.request_text.strip():
            return pb.SubmitTaskResponse(
                state="FAILED",
                error_code="INVALID_REQUEST",
                error_message="request_text cannot be empty",
            )
        personal_profile = (
            self.profiles.profile_for_device(request.origin_device_id)
            if request.origin_device_id
            else None
        )
        preferred_mode = request.preferred_mode or "auto"
        if (
            preferred_mode == "auto"
            and personal_profile is not None
            and personal_profile.preferred_mode != "auto"
        ):
            preferred_mode = personal_profile.preferred_mode
        execution_mode = request.execution_mode or "auto"
        reducer = request.reducer or ReducerMode.MOCK_SYNTHESIS.value
        supported_reducers = {item.value for item in ReducerMode}
        if reducer not in supported_reducers:
            return pb.SubmitTaskResponse(
                state="FAILED",
                error_code="INVALID_REDUCER",
                error_message=(
                    f"unsupported reducer {reducer!r}; choose "
                    f"{', '.join(sorted(supported_reducers))}"
                ),
                origin_device_id=request.origin_device_id,
                reducer=reducer,
            )

        eligible_ids = {
            device.device_id for device in self.registry.eligible()
        }
        exclusion_reasons = tuple(
            self._device_exclusion_reason(record)
            for record in self.registry.records()
            if record.device.device_id not in eligible_ids
        )
        exclusion_reasons = tuple(reason for reason in exclusion_reasons if reason)
        devices = list(self.registry.eligible())
        if not devices:
            return pb.SubmitTaskResponse(
                state="FAILED",
                error_code="NO_ELIGIBLE_FALLBACK",
                error_message="no eligible devices are connected",
                origin_device_id=request.origin_device_id,
                reducer=reducer,
            )
        private_reason = ""
        if preferred_mode == "private":
            if not request.origin_device_id:
                return pb.SubmitTaskResponse(
                    state="FAILED",
                    error_code="ORIGIN_DEVICE_REQUIRED",
                    error_message="private mode requires origin_device_id",
                    reducer=reducer,
                )
            devices = [
                device
                for device in devices
                if device.device_id == request.origin_device_id
            ]
            if not devices:
                return pb.SubmitTaskResponse(
                    state="FAILED",
                    error_code="NO_ELIGIBLE_FALLBACK",
                    error_message=(
                        f"private origin device {request.origin_device_id!r} is not eligible"
                    ),
                    origin_device_id=request.origin_device_id,
                    reducer=reducer,
                )
            private_reason = (
                f"Private mode restricted routing to origin {request.origin_device_id}; "
                "remote devices were excluded by policy."
            )
        steering = (
            steering_from_proto(request.steering)
            if request.HasField("steering") and request.steering.enabled
            else SteeringSpec()
        )
        use_profile_steering = (
            request.use_profile_steering
            if request.HasField("use_profile_steering")
            else True
        )
        persona_id = request.persona_id or (
            personal_profile.persona_id if personal_profile is not None else "balanced"
        )
        if persona_id not in PERSONA_STEERING:
            return pb.SubmitTaskResponse(
                state="FAILED",
                error_code="INVALID_PERSONA",
                error_message=f"unsupported persona {persona_id!r}",
            )
        profile_steering_reason = ""
        if (
            not steering.enabled
            and use_profile_steering
            and not request.persona_id
            and personal_profile is not None
            and personal_profile.steering_vector_id
        ):
            steering = self._steering_for_profile(personal_profile)
            profile_steering_reason = (
                f"Applied personal profile {personal_profile.person_name!r}: "
                f"steering {steering.vector_id} at alpha {steering.alpha}."
            )
        elif not steering.enabled and use_profile_steering:
            try:
                steering = self._steering_for_persona(persona_id)
            except ProfileError as exc:
                return pb.SubmitTaskResponse(
                    state="FAILED",
                    error_code="STEERING_UNAVAILABLE",
                    error_message=str(exc),
                    origin_device_id=request.origin_device_id,
                    reducer=reducer,
                )
            if steering.enabled:
                profile_steering_reason = (
                    f"Applied persona {persona_id!r}: steering "
                    f"{steering.vector_id} at alpha {steering.alpha}."
                )
        if steering.enabled and self.steering_registry is None:
            return pb.SubmitTaskResponse(
                state="FAILED",
                error_code="STEERING_UNAVAILABLE",
                error_message="Brain has no steering registry configured",
            )
        profile = self.classifier.classify(request.request_text, preferred_mode)
        plan = self.planner.plan(
            request.request_text,
            profile,
            preferred_mode=preferred_mode,
            requested_execution_mode=execution_mode,
            steering=steering,
            origin_device_id=request.origin_device_id,
            reducer=reducer,
        )
        if reducer == ReducerMode.FIRST_SUCCESS and plan.execution_mode != ExecutionMode.DATA_PARALLEL:
            return pb.SubmitTaskResponse(
                task_id=plan.task_id,
                state="FAILED",
                error_code="INVALID_REDUCER",
                error_message="first_success requires execution_mode=data_parallel",
                origin_device_id=request.origin_device_id,
                reducer=reducer,
            )
        try:
            routed, decision = self.router.route(plan, profile, devices)
        except ValueError as exc:
            return pb.SubmitTaskResponse(
                task_id=plan.task_id,
                state="FAILED",
                error_code="NO_ELIGIBLE_FALLBACK",
                error_message=str(exc),
                origin_device_id=request.origin_device_id,
                reducer=reducer,
            )

        if private_reason:
            routed = replace(routed, reasons=(*routed.reasons, private_reason))
            decision = replace(decision, reasons=(*decision.reasons, private_reason))
        if exclusion_reasons:
            routed = replace(routed, reasons=(*routed.reasons, *exclusion_reasons))
            decision = replace(
                decision, reasons=(*decision.reasons, *exclusion_reasons)
            )
        if profile_steering_reason:
            routed = replace(
                routed, reasons=(*routed.reasons, profile_steering_reason)
            )
            decision = replace(
                decision, reasons=(*decision.reasons, profile_steering_reason)
            )

        self._route_reasons[plan.task_id] = decision.reasons
        self._steering_specs[plan.task_id] = steering
        self.task_profiles[plan.task_id] = profile
        self.execution_plans[plan.task_id] = routed
        timeout_ms = request.timeout_ms or self.config.default_task_timeout_ms
        use_profile_context = (
            request.use_profile_context
            if request.HasField("use_profile_context")
            else True
        )
        profile_context = (
            personal_profile.notes
            if use_profile_context and personal_profile is not None
            else ""
        )
        if plan.execution_mode == ExecutionMode.DATA_PARALLEL:
            return await self._submit_data_parallel(
                request,
                routed,
                profile,
                timeout_ms,
                profile_context,
            )
        if plan.execution_mode == ExecutionMode.LAYER_PIPELINE:
            return await self._submit_layer_pipeline(
                request,
                routed,
                timeout_ms,
                profile_context,
            )
        candidate_ids = (
            decision.selected_device_id,
            *decision.fallback_device_ids,
        )
        model_by_device = {decision.selected_device_id: decision.selected_model_id}

        async def execute_attempt(
            task_id: str,
            attempt_id: str,
            device: Device,
        ) -> TaskResult:
            session = await self._resolve_session(device.device_id)
            model_id = model_by_device.get(device.device_id)
            if model_id is None:
                fallback_plan, fallback_decision = self.router.route(
                    plan, profile, [device]
                )
                model_id = fallback_decision.selected_model_id
                model_by_device[device.device_id] = model_id
                del fallback_plan
            self._attempt_models[attempt_id] = model_id
            result = await session.execute(
                pb.ExecuteTask(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    request_text=self._request_text_for_session(
                        session, request.request_text, profile_context
                    ),
                    model_id=model_id,
                    timeout_ms=timeout_ms,
                    steering=steering_to_proto(routed.steering),
                ),
                timeout_seconds=timeout_ms / 1000,
            )
            return task_result_from_proto(result)

        dispatched = await self.dispatch.submit(
            request.request_text,
            candidate_ids,
            execute_attempt,
            task_id=routed.task_id,
        )
        return self._response_for_task(dispatched.task)

    async def _submit_data_parallel(
        self,
        request: pb.SubmitTaskRequest,
        routed: ExecutionPlan,
        profile: TaskProfile,
        timeout_ms: int,
        profile_context: str = "",
    ) -> pb.SubmitTaskResponse:
        parent = self.tasks.create(request.request_text, task_id=routed.task_id)
        if routed.reducer == ReducerMode.FIRST_SUCCESS:
            return await self._submit_replica_race(
                request,
                parent,
                routed,
                profile,
                timeout_ms,
                profile_context,
            )

        async def run_shard(shard: PlannedTask):
            child_task_id = f"{parent.task_id}:{shard.shard_id}"
            candidates = (shard.selected_device_id, *shard.fallback_device_ids)
            model_by_device = {shard.selected_device_id: shard.selected_model_id}

            async def execute_attempt(
                internal_task_id: str,
                attempt_id: str,
                device: Device,
            ) -> TaskResult:
                session = await self._resolve_session(device.device_id)
                model_id = model_by_device.get(device.device_id)
                if model_id is None:
                    fallback_plan = ExecutionPlan(
                        task_id=parent.task_id,
                        execution_mode=ExecutionMode.DATA_PARALLEL,
                        request_text=shard.request_text,
                        tasks=(
                            replace(
                                shard,
                                selected_device_id="",
                                selected_model_id="",
                                fallback_device_ids=(),
                            ),
                        ),
                        steering=routed.steering,
                    )
                    _, fallback = self.router.route(
                        fallback_plan, profile, [device]
                    )
                    model_id = fallback.selected_model_id
                    model_by_device[device.device_id] = model_id
                self._attempt_models[attempt_id] = model_id
                partial = await session.execute_shard(
                    pb.ExecuteShard(
                        task_id=parent.task_id,
                        attempt_id=attempt_id,
                        shard_id=shard.shard_id,
                        request_text=self._request_text_for_session(
                            session, shard.request_text, profile_context
                        ),
                        model_id=model_id,
                        timeout_ms=timeout_ms,
                        steering=steering_to_proto(routed.steering),
                    ),
                    timeout_seconds=timeout_ms / 1000,
                )
                result = partial_result_from_proto(partial)
                return replace(result, task_id=internal_task_id)

            return await self.dispatch.submit(
                shard.request_text,
                candidates,
                execute_attempt,
                task_id=child_task_id,
            )

        shard_results = await asyncio.gather(
            *(run_shard(shard) for shard in routed.tasks)
        )
        reducer_attempt = self.tasks.assign(parent.task_id, "brain-reducer")
        self.tasks.mark_running(reducer_attempt.attempt_id)
        self._attempt_models[reducer_attempt.attempt_id] = "parallel-reducer"
        failed = [
            item.task for item in shard_results if item.task.state.value != "SUCCEEDED"
        ]
        if failed:
            result = TaskResult(
                task_id=parent.task_id,
                attempt_id=reducer_attempt.attempt_id,
                success=False,
                output_text="",
                device_id="brain",
                error_code="SHARD_EXECUTION_FAILED",
                error_message=", ".join(
                    f"{task.task_id}:{task.error_code}" for task in failed
                ),
            )
            self.tasks.record_result(
                reducer_attempt.attempt_id,
                result,
                success=False,
                error_code=result.error_code,
                error_message=result.error_message,
            )
            return self._response_for_task(self.tasks.get(parent.task_id))

        parts = []
        total_latency = 0
        for shard, dispatched in zip(routed.tasks, shard_results, strict=True):
            result = dispatched.task.result
            if not isinstance(result, TaskResult):
                raise RuntimeError(f"{shard.shard_id} returned an invalid result")
            parts.append(f"[{shard.shard_id}]\n{result.output_text}")
            total_latency = max(total_latency, result.latency_ms)
        output = "\n\n".join(parts)
        result = TaskResult(
            task_id=parent.task_id,
            attempt_id=reducer_attempt.attempt_id,
            success=True,
            output_text=output,
            device_id="brain",
            latency_ms=total_latency,
            metrics=ExecutionMetrics(
                model_id="parallel-reducer",
                model_version="mock-v1",
                runtime_name=RuntimeName.MOCK.value,
                runtime_version="dragon-nest-0.1.0",
                accelerator="cpu",
                execution_latency_ms=total_latency,
            ),
        )
        self.tasks.record_result(reducer_attempt.attempt_id, result)
        return self._response_for_task(self.tasks.get(parent.task_id))

    async def _submit_replica_race(
        self,
        request: pb.SubmitTaskRequest,
        parent: TaskRecord,
        routed: ExecutionPlan,
        profile: TaskProfile,
        timeout_ms: int,
        profile_context: str = "",
    ) -> pb.SubmitTaskResponse:
        shard = routed.tasks[0]
        candidate_ids = tuple(
            dict.fromkeys((shard.selected_device_id, *shard.fallback_device_ids))
        )[:2]
        if len(candidate_ids) < 2:
            self.tasks.fail(
                parent.task_id,
                "NO_ELIGIBLE_FALLBACK",
                "replica race requires at least two eligible devices",
            )
            return self._response_for_task(self.tasks.get(parent.task_id))

        child_task_id = f"{parent.task_id}:{shard.shard_id}"
        self.tasks.create(shard.request_text, task_id=child_task_id)
        eligible = {
            device.device_id: device
            for device in self.registry.eligible(candidate_ids)
        }
        model_by_device = {shard.selected_device_id: shard.selected_model_id}
        running: dict[asyncio.Task[TaskResult], tuple[str, str]] = {}

        for device_id in candidate_ids:
            device = eligible.get(device_id)
            if device is None:
                continue
            model_id = model_by_device.get(device_id)
            if model_id is None:
                fallback_plan = ExecutionPlan(
                    task_id=parent.task_id,
                    execution_mode=ExecutionMode.DATA_PARALLEL,
                    request_text=shard.request_text,
                    tasks=(PlannedTask(shard.shard_id, shard.request_text),),
                    steering=routed.steering,
                    origin_device_id=routed.origin_device_id,
                )
                _, fallback = self.router.route(fallback_plan, profile, [device])
                model_id = fallback.selected_model_id
                model_by_device[device_id] = model_id
            attempt = self.tasks.assign_replica(child_task_id, device_id)
            self.tasks.mark_replica_running(attempt.attempt_id)
            self._attempt_models[attempt.attempt_id] = model_id

            async def execute_replica(
                replica_device: Device = device,
                replica_attempt_id: str = attempt.attempt_id,
                replica_model_id: str = model_id,
            ) -> TaskResult:
                session = await self._resolve_session(replica_device.device_id)
                partial = await session.execute_shard(
                    pb.ExecuteShard(
                        task_id=parent.task_id,
                        attempt_id=replica_attempt_id,
                        shard_id=shard.shard_id,
                        request_text=self._request_text_for_session(
                            session, request.request_text, profile_context
                        ),
                        model_id=replica_model_id,
                        timeout_ms=timeout_ms,
                        steering=steering_to_proto(routed.steering),
                    ),
                    timeout_seconds=timeout_ms / 1000,
                )
                result = partial_result_from_proto(partial)
                return replace(result, task_id=child_task_id)

            task = asyncio.create_task(execute_replica())
            running[task] = (attempt.attempt_id, device_id)

        if len(running) < 2:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            self.tasks.fail(
                child_task_id,
                "NO_ELIGIBLE_FALLBACK",
                "replica race could not start two eligible replicas",
            )
            self.tasks.fail(
                parent.task_id,
                "NO_ELIGIBLE_FALLBACK",
                "replica race could not start two eligible replicas",
            )
            return self._response_for_task(self.tasks.get(parent.task_id))

        pending = set(running)
        winner_result: TaskResult | None = None
        winner_attempt_id = ""
        winner_device_id = ""
        while pending and winner_result is None:
            completed, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in completed:
                attempt_id, device_id = running[task]
                try:
                    result = task.result()
                except DeviceOfflineError as exc:
                    self.tasks.record_replica_result(
                        attempt_id,
                        None,
                        success=False,
                        error_code="DEVICE_OFFLINE",
                        error_message=str(exc),
                    )
                    continue
                except Exception as exc:
                    self.tasks.record_replica_result(
                        attempt_id,
                        None,
                        success=False,
                        error_code="EXECUTION_FAILED",
                        error_message=str(exc),
                    )
                    continue

                if winner_result is not None:
                    self.tasks.record_result(
                        attempt_id,
                        result,
                        success=result.success,
                        error_code=result.error_code,
                        error_message=result.error_message,
                    )
                elif result.success:
                    self.tasks.record_replica_result(attempt_id, result)
                    winner_result = result
                    winner_attempt_id = attempt_id
                    winner_device_id = device_id
                else:
                    self.tasks.record_replica_result(
                        attempt_id,
                        result,
                        success=False,
                        error_code=result.error_code,
                        error_message=result.error_message,
                    )

        if winner_result is None:
            self.tasks.fail(
                parent.task_id,
                "REPLICA_RACE_FAILED",
                "all replicas failed",
            )
            return self._response_for_task(self.tasks.get(parent.task_id))

        loser_attempts = [
            attempt
            for attempt in self.tasks.get(child_task_id).attempts
            if attempt.attempt_id != winner_attempt_id
            and attempt.state == AttemptState.CANCELLED
        ]
        for loser in loser_attempts:
            session = await self.sessions.get(loser.device_id)
            if session is not None:
                await session.cancel(
                    parent.task_id,
                    loser.attempt_id,
                    f"replica race won by {winner_device_id}",
                )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        accepted = self.tasks.assign(parent.task_id, winner_device_id)
        self.tasks.mark_running(accepted.attempt_id)
        self._attempt_models[accepted.attempt_id] = self._attempt_models[
            winner_attempt_id
        ]
        final_result = replace(
            winner_result,
            task_id=parent.task_id,
            attempt_id=accepted.attempt_id,
        )
        self.tasks.record_result(accepted.attempt_id, final_result)
        return self._response_for_task(self.tasks.get(parent.task_id))

    async def _submit_layer_pipeline(
        self,
        request: pb.SubmitTaskRequest,
        routed: ExecutionPlan,
        timeout_ms: int,
        profile_context: str = "",
    ) -> pb.SubmitTaskResponse:
        parent = self.tasks.create(request.request_text, task_id=routed.task_id)
        boundary: pb.BoundaryTensor | None = None
        final_result: PipelineAttemptResult | None = None

        for index, stage in enumerate(routed.stages):
            child_task_id = f"{parent.task_id}:{stage.stage_id}"
            candidates, models_by_device = self._pipeline_stage_candidates(stage)

            async def execute_attempt(
                internal_task_id: str,
                attempt_id: str,
                device: Device,
                current_stage: PipelineStage = stage,
                input_boundary: pb.BoundaryTensor | None = boundary,
                final_stage: bool = index == len(routed.stages) - 1,
            ) -> PipelineAttemptResult:
                session = await self._resolve_session(device.device_id)
                model_id = models_by_device[device.device_id]
                self._attempt_models[attempt_id] = model_id
                stage_steering = (
                    routed.steering
                    if routed.steering.enabled
                    and current_stage.start_layer
                    <= routed.steering.target_layer
                    < current_stage.end_layer
                    else SteeringSpec()
                )
                result = await session.execute_pipeline_stage(
                    pb.ExecutePipelineStage(
                        task_id=parent.task_id,
                        attempt_id=attempt_id,
                        stage_id=current_stage.stage_id,
                        stage_index=current_stage.stage_index,
                        request_text=self._request_text_for_session(
                            session, request.request_text, profile_context
                        ),
                        model_id=model_id,
                        input_boundary=input_boundary,
                        final_stage=final_stage,
                        timeout_ms=timeout_ms,
                        steering=steering_to_proto(stage_steering),
                    ),
                    timeout_seconds=timeout_ms / 1000,
                )
                internal = pipeline_result_from_proto(result)
                output_boundary = (
                    result.output_boundary
                    if result.HasField("output_boundary")
                    and result.output_boundary.data
                    else None
                )
                if internal.success and not final_stage:
                    if output_boundary is None:
                        return PipelineAttemptResult(
                            False,
                            "",
                            device.device_id,
                            error_code="BOUNDARY_TENSOR_MISSING",
                            error_message=(
                                f"{current_stage.stage_id} returned no boundary tensor"
                            ),
                            metrics=internal.metrics,
                        )
                    error = self._boundary_error(output_boundary)
                    if error:
                        return PipelineAttemptResult(
                            False,
                            "",
                            device.device_id,
                            error_code="INVALID_BOUNDARY_TENSOR",
                            error_message=error,
                            metrics=internal.metrics,
                        )
                return PipelineAttemptResult(
                    success=internal.success,
                    output_text=internal.output_text,
                    device_id=device.device_id,
                    boundary=output_boundary,
                    error_code=internal.error_code,
                    error_message=internal.error_message,
                    metrics=internal.metrics,
                )

            dispatched = await self.dispatch.submit(
                request.request_text,
                candidates,
                execute_attempt,
                task_id=child_task_id,
            )
            if dispatched.task.state.value != "SUCCEEDED" or not isinstance(
                dispatched.task.result, PipelineAttemptResult
            ):
                return self._fail_pipeline_parent(
                    parent.task_id,
                    stage.stage_id,
                    dispatched.task.error_code or "PIPELINE_STAGE_FAILED",
                    dispatched.task.error_message,
                )
            final_result = dispatched.task.result
            boundary = final_result.boundary

        if final_result is None:
            return self._fail_pipeline_parent(
                parent.task_id,
                "",
                "PIPELINE_STAGE_FAILED",
                "pipeline contained no stages",
            )
        accepted = self.tasks.assign(parent.task_id, "brain-pipeline")
        self.tasks.mark_running(accepted.attempt_id)
        pipeline_id = routed.stages[0].pipeline_id
        self._attempt_models[accepted.attempt_id] = pipeline_id
        result = TaskResult(
            task_id=parent.task_id,
            attempt_id=accepted.attempt_id,
            success=True,
            output_text=(
                final_result.output_text
                + (
                    f"\nSteering: {routed.steering.vector_id} "
                    f"alpha={routed.steering.alpha} "
                    f"layer={routed.steering.target_layer} "
                    f"positions={routed.steering.positions}"
                    if routed.steering.enabled
                    else ""
                )
            ),
            device_id=final_result.device_id,
            latency_ms=(
                final_result.metrics.execution_latency_ms if final_result.metrics else 0
            ),
            metrics=final_result.metrics,
        )
        self.tasks.record_result(accepted.attempt_id, result)
        return self._response_for_task(self.tasks.get(parent.task_id))

    def _pipeline_stage_candidates(
        self, stage: PipelineStage
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        candidates = [stage.selected_device_id]
        models = {stage.selected_device_id: stage.selected_model_id}
        for device in self.registry.eligible():
            if device.device_id == stage.selected_device_id:
                continue
            for model in device.models:
                segment = model.segment
                if (
                    segment
                    and model.supports_layer_pipeline
                    and segment.pipeline_id == stage.pipeline_id
                    and segment.start_layer == stage.start_layer
                    and segment.end_layer == stage.end_layer
                    and model.model_family == stage.model_family
                    and model.model_version == stage.model_version
                    and model.tokenizer_id == stage.tokenizer_id
                    and model.precision == stage.precision
                    and model.boundary_format == stage.boundary_format
                    and device.health.available_memory_mb >= model.min_memory_mb
                ):
                    candidates.append(device.device_id)
                    models[device.device_id] = model.model_id
                    break
        return tuple(candidates), models

    def _boundary_error(self, boundary: pb.BoundaryTensor) -> str:
        if len(boundary.data) > self.config.max_boundary_bytes:
            return (
                f"boundary has {len(boundary.data)} bytes; maximum is "
                f"{self.config.max_boundary_bytes}"
            )
        expected = f"sha256:{hashlib.sha256(boundary.data).hexdigest()}"
        if boundary.checksum != expected:
            return "boundary checksum mismatch"
        if not boundary.dtype or not boundary.shape:
            return "boundary dtype and shape are required"
        return ""

    def _fail_pipeline_parent(
        self,
        task_id: str,
        stage_id: str,
        error_code: str,
        error_message: str,
    ) -> pb.SubmitTaskResponse:
        attempt = self.tasks.assign(task_id, "brain-pipeline")
        self.tasks.mark_running(attempt.attempt_id)
        result = TaskResult(
            task_id=task_id,
            attempt_id=attempt.attempt_id,
            success=False,
            output_text="",
            device_id="brain",
            error_code=error_code,
            error_message=(
                f"{stage_id}: {error_message}" if stage_id else error_message
            ),
        )
        self.tasks.record_result(
            attempt.attempt_id,
            result,
            success=False,
            error_code=error_code,
            error_message=result.error_message,
        )
        return self._response_for_task(self.tasks.get(task_id))

    async def GetTask(
        self,
        request: pb.GetTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.SubmitTaskResponse:
        try:
            task = self.tasks.get(request.task_id)
        except KeyError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "task not found")
            raise AssertionError("unreachable")
        return self._response_for_task(task)

    async def _consume_agent(
        self,
        session: AgentSession,
        request_iterator: AsyncIterator[pb.DeviceToBrain],
    ) -> None:
        async for message in request_iterator:
            kind = message.WhichOneof("payload")
            if kind == "health_update":
                health = message.health_update
                if health.device_id != session.device_id:
                    continue
                simulation = self._device_simulations.get(session.device_id, {})
                if simulation.get("offline") is True:
                    self.registry.mark_offline(
                        session.device_id, reason="simulated_offline"
                    )
                    await session.outbound.put(
                        pb.BrainToDevice(
                            heartbeat_ack=pb.HeartbeatAck(
                                brain_timestamp_ms=int(time.time() * 1000)
                            )
                        )
                    )
                    continue
                incoming_health = health_from_proto(health)
                health_overrides = {
                    key: value
                    for key, value in simulation.items()
                    if key != "offline"
                }
                if health_overrides:
                    incoming_health = replace(incoming_health, **health_overrides)
                self.registry.heartbeat(
                    session.device_id,
                    incoming_health,
                    active_task_ids=health.active_task_ids,
                    warm_model_ids=health.warm_model_ids,
                    simulated_constraint=bool(simulation)
                    or health.simulated_constraint,
                )
                await session.outbound.put(
                    pb.BrainToDevice(
                        heartbeat_ack=pb.HeartbeatAck(
                            brain_timestamp_ms=int(time.time() * 1000)
                        )
                    )
                )
            elif kind == "task_result":
                result = message.task_result
                if result.device_id and result.device_id != session.device_id:
                    continue
                if not session.resolve(result):
                    internal = task_result_from_proto(result)
                    with contextlib.suppress(KeyError):
                        self.tasks.record_result(
                            result.attempt_id,
                            internal,
                            success=internal.success,
                            error_code=internal.error_code,
                            error_message=internal.error_message,
                        )
            elif kind == "partial_task_result":
                result = message.partial_task_result
                if result.device_id and result.device_id != session.device_id:
                    continue
                if not session.resolve(result):
                    internal = partial_result_from_proto(result)
                    with contextlib.suppress(KeyError):
                        self.tasks.record_result(
                            result.attempt_id,
                            internal,
                            success=internal.success,
                            error_code=internal.error_code,
                            error_message=internal.error_message,
                        )
            elif kind == "pipeline_stage_result":
                result = message.pipeline_stage_result
                if result.device_id and result.device_id != session.device_id:
                    continue
                if not session.resolve(result):
                    internal = pipeline_result_from_proto(result)
                    with contextlib.suppress(KeyError):
                        self.tasks.record_result(
                            result.attempt_id,
                            internal,
                            success=internal.success,
                            error_code=internal.error_code,
                            error_message=internal.error_message,
                        )
            elif kind == "shutdown":
                if message.shutdown.device_id != session.device_id:
                    continue
                session.graceful = True
                self.registry.mark_offline(
                    session.device_id,
                    reason=message.shutdown.reason or "agent_shutdown",
                )
                await session.close(graceful=True)
                return

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.sweep_interval_seconds)
            for event in self.registry.sweep():
                if event.status == HealthStatus.OFFLINE:
                    self.dispatch.handle_device_offline(event.device_id)
                    session = await self.sessions.get(event.device_id)
                    if session is not None and session.transport == "grpc_stream":
                        await self.sessions.close_device(event.device_id)

    def _registration_error(
        self, registration: pb.RegisterDevice, peer_fingerprint: str = ""
    ) -> str:
        error, _ = self._authorize_registration(registration, peer_fingerprint)
        return error

    def _authorize_registration(
        self, registration: pb.RegisterDevice, peer_fingerprint: str = ""
    ) -> tuple[str, str]:
        if not registration.device_id:
            return "device_id is required", ""
        if not registration.models:
            return "at least one model capability is required", ""
        issued_credential = ""
        if self.config.dev_mode:
            credential = registration.enrollment_token
            bootstrap_session = self.enrollment.session_for_bootstrap(credential)
            profile_values: dict[str, object] | None = None
            if registration.HasField("personal_profile"):
                try:
                    profile_values = self._profile_values_from_registration(registration)
                except ProfileError as exc:
                    return f"personal profile is invalid: {exc}", ""
            claim = None
            if credential == self.config.enrollment_token:
                pass
            elif self.enrollment.valid_device_credential(
                registration.device_id, credential
            ):
                if registration.HasField("personal_profile"):
                    association = self.profiles.association_for_device(
                        registration.device_id
                    )
                    if association is not None:
                        try:
                            update_values = self._profile_values_from_registration(
                                registration
                            )
                            self.profiles.update(
                                association.profile_id, **update_values
                            )
                        except ProfileError as exc:
                            return f"personal profile is invalid: {exc}", ""
            else:
                claim = self.enrollment.claim(credential, registration.device_id)
                if claim is None:
                    return "invalid enrollment token or expired credential", ""
                issued_credential = claim.device_credential
            try:
                association = self.profiles.association_for_device(
                    registration.device_id
                )
                profile_id = association.profile_id if association else ""
                if claim is not None and claim.profile_id:
                    profile_id = claim.profile_id
                device_name = (
                    (claim.device_name if claim is not None else "")
                    or (association.device_name if association is not None else "")
                    or registration.display_name
                )
                if profile_values is not None:
                    if profile_id:
                        self.profiles.update(profile_id, **profile_values)
                    else:
                        profile_id = self.profiles.create(**profile_values).profile_id
                if profile_id:
                    self.profiles.associate_device(
                        registration.device_id, profile_id, device_name
                    )
                    if (
                        claim is not None
                        and bootstrap_session is not None
                        and not bootstrap_session.profile_id
                    ):
                        self.enrollment.attach_profile(
                            claim.session_id, profile_id, device_name
                        )
            except (EnrollmentError, ProfileError) as exc:
                return f"personal profile association failed: {exc}", ""
        if not self.config.dev_mode:
            if not peer_fingerprint:
                return (
                    "production enrollment requires a verified client certificate",
                    "",
                )
            if registration.certificate_fingerprint != peer_fingerprint:
                return "reported certificate fingerprint does not match TLS peer", ""
            if peer_fingerprint in self.revoked_certificate_fingerprints:
                return "client certificate has been revoked", ""
            if registration.HasField("personal_profile"):
                try:
                    values = self._profile_values_from_registration(registration)
                    association = self.profiles.association_for_device(
                        registration.device_id
                    )
                    if association is None:
                        profile_id = self.profiles.create(**values).profile_id
                    else:
                        profile_id = association.profile_id
                        self.profiles.update(profile_id, **values)
                    self.profiles.associate_device(
                        registration.device_id,
                        profile_id,
                        association.device_name
                        if association is not None
                        else registration.display_name,
                    )
                except ProfileError as exc:
                    return f"personal profile association failed: {exc}", ""
        return "", issued_credential

    def _profile_values_from_registration(
        self, registration: pb.RegisterDevice
    ) -> dict[str, object]:
        profile = registration.personal_profile
        values: dict[str, object] = {
            "person_name": profile.person_name,
            "preferred_mode": profile.preferred_mode or "auto",
            "steering_vector_id": profile.steering_vector_id,
            "steering_alpha": profile.steering_alpha,
            "steering_positions": profile.steering_positions or "last",
            "allow_remote_vector": profile.allow_remote_vector,
            "notes": profile.notes,
            "persona_id": profile.persona_id or self._persona_from_steering(
                profile.steering_vector_id, profile.steering_alpha
            ),
        }
        persona_id = str(values["persona_id"])
        if persona_id not in PERSONA_STEERING:
            raise ProfileError(f"unsupported persona {persona_id!r}")
        persona_alpha = PERSONA_STEERING[persona_id]
        if persona_alpha is None:
            values["steering_vector_id"] = ""
            values["steering_alpha"] = 0.0
        else:
            values["steering_vector_id"] = PERSONA_VECTOR_ID
            values["steering_alpha"] = persona_alpha
            values["steering_positions"] = "last"
        vector_id = str(values["steering_vector_id"])
        if vector_id:
            if self.steering_registry is None:
                raise ProfileError("Brain has no steering registry configured")
            vectors = {
                vector.vector_id: vector for vector in self.steering_registry.vectors()
            }
            vector = vectors.get(vector_id)
            if vector is None:
                raise ProfileError(f"unknown steering vector {vector_id}")
            alpha = float(values["steering_alpha"])
            positions = str(values["steering_positions"])
            if not vector.alpha_min <= alpha <= vector.alpha_max:
                raise ProfileError(
                    f"steering alpha must be between {vector.alpha_min} and {vector.alpha_max}"
                )
            if positions not in vector.positions:
                raise ProfileError(
                    f"steering positions {positions!r} are not supported"
                )
        # ProfileStore performs the canonical string and numeric validation.
        return values

    def _steering_for_profile(self, profile: PersonalProfile) -> SteeringSpec:
        if self.steering_registry is None:
            return SteeringSpec()
        default = self.steering_registry.default_spec(profile.steering_vector_id)
        return replace(
            default,
            alpha=profile.steering_alpha,
            positions=profile.steering_positions,
            allow_remote_vector=(
                default.allow_remote_vector and profile.allow_remote_vector
            ),
        )

    def _steering_for_persona(self, persona_id: str) -> SteeringSpec:
        alpha = PERSONA_STEERING[persona_id]
        if alpha is None:
            return SteeringSpec()
        if self.steering_registry is None:
            raise ProfileError(f"Persona {persona_id!r} requires steering")
        try:
            default = self.steering_registry.default_spec(PERSONA_VECTOR_ID)
        except KeyError as exc:
            raise ProfileError(
                f"Persona {persona_id!r} is not available on this Brain"
            ) from exc
        return replace(default, alpha=alpha, positions="last")

    @staticmethod
    def _persona_from_steering(vector_id: str, alpha: float) -> str:
        if not vector_id or alpha == 0:
            return "balanced"
        return "concise" if alpha < 0 else "detailed"

    @staticmethod
    def _with_profile_context(request_text: str, profile_context: str) -> str:
        context = profile_context.strip()
        if not context:
            return request_text
        return f"About the user:\n{context}\n\nRequest:\n{request_text}"

    @classmethod
    def _request_text_for_session(
        cls,
        session: DeviceSession,
        request_text: str,
        profile_context: str,
    ) -> str:
        if not session.allow_profile_context:
            return request_text
        return cls._with_profile_context(request_text, profile_context)

    @staticmethod
    def _device_exclusion_reason(record) -> str:
        device = record.device
        if record.status == HealthStatus.UNHEALTHY:
            return (
                f"Excluded {device.device_id}: health status UNHEALTHY "
                f"(thermal={device.health.thermal_level:.2f}, "
                f"accelerator={device.health.accelerator_utilization:.2f}, "
                f"rtt={device.health.network_rtt_ms:.0f} ms)."
            )
        if record.status == HealthStatus.OFFLINE:
            return f"Excluded {device.device_id}: device is OFFLINE."
        if not record.stream_connected:
            return f"Excluded {device.device_id}: Agent stream is disconnected."
        if not device.health.reachable:
            return f"Excluded {device.device_id}: Agent reported unreachable."
        return ""

    @staticmethod
    def _peer_certificate_fingerprint(
        context: grpc.aio.ServicerContext,
    ) -> str:
        auth_context = context.auth_context()
        certificates = auth_context.get("x509_pem_cert", ())
        if not certificates:
            return ""
        certificate = certificates[0]
        try:
            der = ssl.PEM_cert_to_DER_cert(certificate.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            der = certificate
        return f"sha256:{hashlib.sha256(der).hexdigest()}"

    async def revoke_certificate(self, fingerprint: str) -> tuple[str, ...]:
        self.revoked_certificate_fingerprints.add(fingerprint)
        affected = tuple(
            device_id
            for device_id, current in self.certificate_fingerprints.items()
            if current == fingerprint
        )
        for device_id in affected:
            self.registry.mark_offline(device_id, reason="certificate_revoked")
            self.dispatch.handle_device_offline(device_id)
            await self.sessions.close_device(device_id)
        return affected

    def _response_for_task(self, task: TaskRecord) -> pb.SubmitTaskResponse:
        result = task.result if isinstance(task.result, TaskResult) else None
        model_id = self._attempt_models.get(task.accepted_attempt_id, "")
        plan = self.execution_plans.get(task.task_id)
        device_display_name = ""
        if result and result.device_id and result.device_id != "brain":
            with contextlib.suppress(KeyError):
                device_display_name = self.registry.get(
                    result.device_id
                ).device.display_name
        return pb.SubmitTaskResponse(
            task_id=task.task_id,
            state=task.state.value,
            success=bool(result and result.success),
            output_text=result.output_text if result else "",
            error_code=task.error_code or (result.error_code if result else ""),
            error_message=task.error_message
            or (result.error_message if result else ""),
            accepted_attempt_id=task.accepted_attempt_id,
            device_id=result.device_id if result else "",
            model_id=model_id,
            route_reasons=self._route_reasons.get(task.task_id, ()),
            steering=steering_to_proto(
                self._steering_specs.get(task.task_id, SteeringSpec())
            ),
            origin_device_id=plan.origin_device_id if plan else "",
            reducer=plan.reducer if plan else "",
            device_display_name=device_display_name,
        )


async def create_server(
    service: BrainService,
    address: str = "127.0.0.1:50051",
) -> tuple[grpc.aio.Server, int]:
    server = grpc.aio.server(
        options=(
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        )
    )
    pb_grpc.add_BrainControlServicer_to_server(service, server)
    if service.config.tls_server_certificate_path:
        certificate = _read_bytes(service.config.tls_server_certificate_path)
        key = _read_bytes(service.config.tls_server_key_path)
        client_ca = (
            _read_bytes(service.config.tls_client_ca_path)
            if service.config.tls_client_ca_path
            else None
        )
        credentials = grpc.ssl_server_credentials(
            ((key, certificate),),
            root_certificates=client_ca,
            require_client_auth=service.config.require_client_certificate,
        )
        port = server.add_secure_port(address, credentials)
    else:
        if not service.config.dev_mode:
            raise RuntimeError("production mode requires a TLS server certificate")
        port = server.add_insecure_port(address)
    if port == 0:
        raise RuntimeError(f"failed to bind gRPC server to {address}")
    await service.start()
    await server.start()
    return server, port


async def stop_server(server: grpc.aio.Server, service: BrainService) -> None:
    await service.stop()
    await server.stop(grace=1)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()

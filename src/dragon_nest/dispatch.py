from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from .models import Device, HealthStatus
from .registry import DeviceRegistry
from .tasks import ResultDisposition, TaskRecord, TaskStore


class DeviceOfflineError(RuntimeError):
    pass


ExecuteAttempt = Callable[[str, str, Device], Awaitable[Any]]


@dataclass(frozen=True)
class DispatchResult:
    task: TaskRecord
    disposition: ResultDisposition | None


class DispatchManager:
    def __init__(
        self,
        registry: DeviceRegistry,
        tasks: TaskStore,
        max_offline_retries: int = 1,
    ):
        if max_offline_retries < 0:
            raise ValueError("max_offline_retries cannot be negative")
        self.registry = registry
        self.tasks = tasks
        self.max_offline_retries = max_offline_retries

    async def submit(
        self,
        request: Any,
        candidate_device_ids: Iterable[str],
        execute: ExecuteAttempt,
        task_id: str | None = None,
    ) -> DispatchResult:
        task = self.tasks.create(request, task_id=task_id)
        candidates = tuple(dict.fromkeys(candidate_device_ids))
        tried: set[str] = set()
        retries = 0

        while True:
            device = self._best_eligible(candidates, tried)
            if device is None:
                failed = self.tasks.fail(
                    task.task_id,
                    "NO_ELIGIBLE_FALLBACK",
                    "no eligible device remains for this task",
                )
                return DispatchResult(failed, None)

            tried.add(device.device_id)
            attempt = self.tasks.assign(task.task_id, device.device_id)
            self.tasks.mark_running(attempt.attempt_id)
            try:
                result = await execute(task.task_id, attempt.attempt_id, device)
            except DeviceOfflineError:
                self._mark_device_and_attempt_offline(
                    device.device_id, attempt.attempt_id
                )
                if retries >= self.max_offline_retries:
                    failed = self.tasks.fail(
                        task.task_id,
                        "NO_ELIGIBLE_FALLBACK",
                        "offline retry limit exhausted",
                    )
                    return DispatchResult(failed, None)
                retries += 1
                continue
            except Exception as exc:
                disposition = self.tasks.record_result(
                    attempt.attempt_id,
                    None,
                    success=False,
                    error_code="EXECUTION_FAILED",
                    error_message=str(exc),
                )
                return DispatchResult(self.tasks.get(task.task_id), disposition)

            if self.registry.get(device.device_id).status == HealthStatus.OFFLINE:
                self.tasks.mark_device_offline(attempt.attempt_id)
                self.tasks.record_result(attempt.attempt_id, result)
                if retries >= self.max_offline_retries:
                    failed = self.tasks.fail(
                        task.task_id,
                        "NO_ELIGIBLE_FALLBACK",
                        "offline retry limit exhausted",
                    )
                    return DispatchResult(failed, ResultDisposition.STALE)
                retries += 1
                continue

            success = bool(getattr(result, "success", True))
            disposition = self.tasks.record_result(
                attempt.attempt_id,
                result,
                success=success,
                error_code=str(getattr(result, "error_code", "EXECUTION_FAILED")),
                error_message=str(getattr(result, "error_message", "")),
            )
            return DispatchResult(self.tasks.get(task.task_id), disposition)

    def handle_device_offline(self, device_id: str) -> tuple[str, ...]:
        self.registry.mark_offline(device_id, reason="dispatch_offline_event")
        affected: list[str] = []
        for attempt in self.tasks.active_attempts_for_device(device_id):
            self.tasks.mark_device_offline(attempt.attempt_id)
            affected.append(attempt.attempt_id)
        return tuple(affected)

    def _best_eligible(
        self,
        candidate_device_ids: tuple[str, ...],
        tried: set[str],
    ) -> Device | None:
        remaining = tuple(
            device_id for device_id in candidate_device_ids if device_id not in tried
        )
        eligible = {
            device.device_id: device
            for device in self.registry.eligible(remaining, allow_stale_fallback=True)
        }
        for device_id in remaining:
            if device_id in eligible:
                return eligible[device_id]
        return None

    def _mark_device_and_attempt_offline(self, device_id: str, attempt_id: str) -> None:
        if self.registry.get(device_id).status != HealthStatus.OFFLINE:
            self.registry.mark_offline(device_id, reason="execution_reported_offline")
        self.tasks.mark_device_offline(attempt_id)

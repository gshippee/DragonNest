from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Callable


class TaskState(StrEnum):
    QUEUED = "QUEUED"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"


class AttemptState(StrEnum):
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    DEVICE_OFFLINE = "DEVICE_OFFLINE"
    SUPERSEDED = "SUPERSEDED"
    CANCELLED = "CANCELLED"


class ResultDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    FAILED = "FAILED"
    STALE = "STALE"


@dataclass(frozen=True)
class TaskAttempt:
    attempt_id: str
    device_id: str
    state: AttemptState
    created_at: float
    updated_at: float
    error_code: str = ""
    error_message: str = ""
    result: Any = None


@dataclass(frozen=True)
class StaleResult:
    attempt_id: str
    received_at: float
    result: Any
    reason: str


@dataclass(frozen=True)
class TaskEvent:
    timestamp: float
    task_id: str
    event_type: str
    message: str
    attempt_id: str = ""
    device_id: str = ""


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    request: Any
    state: TaskState
    created_at: float
    updated_at: float
    attempts: tuple[TaskAttempt, ...] = ()
    current_attempt_id: str = ""
    accepted_attempt_id: str = ""
    result: Any = None
    error_code: str = ""
    error_message: str = ""
    stale_results: tuple[StaleResult, ...] = ()


_FINAL_STATES = {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED}
_ACTIVE_ATTEMPTS = {AttemptState.DISPATCHED, AttemptState.RUNNING}


class TaskStore:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._tasks: dict[str, TaskRecord] = {}
        self._attempt_tasks: dict[str, str] = {}
        self._events: list[TaskEvent] = []
        self._lock = threading.RLock()

    def create(self, request: Any, task_id: str | None = None) -> TaskRecord:
        timestamp = self._clock()
        task_id = task_id or f"task-{uuid.uuid4().hex}"
        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"duplicate task_id {task_id}")
            record = TaskRecord(
                task_id, request, TaskState.QUEUED, timestamp, timestamp
            )
            self._tasks[task_id] = record
            self._event(record, "QUEUED", "task queued")
            return record

    def assign(self, task_id: str, device_id: str) -> TaskAttempt:
        timestamp = self._clock()
        with self._lock:
            task = self._require(task_id)
            if task.state not in {TaskState.QUEUED, TaskState.RETRYING}:
                raise ValueError(f"cannot assign task {task_id} from {task.state}")
            attempts = list(task.attempts)
            if task.current_attempt_id:
                index, current = self._find_attempt(task, task.current_attempt_id)
                if current.state in _ACTIVE_ATTEMPTS:
                    attempts[index] = replace(
                        current,
                        state=AttemptState.SUPERSEDED,
                        updated_at=timestamp,
                        error_code="SUPERSEDED",
                    )
            attempt = TaskAttempt(
                attempt_id=f"attempt-{uuid.uuid4().hex}",
                device_id=device_id,
                state=AttemptState.DISPATCHED,
                created_at=timestamp,
                updated_at=timestamp,
            )
            attempts.append(attempt)
            updated = replace(
                task,
                state=TaskState.DISPATCHED,
                updated_at=timestamp,
                attempts=tuple(attempts),
                current_attempt_id=attempt.attempt_id,
                error_code="",
                error_message="",
            )
            self._tasks[task_id] = updated
            self._attempt_tasks[attempt.attempt_id] = task_id
            self._event(
                updated,
                "DISPATCHED",
                f"attempt dispatched to {device_id}",
                attempt,
            )
            return attempt

    def assign_replica(self, task_id: str, device_id: str) -> TaskAttempt:
        timestamp = self._clock()
        with self._lock:
            task = self._require(task_id)
            if task.state not in {
                TaskState.QUEUED,
                TaskState.DISPATCHED,
                TaskState.RUNNING,
            }:
                raise ValueError(f"cannot assign replica for {task_id} from {task.state}")
            attempt = TaskAttempt(
                attempt_id=f"attempt-{uuid.uuid4().hex}",
                device_id=device_id,
                state=AttemptState.DISPATCHED,
                created_at=timestamp,
                updated_at=timestamp,
            )
            updated = replace(
                task,
                state=TaskState.DISPATCHED,
                updated_at=timestamp,
                attempts=(*task.attempts, attempt),
                current_attempt_id=task.current_attempt_id or attempt.attempt_id,
                error_code="",
                error_message="",
            )
            self._tasks[task_id] = updated
            self._attempt_tasks[attempt.attempt_id] = task_id
            self._event(
                updated,
                "REPLICA_DISPATCHED",
                f"replica dispatched to {device_id}",
                attempt,
            )
            return attempt

    def mark_running(self, attempt_id: str) -> TaskAttempt:
        timestamp = self._clock()
        with self._lock:
            task = self._task_for_attempt(attempt_id)
            index, attempt = self._find_attempt(task, attempt_id)
            if (
                task.current_attempt_id != attempt_id
                or attempt.state != AttemptState.DISPATCHED
            ):
                raise ValueError(f"cannot start inactive attempt {attempt_id}")
            updated_attempt = replace(
                attempt, state=AttemptState.RUNNING, updated_at=timestamp
            )
            self._replace_attempt(
                task,
                index,
                updated_attempt,
                state=TaskState.RUNNING,
                timestamp=timestamp,
            )
            self._event(
                self._tasks[task.task_id],
                "RUNNING",
                f"attempt running on {attempt.device_id}",
                updated_attempt,
            )
            return updated_attempt

    def mark_replica_running(self, attempt_id: str) -> TaskAttempt:
        timestamp = self._clock()
        with self._lock:
            task = self._task_for_attempt(attempt_id)
            index, attempt = self._find_attempt(task, attempt_id)
            if attempt.state != AttemptState.DISPATCHED:
                raise ValueError(f"cannot start replica {attempt_id} from {attempt.state}")
            updated_attempt = replace(
                attempt, state=AttemptState.RUNNING, updated_at=timestamp
            )
            self._replace_attempt(
                task,
                index,
                updated_attempt,
                state=TaskState.RUNNING,
                timestamp=timestamp,
            )
            self._event(
                self._tasks[task.task_id],
                "REPLICA_RUNNING",
                f"replica running on {attempt.device_id}",
                updated_attempt,
            )
            return updated_attempt

    def record_replica_result(
        self,
        attempt_id: str,
        result: Any,
        success: bool = True,
        error_code: str = "",
        error_message: str = "",
    ) -> ResultDisposition:
        timestamp = self._clock()
        with self._lock:
            task = self._task_for_attempt(attempt_id)
            index, attempt = self._find_attempt(task, attempt_id)
            if task.state in _FINAL_STATES or attempt.state not in _ACTIVE_ATTEMPTS:
                reason = (
                    f"task already {task.state}"
                    if task.state in _FINAL_STATES
                    else f"attempt is {attempt.state}"
                )
                stale = StaleResult(attempt_id, timestamp, result, reason)
                self._tasks[task.task_id] = replace(
                    task,
                    updated_at=timestamp,
                    stale_results=(*task.stale_results, stale),
                )
                self._event(self._tasks[task.task_id], "STALE_RESULT", reason, attempt)
                return ResultDisposition.STALE

            attempts = list(task.attempts)
            if success:
                winner = replace(
                    attempt,
                    state=AttemptState.SUCCEEDED,
                    updated_at=timestamp,
                    result=result,
                )
                attempts[index] = winner
                cancelled: list[TaskAttempt] = []
                for other_index, other in enumerate(attempts):
                    if other.attempt_id == attempt_id or other.state not in _ACTIVE_ATTEMPTS:
                        continue
                    loser = replace(
                        other,
                        state=AttemptState.CANCELLED,
                        updated_at=timestamp,
                        error_code="REPLICA_LOST",
                        error_message=f"cancelled after {attempt.device_id} won replica race",
                    )
                    attempts[other_index] = loser
                    cancelled.append(loser)
                updated = replace(
                    task,
                    state=TaskState.SUCCEEDED,
                    updated_at=timestamp,
                    attempts=tuple(attempts),
                    current_attempt_id=attempt_id,
                    accepted_attempt_id=attempt_id,
                    result=result,
                    error_code="",
                    error_message="",
                )
                self._tasks[task.task_id] = updated
                self._event(
                    updated,
                    "REPLICA_WON",
                    f"accepted first successful result from {attempt.device_id}",
                    winner,
                )
                for loser in cancelled:
                    self._event(
                        updated,
                        "REPLICA_CANCELLED",
                        f"cancelled losing replica on {loser.device_id}",
                        loser,
                    )
                return ResultDisposition.ACCEPTED

            failed = replace(
                attempt,
                state=AttemptState.FAILED,
                updated_at=timestamp,
                result=result,
                error_code=error_code or "EXECUTION_FAILED",
                error_message=error_message,
            )
            attempts[index] = failed
            active = [item for item in attempts if item.state in _ACTIVE_ATTEMPTS]
            updated = replace(
                task,
                state=TaskState.RUNNING if active else TaskState.FAILED,
                updated_at=timestamp,
                attempts=tuple(attempts),
                current_attempt_id=(active[0].attempt_id if active else attempt_id),
                error_code="" if active else failed.error_code,
                error_message="" if active else error_message,
            )
            self._tasks[task.task_id] = updated
            self._event(
                updated,
                "REPLICA_FAILED" if active else "FAILED",
                error_message or failed.error_code,
                failed,
            )
            return ResultDisposition.FAILED

    def mark_device_offline(self, attempt_id: str) -> TaskAttempt:
        timestamp = self._clock()
        with self._lock:
            task = self._task_for_attempt(attempt_id)
            index, attempt = self._find_attempt(task, attempt_id)
            if attempt.state == AttemptState.DEVICE_OFFLINE:
                return attempt
            if attempt.state not in _ACTIVE_ATTEMPTS:
                raise ValueError(f"cannot mark completed attempt {attempt_id} offline")
            updated_attempt = replace(
                attempt,
                state=AttemptState.DEVICE_OFFLINE,
                updated_at=timestamp,
                error_code="DEVICE_OFFLINE",
                error_message="assigned device became offline",
            )
            self._replace_attempt(
                task,
                index,
                updated_attempt,
                state=(
                    TaskState.CANCELLING
                    if task.state == TaskState.CANCELLING
                    else TaskState.RETRYING
                ),
                timestamp=timestamp,
                error_code="DEVICE_OFFLINE",
                error_message="assigned device became offline",
            )
            self._event(
                self._tasks[task.task_id],
                "DEVICE_OFFLINE",
                f"device {attempt.device_id} became offline; retry pending",
                updated_attempt,
            )
            return updated_attempt

    def record_result(
        self,
        attempt_id: str,
        result: Any,
        success: bool = True,
        error_code: str = "",
        error_message: str = "",
    ) -> ResultDisposition:
        timestamp = self._clock()
        with self._lock:
            task = self._task_for_attempt(attempt_id)
            index, attempt = self._find_attempt(task, attempt_id)
            stale_reason = ""
            if task.state not in {TaskState.DISPATCHED, TaskState.RUNNING}:
                stale_reason = f"task already {task.state}"
            elif task.current_attempt_id != attempt_id:
                stale_reason = "attempt superseded"
            elif attempt.state not in _ACTIVE_ATTEMPTS:
                stale_reason = f"attempt is {attempt.state}"
            if stale_reason:
                stale = StaleResult(attempt_id, timestamp, result, stale_reason)
                self._tasks[task.task_id] = replace(
                    task,
                    updated_at=timestamp,
                    stale_results=(*task.stale_results, stale),
                )
                self._event(
                    self._tasks[task.task_id],
                    "STALE_RESULT",
                    stale_reason,
                    attempt,
                )
                return ResultDisposition.STALE

            if success:
                updated_attempt = replace(
                    attempt,
                    state=AttemptState.SUCCEEDED,
                    updated_at=timestamp,
                    result=result,
                )
                self._replace_attempt(
                    task,
                    index,
                    updated_attempt,
                    state=TaskState.SUCCEEDED,
                    timestamp=timestamp,
                    result=result,
                    accepted_attempt_id=attempt_id,
                )
                self._event(
                    self._tasks[task.task_id],
                    "SUCCEEDED",
                    f"accepted result from {attempt.device_id}",
                    updated_attempt,
                )
                return ResultDisposition.ACCEPTED

            updated_attempt = replace(
                attempt,
                state=AttemptState.FAILED,
                updated_at=timestamp,
                result=result,
                error_code=error_code or "EXECUTION_FAILED",
                error_message=error_message,
            )
            self._replace_attempt(
                task,
                index,
                updated_attempt,
                state=TaskState.FAILED,
                timestamp=timestamp,
                error_code=updated_attempt.error_code,
                error_message=error_message,
            )
            self._event(
                self._tasks[task.task_id],
                "FAILED",
                error_message or updated_attempt.error_code,
                updated_attempt,
            )
            return ResultDisposition.FAILED

    def fail(
        self, task_id: str, error_code: str, error_message: str = ""
    ) -> TaskRecord:
        timestamp = self._clock()
        with self._lock:
            task = self._require(task_id)
            if task.state in _FINAL_STATES:
                return task
            updated = replace(
                task,
                state=TaskState.FAILED,
                updated_at=timestamp,
                error_code=error_code,
                error_message=error_message,
            )
            self._tasks[task_id] = updated
            self._event(updated, "FAILED", error_message or error_code)
            return updated

    def begin_cancellation(self, task_id: str) -> TaskRecord:
        timestamp = self._clock()
        with self._lock:
            task = self._require(task_id)
            if task.state != TaskState.RUNNING:
                raise ValueError(f"cannot cancel task {task_id} from {task.state}")
            updated = replace(task, state=TaskState.CANCELLING, updated_at=timestamp)
            self._tasks[task_id] = updated
            self._event(updated, "CANCELLING", "cancellation requested")
            return updated

    def complete_cancellation(self, task_id: str) -> TaskRecord:
        timestamp = self._clock()
        with self._lock:
            task = self._require(task_id)
            if task.state != TaskState.CANCELLING:
                raise ValueError(f"task {task_id} is not cancelling")
            attempts = list(task.attempts)
            if task.current_attempt_id:
                index, attempt = self._find_attempt(task, task.current_attempt_id)
                if attempt.state in _ACTIVE_ATTEMPTS:
                    attempts[index] = replace(
                        attempt, state=AttemptState.CANCELLED, updated_at=timestamp
                    )
            updated = replace(
                task,
                state=TaskState.CANCELLED,
                updated_at=timestamp,
                attempts=tuple(attempts),
            )
            self._tasks[task_id] = updated
            self._event(updated, "CANCELLED", "task cancelled")
            return updated

    def active_attempts_for_device(self, device_id: str) -> tuple[TaskAttempt, ...]:
        with self._lock:
            return tuple(
                attempt
                for task in self._tasks.values()
                for attempt in task.attempts
                if attempt.device_id == device_id and attempt.state in _ACTIVE_ATTEMPTS
            )

    def get(self, task_id: str) -> TaskRecord:
        with self._lock:
            return self._require(task_id)

    def records(self) -> tuple[TaskRecord, ...]:
        with self._lock:
            return tuple(sorted(self._tasks.values(), key=lambda task: task.created_at))

    def events(self) -> tuple[TaskEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def _replace_attempt(
        self,
        task: TaskRecord,
        index: int,
        attempt: TaskAttempt,
        state: TaskState,
        timestamp: float,
        **changes: Any,
    ) -> TaskRecord:
        attempts = list(task.attempts)
        attempts[index] = attempt
        updated = replace(
            task,
            state=state,
            updated_at=timestamp,
            attempts=tuple(attempts),
            **changes,
        )
        self._tasks[task.task_id] = updated
        return updated

    def _task_for_attempt(self, attempt_id: str) -> TaskRecord:
        try:
            task_id = self._attempt_tasks[attempt_id]
        except KeyError as exc:
            raise KeyError(f"unknown attempt {attempt_id}") from exc
        return self._require(task_id)

    @staticmethod
    def _find_attempt(task: TaskRecord, attempt_id: str) -> tuple[int, TaskAttempt]:
        for index, attempt in enumerate(task.attempts):
            if attempt.attempt_id == attempt_id:
                return index, attempt
        raise KeyError(f"attempt {attempt_id} is not attached to task {task.task_id}")

    def _require(self, task_id: str) -> TaskRecord:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task {task_id}") from exc

    def _event(
        self,
        task: TaskRecord,
        event_type: str,
        message: str,
        attempt: TaskAttempt | None = None,
    ) -> None:
        self._events.append(
            TaskEvent(
                timestamp=self._clock(),
                task_id=task.task_id,
                event_type=event_type,
                message=message,
                attempt_id=attempt.attempt_id if attempt else "",
                device_id=attempt.device_id if attempt else "",
            )
        )

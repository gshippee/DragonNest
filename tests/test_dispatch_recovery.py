from __future__ import annotations

import asyncio

from dragon_nest.dispatch import DeviceOfflineError, DispatchManager
from dragon_nest.models import Device, HealthState, TaskResult
from dragon_nest.registry import DeviceRegistry
from dragon_nest.tasks import AttemptState, ResultDisposition, TaskState, TaskStore


def _device(device_id: str) -> Device:
    return Device(
        device_id,
        device_id,
        "pc",
        "windows",
        8192,
        HealthState(battery_pct=100, available_memory_mb=4096),
        (),
    )


def _manager(*device_ids: str) -> tuple[DeviceRegistry, TaskStore, DispatchManager]:
    registry = DeviceRegistry()
    for device_id in device_ids:
        registry.register(_device(device_id))
    tasks = TaskStore()
    return registry, tasks, DispatchManager(registry, tasks)


def test_mid_job_offline_retries_once_with_same_task_and_new_attempt():
    registry, tasks, manager = _manager("pc-01", "phone-01")
    calls: list[tuple[str, str, str]] = []

    async def execute(task_id, attempt_id, device):
        calls.append((task_id, attempt_id, device.device_id))
        if device.device_id == "pc-01":
            raise DeviceOfflineError("connection lost")
        return TaskResult(task_id, True, "fallback result", device.device_id)

    dispatched = asyncio.run(
        manager.submit(
            "user request",
            ["pc-01", "phone-01"],
            execute,
            task_id="task-stable",
        )
    )

    assert dispatched.task.state == TaskState.SUCCEEDED
    assert dispatched.task.task_id == "task-stable"
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0]
    assert calls[0][1] != calls[1][1]
    assert dispatched.task.attempts[0].state == AttemptState.DEVICE_OFFLINE
    assert dispatched.task.attempts[1].state == AttemptState.SUCCEEDED
    assert dispatched.task.accepted_attempt_id == calls[1][1]
    assert registry.get("pc-01").status.value == "OFFLINE"


def test_late_result_from_offline_attempt_is_recorded_but_not_accepted():
    _, tasks, manager = _manager("pc-01", "phone-01")

    async def execute(task_id, attempt_id, device):
        if device.device_id == "pc-01":
            raise DeviceOfflineError("connection lost")
        return "accepted fallback"

    dispatched = asyncio.run(manager.submit("request", ["pc-01", "phone-01"], execute))
    offline_attempt = dispatched.task.attempts[0]

    disposition = tasks.record_result(offline_attempt.attempt_id, "late original")
    final = tasks.get(dispatched.task.task_id)

    assert disposition == ResultDisposition.STALE
    assert final.result == "accepted fallback"
    assert final.accepted_attempt_id == dispatched.task.attempts[1].attempt_id
    assert final.stale_results[-1].result == "late original"


def test_no_fallback_returns_controlled_failure():
    _, _, manager = _manager("pc-01")

    async def execute(task_id, attempt_id, device):
        raise DeviceOfflineError("connection lost")

    dispatched = asyncio.run(manager.submit("request", ["pc-01"], execute))

    assert dispatched.task.state == TaskState.FAILED
    assert dispatched.task.error_code == "NO_ELIGIBLE_FALLBACK"


def test_offline_devices_never_receive_new_assignments():
    registry, _, manager = _manager("pc-01", "phone-01")
    registry.mark_offline("pc-01")
    called: list[str] = []

    async def execute(task_id, attempt_id, device):
        called.append(device.device_id)
        return "ok"

    dispatched = asyncio.run(manager.submit("request", ["pc-01", "phone-01"], execute))

    assert dispatched.task.state == TaskState.SUCCEEDED
    assert called == ["phone-01"]


def test_external_offline_event_moves_active_attempt_to_retrying():
    registry, tasks, manager = _manager("pc-01")
    task = tasks.create("request")
    attempt = tasks.assign(task.task_id, "pc-01")
    tasks.mark_running(attempt.attempt_id)

    affected = manager.handle_device_offline("pc-01")

    assert affected == (attempt.attempt_id,)
    assert tasks.get(task.task_id).state == TaskState.RETRYING
    assert registry.get("pc-01").status.value == "OFFLINE"

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dragon_nest.dispatch import DeviceOfflineError, DispatchManager
from dragon_nest.models import Device, HealthState, TaskResult
from dragon_nest.registry import DeviceRegistry
from dragon_nest.tasks import TaskStore


def device(device_id: str, device_type: str) -> Device:
    return Device(
        device_id=device_id,
        display_name=device_id,
        device_type=device_type,
        platform="android" if device_type == "phone" else "windows",
        total_memory_mb=8192,
        health=HealthState(battery_pct=80, available_memory_mb=4096),
        models=(),
    )


async def main() -> None:
    registry = DeviceRegistry()
    registry.register(device("pc-01", "pc"))
    registry.register(device("phone-01", "phone"))
    tasks = TaskStore()
    dispatch = DispatchManager(registry, tasks)

    async def execute(task_id: str, attempt_id: str, selected: Device) -> TaskResult:
        print(f"dispatch {attempt_id} -> {selected.device_id}")
        if selected.device_id == "pc-01":
            raise DeviceOfflineError("simulated stream loss")
        return TaskResult(
            task_id=task_id,
            attempt_id=attempt_id,
            success=True,
            output_text="fallback completed",
            device_id=selected.device_id,
        )

    result = await dispatch.submit(
        "Demonstrate mid-job recovery",
        ["pc-01", "phone-01"],
        execute,
    )
    print(f"task {result.task.task_id}: {result.task.state}")
    for attempt in result.task.attempts:
        print(f"- {attempt.attempt_id}: {attempt.device_id} -> {attempt.state}")
    print(f"accepted result: {result.task.result.output_text}")


if __name__ == "__main__":
    asyncio.run(main())

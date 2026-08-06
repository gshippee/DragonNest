from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from .models import Device, HealthState


@dataclass(frozen=True)
class TelemetrySnapshot:
    health: HealthState
    active_task_ids: tuple[str, ...] = ()
    warm_model_ids: tuple[str, ...] = ()
    simulated_constraint: bool = False


class PlatformTelemetry(Protocol):
    def sample(self) -> TelemetrySnapshot: ...


class SystemTelemetry:
    """Best-effort host telemetry with explicit unknown values."""

    def __init__(self, device: Device):
        self.device = device

    def sample(self) -> TelemetrySnapshot:
        battery_pct, charging = _battery_state()
        return TelemetrySnapshot(
            health=HealthState(
                battery_pct=battery_pct,
                charging=charging,
                thermal_level=_thermal_level(),
                cpu_utilization=_cpu_utilization(),
                accelerator_utilization=-1,
                available_memory_mb=_available_memory_mb(),
                network_rtt_ms=-1,
                reachable=True,
            ),
            warm_model_ids=tuple(
                model.model_id for model in self.device.models if model.warm
            ),
        )


class SimulatedTelemetry:
    def __init__(self, source: PlatformTelemetry, **overrides: float | bool | int):
        self.source = source
        self.overrides = {key: value for key, value in overrides.items() if value is not None}

    def sample(self) -> TelemetrySnapshot:
        snapshot = self.source.sample()
        return replace(
            snapshot,
            health=replace(snapshot.health, **self.overrides),
            simulated_constraint=bool(self.overrides),
        )


def _available_memory_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical // (1024 * 1024))
        except (AttributeError, OSError):
            pass
    return 0


def _cpu_utilization() -> float:
    try:
        cpu_count = os.cpu_count() or 1
        return min(max(os.getloadavg()[0] / cpu_count, 0.0), 1.0)
    except (AttributeError, OSError):
        return -1


def _thermal_level() -> float:
    temperatures: list[float] = []
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            value = float(Path(path).read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        temperatures.append(value / 1000 if value > 1000 else value)
    if not temperatures:
        return -1
    return min(max(max(temperatures) / 100.0, 0.0), 1.0)


def _battery_state() -> tuple[float, bool]:
    for directory in glob.glob("/sys/class/power_supply/BAT*"):
        try:
            percentage = float(
                (Path(directory) / "capacity").read_text(encoding="ascii").strip()
            )
            status = (Path(directory) / "status").read_text(encoding="ascii").strip()
        except (OSError, ValueError):
            continue
        return percentage, status.lower() in {"charging", "full"}
    return -1, False

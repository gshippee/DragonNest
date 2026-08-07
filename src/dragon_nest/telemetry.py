from __future__ import annotations

import glob
import os
import re
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
        self._windows_cpu = _WindowsCpuSampler() if sys.platform == "win32" else None
        self._windows_gpu_npu = (
            _WindowsGpuNpuSampler() if sys.platform == "win32" else None
        )

    def sample(self) -> TelemetrySnapshot:
        battery_pct, charging = _battery_state()
        if self._windows_cpu is not None:
            cpu_utilization = self._windows_cpu.sample()
        else:
            cpu_utilization = _cpu_utilization()
        if self._windows_gpu_npu is not None:
            gpu_utilization, npu_utilization = self._windows_gpu_npu.sample()
        else:
            gpu_utilization, npu_utilization = -1.0, -1.0
        return TelemetrySnapshot(
            health=HealthState(
                battery_pct=battery_pct,
                charging=charging,
                thermal_level=_thermal_level(),
                cpu_utilization=cpu_utilization,
                accelerator_utilization=max(gpu_utilization, npu_utilization),
                gpu_utilization=gpu_utilization,
                npu_utilization=npu_utilization,
                available_memory_mb=_available_memory_mb(),
                network_rtt_ms=-1,
                reachable=True,
            ),
            warm_model_ids=tuple(
                model.model_id for model in self.device.models if model.warm
            ),
        )


class _WindowsCpuSampler:
    """Delta-based system-wide CPU utilization via ``GetSystemTimes``.

    No subprocess involved (unlike ``Get-Counter``, which costs ~300ms per
    call — too slow for a 2s heartbeat cadence). The first sample has no
    prior delta to compute from and returns -1; every sample after that is a
    real, cheap, system-wide number.
    """

    def __init__(self):
        self._prev: tuple[int, int, int] | None = None

    def sample(self) -> float:
        raw = self._read_system_times()
        if raw is None:
            return -1.0
        idle, kernel, user = raw
        if self._prev is None:
            self._prev = (idle, kernel, user)
            return -1.0
        prev_idle, prev_kernel, prev_user = self._prev
        self._prev = (idle, kernel, user)
        # lpKernelTime includes idle time on Windows; total = kernel + user.
        total_delta = (kernel - prev_kernel) + (user - prev_user)
        idle_delta = idle - prev_idle
        if total_delta <= 0:
            return 0.0
        return min(max((total_delta - idle_delta) / total_delta, 0.0), 1.0)

    @staticmethod
    def _read_system_times() -> tuple[int, int, int] | None:
        try:
            import ctypes
            from ctypes import wintypes

            class _FILETIME(ctypes.Structure):
                _fields_ = [
                    ("dwLowDateTime", wintypes.DWORD),
                    ("dwHighDateTime", wintypes.DWORD),
                ]

            idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
            ok = ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            )
            if not ok:
                return None

            def _as_int(ft: _FILETIME) -> int:
                return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

            return _as_int(idle), _as_int(kernel), _as_int(user)
        except (AttributeError, OSError):
            return None


class _WindowsGpuNpuSampler:
    """GPU and NPU utilization via the ``\\GPU Engine(*)\\Utilization
    Percentage`` performance counter.

    Windows has no dedicated "NPU Engine" counter category. On Copilot+
    hardware (e.g. Snapdragon X Elite) the NPU instead enumerates as its own
    DirectX compute-only adapter and is exposed through this same GPU Engine
    counter API, under a separate adapter LUID. We classify a LUID as the
    NPU when the ONLY engine type it exposes is "compute" (no 3d/video
    engines, which every real GPU has); anything else active is treated as
    the GPU. This is a heuristic based on observed WDDM compute-only-adapter
    behavior on this SoC class, not a documented Windows contract — verified
    against a real Snapdragon X Elite (Hexagon NPU, PNP class
    ComputeAccelerator) during development. Ambiguous/absent adapters fall
    back to -1 (unavailable) rather than guessing wrong.
    """

    _INSTANCE_RE = re.compile(
        r"^pid_\d+_luid_(?P<luid_hi>0x[0-9A-Fa-f]+)_(?P<luid_lo>0x[0-9A-Fa-f]+)_"
        r"phys_\d+_eng_(?P<eng>\d+)_engtype_(?P<engtype>.+)$"
    )
    _PDH_FMT_DOUBLE = 0x00000200

    def __init__(self):
        self._pdh = None
        self._query = None
        self._counter = None
        self._available = self._open()

    def _open(self) -> bool:
        try:
            import ctypes
            from ctypes import wintypes

            pdh = ctypes.WinDLL("pdh.dll")
            query = wintypes.HANDLE()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                return False
            counter = wintypes.HANDLE()
            path = r"\GPU Engine(*)\Utilization Percentage"
            if pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter)) != 0:
                pdh.PdhCloseQuery(query)
                return False
            pdh.PdhCollectQueryData(query)
            self._pdh, self._query, self._counter = pdh, query, counter
            return True
        except (AttributeError, OSError):
            return False

    def sample(self) -> tuple[float, float]:
        """Returns ``(gpu_utilization, npu_utilization)``, each -1 if unavailable."""
        if not self._available:
            return -1.0, -1.0
        try:
            return self._sample_unsafe()
        except (AttributeError, OSError):
            return -1.0, -1.0

    def _sample_unsafe(self) -> tuple[float, float]:
        import ctypes
        from ctypes import wintypes

        class _PdhFmtCounterValue(ctypes.Structure):
            _fields_ = [
                ("CStatus", wintypes.DWORD),
                ("_padding", wintypes.DWORD),
                ("doubleValue", ctypes.c_double),
            ]

        class _PdhFmtCounterValueItem(ctypes.Structure):
            _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhFmtCounterValue)]

        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return -1.0, -1.0

        buffer_size = wintypes.DWORD(0)
        item_count = wintypes.DWORD(0)
        self._pdh.PdhGetFormattedCounterArrayW(
            self._counter,
            self._PDH_FMT_DOUBLE,
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            None,
        )
        if buffer_size.value == 0:
            return -1.0, -1.0
        buf = ctypes.create_string_buffer(buffer_size.value)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            self._counter,
            self._PDH_FMT_DOUBLE,
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            buf,
        )
        if status != 0:
            return -1.0, -1.0
        items = ctypes.cast(buf, ctypes.POINTER(_PdhFmtCounterValueItem))

        per_engine: dict[tuple[str, str], float] = {}
        luid_engtypes: dict[str, set[str]] = {}
        for i in range(item_count.value):
            item = items[i]
            if item.FmtValue.CStatus != 0 or not item.szName:
                continue
            match = self._INSTANCE_RE.match(item.szName)
            if not match:
                continue
            luid = f"{match['luid_hi']}_{match['luid_lo']}"
            key = (luid, match["eng"])
            per_engine[key] = per_engine.get(key, 0.0) + item.FmtValue.doubleValue
            luid_engtypes.setdefault(luid, set()).add(match["engtype"].lower())

        gpu_util, npu_util = -1.0, -1.0
        for luid, engtypes in luid_engtypes.items():
            adapter_util = max(
                value for (candidate_luid, _eng), value in per_engine.items()
                if candidate_luid == luid
            )
            adapter_util = min(max(adapter_util / 100.0, 0.0), 1.0)
            if engtypes == {"compute"}:
                npu_util = max(npu_util, adapter_util)
            else:
                gpu_util = max(gpu_util, adapter_util)
        return gpu_util, npu_util


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
    if sys.platform == "win32":
        return _windows_available_memory_mb()
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


def _windows_available_memory_mb() -> int:
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys // (1024 * 1024))
    except OSError:
        pass
    return 0


def _windows_battery_state() -> tuple[float, bool]:
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_uint32),
            ("BatteryFullLifeTime", ctypes.c_uint32),
        ]

    status = SystemPowerStatus()
    try:
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return -1, False
    except OSError:
        return -1, False
    percentage = float(status.BatteryLifePercent)
    if percentage > 100:  # 255 means unknown
        return -1, status.ACLineStatus == 1
    return percentage, status.ACLineStatus == 1


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
    if sys.platform == "win32":
        return _windows_battery_state()
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

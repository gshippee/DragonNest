"""Local hardware probe for the Brain's own host machine.

``RestDeviceRegistration`` (see dashboard.py) normally auto-fills a device's
``HardwareInventory`` by calling ``GET {base_url}/info`` on a remote HTTP
endpoint the device already runs -- the same shape the Android agent reports
over gRPC via ``AndroidHardwareInventory.snapshot()`` at registration time.
That works only for devices that expose their own HTTP server.

A "local device" (the Brain's own host, or another machine reachable only
via the filesystem/process it's already running in) has no such endpoint to
probe over the network. This module gathers the same ``HardwareInventory``
fields in-process instead, so local device registration can auto-fill
hardware without a round-trip.

Field-gathering technique mirrors QUAD-Client's local hardware probe
(github.qualcomm.com/pavanr/QUAD-Client, src/quad_mcp_client/local/hardware.py
and _platform/*.py): Windows via one batched PowerShell CIM query, Linux via
/proc + /etc/os-release, macOS via sysctl. NPU presence is inferred from a
Qualcomm AI SDK install (QAIRT/QNN/SNPE env vars or a vendor default path)
plus an optional qnn-platform-validator probe -- never claimed without
evidence, matching the "available" | "not_probed" | "unavailable" vocabulary
``regimes.py`` already scores against.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+(?:\.\d+)?)")

# Chipset substring -> Hexagon NPU/DSP name. Trimmed to the NPU-only subset
# of QUAD-Client's capabilities.py table -- HardwareInventory has no
# gpu/tflops/tops fields, so only the substring -> npu_name mapping is
# carried over.
_NPU_NAMES: tuple[tuple[str, str], ...] = (
    ("x1e", "Hexagon NPU (v73)"),
    ("snapdragon x elite", "Hexagon NPU (v73)"),
    ("oryon", "Hexagon NPU (v73)"),
    ("sm8750", "Hexagon NPU (v73)"),
    ("snapdragon 8 elite", "Hexagon NPU (v73)"),
    ("sm8650", "Hexagon NPU (v69)"),
    ("snapdragon 8 gen 3", "Hexagon NPU (v69)"),
    ("qcs2210", "Hexagon DSP (v66)"),
    ("qcs9075", "Hexagon NPU (v73)"),
    ("qcs9100", "Hexagon NPU (v73)"),
    ("qcs6490", "Hexagon NPU (v69)"),
    ("qcs8250", "Hexagon NPU (v68)"),
)


def _npu_name_for(chipset: str) -> str:
    needle = chipset.lower()
    for substr, name in _NPU_NAMES:
        if substr in needle:
            return name
    return ""


def _soc_info(chipset: str) -> tuple[str, str]:
    """Only claim a Qualcomm SoC manufacturer/model when the chipset string
    actually says so -- desktop Intel/AMD/Apple CPUs stay blank rather than
    guessing."""
    needle = chipset.lower()
    if "qualcomm" in needle or "snapdragon" in needle or "oryon" in needle or re.match(r"qcs\d", needle):
        return "Qualcomm", chipset
    return "", ""


def _cpu_abi() -> tuple[str, ...]:
    arch = (platform.machine() or "").lower()
    if arch in {"amd64", "x86_64"}:
        return ("x86_64",)
    if arch in {"arm64", "aarch64"}:
        return ("arm64-v8a",)
    if arch in {"x86", "i386", "i686"}:
        return ("x86",)
    if arch.startswith("arm"):
        return ("armeabi-v7a",)
    return (arch,) if arch else ()


def _version_from_path(path: Path) -> str | None:
    m = _VERSION_RE.search(str(path))
    return m.group(1) if m else None


def _find_sdk() -> tuple[str | None, str | None]:
    """``(sdk_path, sdk_version)`` from env vars, else a vendor default path.

    Mirrors QUAD-Client's ``sdk_scan.find_sdk`` search order (env vars, then
    a vendor install root) minus the "sibling QUAD repo" step, which has no
    equivalent here.
    """
    for env_var in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT"):
        val = os.environ.get(env_var, "").strip()
        if not val:
            continue
        path = Path(val)
        if path.exists():
            return str(path), _version_from_path(path) or path.name

    vendor_root = (
        Path(r"C:\Qualcomm\AIStack\QAIRT")
        if sys.platform == "win32"
        else Path("/opt/qcom/aistack/qairt")
    )
    try:
        children = [p for p in vendor_root.iterdir() if p.is_dir()] if vendor_root.is_dir() else []
    except OSError:
        children = []
    best: tuple[tuple[int, ...], Path] | None = None
    for child in children:
        m = _VERSION_RE.search(child.name)
        if not m:
            continue
        ver = tuple(int(part) for part in m.group(1).split("."))
        if best is None or ver > best[0]:
            best = (ver, child)
    if best is not None:
        return str(best[1]), _version_from_path(best[1]) or best[1].name
    return None, None


def _npu_status(sdk_path: str | None) -> str:
    if sdk_path is None:
        return "unavailable"
    validator = shutil.which("qnn-platform-validator")
    if validator is None:
        return "not_probed"
    try:
        result = subprocess.run(
            [validator, "--backend", "all"], capture_output=True, text=True, timeout=20.0
        )
    except (OSError, subprocess.TimeoutExpired):
        return "not_probed"
    out = (result.stdout or "") + (result.stderr or "")
    return "available" if ("HTP" in out or "DSP" in out) else "unavailable"


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _disk_usage_mb(path: str) -> tuple[int, int]:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return 0, 0
    return usage.total // (1024 * 1024), usage.free // (1024 * 1024)


def _windows_probe() -> dict[str, Any]:
    query = (
        "$cs = Get-CimInstance Win32_ComputerSystem | "
        "Select-Object -Property Manufacturer,Model,TotalPhysicalMemory; "
        "$cpu = Get-CimInstance Win32_Processor | Select-Object -Property Name,NumberOfCores; "
        "$osq = Get-CimInstance Win32_OperatingSystem | Select-Object -Property Caption; "
        "$disk = Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='$($env:SystemDrive)'\" | "
        "Select-Object -Property Size,FreeSpace; "
        "@{ cpu = $cpu; cs = $cs; os = $osq; disk = $disk } | ConvertTo-Json -Depth 4 -Compress"
    )
    data: dict[str, Any] = {}
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        data = {}

    def _first(value: Any) -> dict[str, Any]:
        if isinstance(value, list):
            return value[0] if value else {}
        return value or {}

    cpu, cs, osq, disk = _first(data.get("cpu")), _first(data.get("cs")), _first(data.get("os")), _first(data.get("disk"))
    total_storage_mb = int(disk.get("Size") or 0) // (1024 * 1024)
    available_storage_mb = int(disk.get("FreeSpace") or 0) // (1024 * 1024)
    if total_storage_mb == 0:
        total_storage_mb, available_storage_mb = _disk_usage_mb(os.environ.get("SystemDrive", "C:") + "\\")

    return {
        "manufacturer": (cs.get("Manufacturer") or "").strip(),
        "model": (cs.get("Model") or "").strip(),
        "device": platform.node(),
        "os_version": (osq.get("Caption") or "").strip() or f"Windows {platform.release()}",
        "chipset": (cpu.get("Name") or platform.processor() or "").strip(),
        "cpu_core_count": int(cpu.get("NumberOfCores") or (os.cpu_count() or 0)),
        "total_storage_mb": total_storage_mb,
        "available_storage_mb": available_storage_mb,
        "total_memory_mb": int(cs.get("TotalPhysicalMemory") or 0) // (1024 * 1024),
    }


def _linux_probe() -> dict[str, Any]:
    cpuinfo = _read_text("/proc/cpuinfo")
    chipset = ""
    cores = 0
    for line in cpuinfo.splitlines():
        line = line.strip()
        if line.startswith("processor"):
            cores += 1
        elif line.startswith("Hardware") and not chipset:
            chipset = line.partition(":")[2].strip()
        elif line.startswith("model name") and not chipset:
            chipset = line.partition(":")[2].strip()

    os_release = _read_text("/etc/os-release")
    m = re.search(r'^PRETTY_NAME\s*=\s*"?([^"\n]+)', os_release, re.MULTILINE)
    os_version = m.group(1).strip() if m else (f"Linux {platform.release()}" if platform.release() else "Linux")

    mem_total_kb = 0
    mm = re.search(r"^MemTotal:\s+(\d+)\s+kB", _read_text("/proc/meminfo"), re.MULTILINE)
    if mm:
        mem_total_kb = int(mm.group(1))

    total_storage_mb, available_storage_mb = _disk_usage_mb("/")

    return {
        "manufacturer": _read_text("/sys/class/dmi/id/sys_vendor").strip(),
        "model": _read_text("/sys/class/dmi/id/product_name").strip(),
        "device": platform.node(),
        "os_version": os_version,
        "chipset": chipset or platform.machine() or "unknown",
        "cpu_core_count": cores or (os.cpu_count() or 0),
        "total_storage_mb": total_storage_mb,
        "available_storage_mb": available_storage_mb,
        "total_memory_mb": mem_total_kb // 1024,
    }


def _macos_probe() -> dict[str, Any]:
    def _sysctl(name: str) -> str:
        try:
            result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    raw_mem = _sysctl("hw.memsize")
    mem_bytes = int(raw_mem) if raw_mem.isdigit() else 0
    total_storage_mb, available_storage_mb = _disk_usage_mb("/")
    mac_ver = platform.mac_ver()[0]

    return {
        "manufacturer": "Apple",
        "model": _sysctl("hw.model"),
        "device": platform.node(),
        "os_version": f"macOS {mac_ver}" if mac_ver else "macOS",
        "chipset": _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        "cpu_core_count": os.cpu_count() or 0,
        "total_storage_mb": total_storage_mb,
        "available_storage_mb": available_storage_mb,
        "total_memory_mb": mem_bytes // (1024 * 1024),
    }


def probe_local_hardware() -> dict[str, Any]:
    """Probe this process's own host machine.

    Returns ``{"hardware": {...}, "platform": ..., "total_memory_mb": ...,
    "display_name": ...}``. ``hardware`` matches
    ``dragon_nest.models.HardwareInventory`` field-for-field so it can be
    passed straight into ``HardwareInventoryPayload(**info["hardware"])``.
    """
    if sys.platform.startswith("win"):
        raw, dn_platform = _windows_probe(), "windows"
    elif sys.platform == "darwin":
        raw, dn_platform = _macos_probe(), "macos"
    else:
        raw, dn_platform = _linux_probe(), "linux"

    chipset = str(raw.get("chipset") or "")
    sdk_path, sdk_version = _find_sdk()
    soc_manufacturer, soc_model = _soc_info(chipset)

    hardware = {
        "manufacturer": raw.get("manufacturer", ""),
        "model": raw.get("model") or chipset,
        "device": raw.get("device", ""),
        "os_version": raw.get("os_version", ""),
        "api_level": 0,
        "soc_manufacturer": soc_manufacturer,
        "soc_model": soc_model,
        "cpu_abis": list(_cpu_abi()),
        "cpu_core_count": int(raw.get("cpu_core_count") or 0),
        "total_storage_mb": int(raw.get("total_storage_mb") or 0),
        "available_storage_mb": int(raw.get("available_storage_mb") or 0),
        "npu_status": _npu_status(sdk_path),
        "npu_name": _npu_name_for(chipset),
        "qnn_runtime_version": sdk_version or "",
    }
    return {
        "hardware": hardware,
        "platform": dn_platform,
        "total_memory_mb": int(raw.get("total_memory_mb") or 0),
        "display_name": raw.get("device") or platform.node(),
    }

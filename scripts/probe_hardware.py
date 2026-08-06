from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dragon_nest.artifacts import ArtifactRegistry  # noqa: E402
from dragon_nest.models import (  # noqa: E402
    ExecutionMode,
    ExecutionPlan,
    PlannedTask,
)
from dragon_nest.runtime.hardware_adapter import (  # noqa: E402
    HardwareRuntimeAdapter,
)
from dragon_nest.telemetry import SystemTelemetry  # noqa: E402
from dragon_nest.config import load_device  # noqa: E402


SAFE_ENV_NAMES = (
    "QAIRT_ROOT",
    "QNN_SDK_ROOT",
    "GENIE_DIR",
    "ANDROID_HOME",
    "ANDROID_SDK_ROOT",
)


async def run(args: argparse.Namespace) -> dict[str, object]:
    registry = ArtifactRegistry.from_yaml(args.manifest)
    device = load_device(args.fabric, args.device_id)
    telemetry = SystemTelemetry(device)
    adapter = HardwareRuntimeAdapter(
        registry,
        compatibility_key=args.compatibility_key,
        runtime_name=args.runtime_name,
        runtime_version=args.runtime_version,
        accelerator_available=args.accelerator_available,
        telemetry=telemetry,
        artifact_store=args.artifact_store,
    )
    artifact = registry.get(args.model_id)
    record: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_level": "verified locally without hardware",
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "tooling": {
            command: shutil.which(command)
            for command in (
                "qnn-net-run",
                "qnn-context-binary-generator",
                "genie-t2t-run.exe",
                "geniex",
                "adb",
            )
        },
        "configured_environment_names": [
            name for name in SAFE_ENV_NAMES if os.environ.get(name)
        ],
        "adapter_capabilities": asdict(adapter.capabilities()),
        "artifact": {
            "artifact_id": artifact.artifact_id,
            "model_id": artifact.model_id,
            "target_compatibility_class": artifact.target_compatibility_class,
            "steering_mode": artifact.steering_mode.value,
            "verification_status_before_probe": artifact.verification_status,
        },
    }
    if platform.system() == "Windows":
        record["windows_hardware_inventory"] = _windows_hardware_inventory()
    try:
        path = adapter.validate_artifact(artifact)
        record["artifact_validation"] = {
            "passed": True,
            "path_sha256": _path_fingerprint(path),
            "declared_checksum": artifact.checksum,
            "size_bytes": _size(path),
        }
        loaded = adapter.load_artifact(artifact.artifact_id)
        record["load"] = asdict(loaded)
    except Exception as exc:
        record["artifact_validation"] = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return record

    if not args.execute:
        return record

    plan = ExecutionPlan(
        task_id=f"physical-smoke-{int(time.time())}",
        execution_mode=ExecutionMode.SINGLE,
        request_text=args.prompt,
        tasks=(
            PlannedTask(
                shard_id="smoke-0",
                request_text=args.prompt,
                selected_device_id=args.device_id,
                selected_model_id=args.model_id,
            ),
        ),
    )
    before = adapter.health()
    result = await adapter.execute(plan, attempt_id="physical-smoke-attempt")
    after = adapter.health()
    record["execution"] = {
        "success": result.success,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "latency_ms": result.latency_ms,
        "output_sha256": hashlib.sha256(
            result.output_text.encode("utf-8")
        ).hexdigest(),
        "output_preview": result.output_text[:240],
        "metrics": asdict(result.metrics) if result.metrics else None,
        "available_memory_mb_before": before.telemetry.health.available_memory_mb,
        "available_memory_mb_after": after.telemetry.health.available_memory_mb,
        "thermal_before": before.telemetry.health.thermal_level,
        "thermal_after": after.telemetry.health.thermal_level,
    }
    if result.success and args.accelerator_available:
        record["evidence_level"] = "verified on physical hardware"
    return record


def _path_fingerprint(path: Path) -> str:
    resolved = str(path.resolve()).encode("utf-8")
    return hashlib.sha256(resolved).hexdigest()


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _windows_hardware_inventory() -> dict[str, object]:
    script = r"""
$system = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,SystemType,TotalPhysicalMemory
$processors = @(Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,Architecture,NumberOfCores,NumberOfLogicalProcessors)
$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture
$accelerators = @(Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.FriendlyName -match 'Qualcomm|Hexagon|NPU|Neural' -or $_.Manufacturer -match 'Qualcomm' } | Select-Object Status,Class,FriendlyName,Manufacturer)
@{ system=$system; processors=$processors; os=$os; accelerator_devices=$accelerators } | ConvertTo-Json -Depth 5 -Compress
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"probe_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a secret-free DragonNest physical-runtime proof record."
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/model-artifacts.yaml")
    parser.add_argument("--fabric", type=Path, default=ROOT / "configs/hardware-fabric.yaml")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--compatibility-key", required=True)
    parser.add_argument("--runtime-name", required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--accelerator-available", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: DragonNest physical NPU smoke test passed.",
    )
    parser.add_argument(
        "--artifact-store",
        type=Path,
        default=Path.home() / ".dragonnest" / "artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("TEMP", "/tmp")) / "dragonnest-hardware-proof.json",
    )
    args = parser.parse_args()
    record = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Proof written to {args.output}")
    validation = record.get("artifact_validation", {})
    execution = record.get("execution")
    passed = bool(validation.get("passed")) and (
        execution is None or bool(execution.get("success"))
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

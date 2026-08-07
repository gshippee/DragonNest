from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx


EXPECTED_OUTPUT = "DRAGONNEST_CROSSHOST_OK"
REQUEST_TEXT = f"Reply with exactly: {EXPECTED_OUTPUT}"
EXPECTED_ARTIFACT_ID = "qwen3-4b-w4a16-xelite-v73-qairt248"
ELIGIBLE_STATUSES = frozenset({"HEALTHY", "DEGRADED"})
TERMINAL_TASK_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
FORBIDDEN_PROOF_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "bundle_path",
        "credential",
        "enrollment_token",
        "model_path",
        "output_text",
        "password",
        "token",
    }
)


class ValidationFailure(RuntimeError):
    """A failed physical-proof precondition or result assertion."""


@dataclass(frozen=True)
class HarnessConfig:
    brain_http: str = "http://127.0.0.1:8080"
    device_id: str = "pc-01"
    model_id: str = "qwen3-4b-genie"
    artifact_id: str = EXPECTED_ARTIFACT_ID
    runs: int = 1
    timeout_seconds: float = 240.0
    poll_interval_seconds: float = 0.25
    idle_stabilization_seconds: float = 3.0
    recovery_seconds: float = 3.0
    calibrate_memory: bool = False
    output: Path = Path(tempfile.gettempdir()) / "dragonnest-crosshost-xelite.json"

    def __post_init__(self) -> None:
        if self.runs <= 0:
            raise ValueError("runs must be positive")
        if not 0 < self.timeout_seconds <= 300:
            raise ValueError("timeout must be between 0 and 300 seconds")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll interval must be positive")
        if self.idle_stabilization_seconds < 0 or self.recovery_seconds < 0:
            raise ValueError("stabilization/recovery windows cannot be negative")


def normalize_brain_url(raw: str) -> str:
    parsed = urlsplit(raw.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--brain-http must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("credentials must not be embedded in --brain-http")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def validate_worker_snapshot(
    devices: Sequence[Mapping[str, Any]],
    *,
    device_id: str,
    model_id: str,
    artifact_id: str = EXPECTED_ARTIFACT_ID,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    device = next((item for item in devices if item.get("device_id") == device_id), None)
    if device is None:
        seen = sorted(str(item.get("device_id", "<missing>")) for item in devices)
        raise ValidationFailure(
            f"worker {device_id!r} is absent; connected device ids: {seen or ['none']}. "
            "Start scripts/run_xelite_worker.ps1 on the X Elite with this "
            "desktop Brain's LAN address."
        )
    if device.get("transport") != "grpc_stream":
        raise ValidationFailure(
            f"{device_id} uses transport {device.get('transport')!r}, not the "
            "physical gRPC Device Agent transport"
        )
    if not device.get("connected"):
        raise ValidationFailure(f"{device_id} exists but its gRPC stream is disconnected")
    if device.get("status") not in ELIGIBLE_STATUSES:
        raise ValidationFailure(
            f"{device_id} status is {device.get('status')!r}; expected one of "
            f"{sorted(ELIGIBLE_STATUSES)}"
        )
    health = device.get("health") or {}
    if not health.get("reachable"):
        raise ValidationFailure(f"{device_id} reports reachable=false")
    available_memory = int(health.get("available_memory_mb") or 0)
    if available_memory <= 0:
        raise ValidationFailure(
            f"{device_id} has unknown/non-positive available_memory_mb "
            f"({available_memory})"
        )

    models = device.get("models") or []
    model = next((item for item in models if item.get("model_id") == model_id), None)
    if model is None:
        advertised = sorted(str(item.get("model_id", "<missing>")) for item in models)
        raise ValidationFailure(
            f"{device_id} does not advertise {model_id!r}; advertised models: "
            f"{advertised or ['none']}"
        )
    if model.get("runtime") != "genie":
        raise ValidationFailure(
            f"{device_id}/{model_id} advertises runtime {model.get('runtime')!r}; "
            "real X Elite proof requires genie"
        )
    accelerators = {str(value).lower() for value in model.get("accelerators") or []}
    if "htp" not in accelerators:
        raise ValidationFailure(
            f"{device_id}/{model_id} does not advertise htp; got {sorted(accelerators)}"
        )
    if model.get("artifact_id") != artifact_id:
        raise ValidationFailure(
            f"{device_id}/{model_id} artifact_id is {model.get('artifact_id')!r}; "
            f"expected {artifact_id!r}"
        )
    if model.get("warm") is not False:
        raise ValidationFailure(
            f"{device_id}/{model_id} must advertise installed/cold (warm=false); "
            f"got warm={model.get('warm')!r}"
        )
    steering_modes = {str(value) for value in model.get("steering_modes") or []}
    if (
        model.get("supports_steering")
        or model.get("steering_vectors")
        or "runtime_vector" in steering_modes
    ):
        raise ValidationFailure(
            f"{device_id}/{model_id} falsely claims runtime steering support"
        )

    deployments = {
        str(item.get("artifact_id")): str(item.get("state"))
        for item in device.get("deployments") or []
    }
    if deployments.get(model_id) != "installed":
        raise ValidationFailure(
            f"{device_id}/{model_id} deployment must be installed (cold); got "
            f"{deployments.get(model_id, 'absent')!r}"
        )
    return device, model


def validate_execution_result(
    submission: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    device_id: str,
    model_id: str,
    expected_output: str = EXPECTED_OUTPUT,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    chosen = (submission.get("route_plan") or {}).get("chosen")
    if not chosen:
        raise ValidationFailure("scheduler did not choose a route")
    if chosen.get("device_id") != device_id:
        raise ValidationFailure(
            f"scheduler chose device {chosen.get('device_id')!r}; expected {device_id!r}"
        )
    if chosen.get("artifact_id") != model_id:
        raise ValidationFailure(
            f"scheduler chose model/catalog artifact {chosen.get('artifact_id')!r}; "
            f"expected {model_id!r}"
        )
    if chosen.get("runtime") != "genie":
        raise ValidationFailure(
            f"scheduler chose runtime {chosen.get('runtime')!r}; expected 'genie'"
        )
    if chosen.get("deployment_state") != "installed":
        raise ValidationFailure(
            f"scheduler route reports deployment {chosen.get('deployment_state')!r}; "
            "expected installed/cold"
        )
    if chosen.get("realization_mode") != "none":
        raise ValidationFailure(
            f"scheduler route used realization {chosen.get('realization_mode')!r}; "
            "real X Elite proof must be unsteered"
        )
    if submission.get("device_id") != device_id:
        raise ValidationFailure(
            f"result came from device {submission.get('device_id')!r}; expected {device_id!r}"
        )
    if submission.get("model_id") != model_id:
        raise ValidationFailure(
            f"result model is {submission.get('model_id')!r}; expected {model_id!r}"
        )
    if not submission.get("success"):
        raise ValidationFailure(
            f"task failed: {submission.get('error_code') or 'unknown'} "
            f"{submission.get('error_message') or ''}".strip()
        )
    steering = submission.get("steering") or {}
    if steering.get("enabled") or steering.get("vector_id"):
        raise ValidationFailure("Brain response falsely reports steering on X Elite")

    result = task.get("result") or {}
    metrics = result.get("metrics") or {}
    if task.get("state") != "SUCCEEDED":
        raise ValidationFailure(
            f"task record state is {task.get('state')!r}; expected 'SUCCEEDED'"
        )
    if not result.get("success"):
        raise ValidationFailure("task record does not contain a successful result")
    if result.get("device_id") != device_id:
        raise ValidationFailure(
            f"task record result device is {result.get('device_id')!r}; "
            f"expected {device_id!r}"
        )
    output = str(result.get("output_text") or "").strip()
    if not output:
        raise ValidationFailure("physical worker returned an empty output")
    if output != expected_output:
        raise ValidationFailure(
            f"physical worker output did not match the deterministic marker; "
            f"sha256={hashlib.sha256(output.encode('utf-8')).hexdigest()}"
        )
    if metrics.get("model_id") != model_id:
        raise ValidationFailure(
            f"execution metrics model_id is {metrics.get('model_id')!r}; "
            f"expected {model_id!r}"
        )
    if metrics.get("runtime_name") != "genie":
        raise ValidationFailure(
            f"execution metrics runtime is {metrics.get('runtime_name')!r}; "
            "mock/other runtimes are forbidden"
        )
    if str(metrics.get("accelerator", "")).lower() != "htp":
        raise ValidationFailure(
            f"execution metrics accelerator is {metrics.get('accelerator')!r}; "
            "expected 'htp'"
        )
    if int(metrics.get("execution_latency_ms") or 0) <= 0:
        raise ValidationFailure("execution metrics did not report a positive latency")
    attempt_id = str(task.get("accepted_attempt_id") or "")
    if not attempt_id or attempt_id != str(submission.get("accepted_attempt_id") or ""):
        raise ValidationFailure("accepted attempt id is missing or inconsistent")
    return chosen, metrics


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one run is required")
    peak_deltas = [int(run["peak_available_memory_delta_mb"]) for run in runs]
    execution_latencies = [int(run["execution_latency_ms"]) for run in runs]
    return {
        "median_peak_available_memory_delta_mb": statistics.median(peak_deltas),
        "max_peak_available_memory_delta_mb": max(peak_deltas),
        "median_execution_latency_ms": statistics.median(execution_latencies),
        "max_execution_latency_ms": max(execution_latencies),
        "brain_telemetry_memory_capture_reliable": all(
            bool(run["telemetry_capture_reliable"]) for run in runs
        ),
    }


def save_secret_free_proof(proof: Mapping[str, Any], output: Path) -> None:
    _assert_secret_free(proof)
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _assert_secret_free(value: Any, path: str = "proof") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_PROOF_KEYS or normalized.endswith("_token"):
                raise ValidationFailure(
                    f"refusing to write proof containing sensitive key {path}.{key}"
                )
            _assert_secret_free(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_secret_free(item, f"{path}[{index}]")


class RemoteXEliteValidator:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.brain_http = normalize_brain_url(config.brain_http)
        self._timeout = httpx.Timeout(config.timeout_seconds + 10.0)

    async def run(self) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.brain_http, timeout=self._timeout
        ) as client:
            worker, model = await self._wait_for_worker(client)
            run_records = []
            for index in range(1, self.config.runs + 1):
                print(
                    f"Run {index}/{self.config.runs}: waiting for idle "
                    f"{self.config.device_id}..."
                )
                run_records.append(await self._run_once(client, index))
            summary = summarize_runs(run_records)
            proof = {
                "schema_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "claim": (
                    "desktop Brain -> scheduler -> LAN gRPC -> X Elite Agent -> "
                    "Qwen3-4B Genie/HTP -> desktop Brain"
                ),
                "brain_http": self.brain_http,
                "request_sha256": hashlib.sha256(
                    REQUEST_TEXT.encode("utf-8")
                ).hexdigest(),
                "expected_output_sha256": hashlib.sha256(
                    EXPECTED_OUTPUT.encode("utf-8")
                ).hexdigest(),
                "expected_device_id": self.config.device_id,
                "expected_model_id": self.config.model_id,
                "expected_artifact_id": self.config.artifact_id,
                "memory_calibration_requested": self.config.calibrate_memory,
                "worker": _sanitized_worker(worker, model),
                "runs": run_records,
                "summary": summary,
            }
            if not summary["brain_telemetry_memory_capture_reliable"]:
                proof["memory_calibration_note"] = (
                    "Brain heartbeat telemetry did not expose enough distinct "
                    "in-task updates to guarantee the true peak. Use the optional "
                    "scripts/sample_xelite_memory.ps1 helper in a separate X Elite "
                    "terminal during a rerun; do not change the catalog estimate "
                    "from this proof alone."
                )
            save_secret_free_proof(proof, self.config.output)
            return proof

    async def _wait_for_worker(
        self, client: httpx.AsyncClient
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        deadline = time.monotonic() + self.config.timeout_seconds
        last_error = "worker has not appeared"
        while time.monotonic() < deadline:
            try:
                devices = await self._devices(client)
                return validate_worker_snapshot(
                    devices,
                    device_id=self.config.device_id,
                    model_id=self.config.model_id,
                    artifact_id=self.config.artifact_id,
                )
            except (ValidationFailure, httpx.HTTPError) as exc:
                last_error = str(exc)
                await asyncio.sleep(min(1.0, self.config.poll_interval_seconds * 2))
        raise ValidationFailure(
            f"timed out after {self.config.timeout_seconds:.0f}s waiting for a "
            f"routable physical worker: {last_error}"
        )

    async def _wait_for_idle(
        self, client: httpx.AsyncClient, stabilization_seconds: float
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        deadline = time.monotonic() + self.config.timeout_seconds
        idle_since: float | None = None
        while time.monotonic() < deadline:
            device, model = validate_worker_snapshot(
                await self._devices(client),
                device_id=self.config.device_id,
                model_id=self.config.model_id,
                artifact_id=self.config.artifact_id,
            )
            if device.get("active_tasks"):
                idle_since = None
            else:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since >= stabilization_seconds:
                    return device, model
            await asyncio.sleep(self.config.poll_interval_seconds)
        raise ValidationFailure(
            f"{self.config.device_id} did not remain idle for "
            f"{stabilization_seconds:.1f}s before timeout"
        )

    async def _run_once(
        self, client: httpx.AsyncClient, run_number: int
    ) -> dict[str, Any]:
        stabilization = (
            self.config.idle_stabilization_seconds
            if self.config.calibrate_memory
            else 0.0
        )
        before_device, before_model = await self._wait_for_idle(
            client, stabilization
        )
        available_before = _available_memory(before_device)
        heartbeat_before = before_device.get("last_heartbeat")
        samples: list[Mapping[str, Any]] = []

        request_started = time.perf_counter()
        submit = asyncio.create_task(
            client.post(
                "/api/behavior-tasks",
                json={
                    "request_text": REQUEST_TEXT,
                    "base_model_family": "qwen3",
                    "estimated_input_tokens": 32,
                    "estimated_output_tokens": 16,
                    "privacy": "trusted_fabric",
                    "latency_preference": "interactive",
                    "timeout_ms": round(self.config.timeout_seconds * 1000),
                },
            )
        )
        while not submit.done():
            device, _ = validate_worker_snapshot(
                await self._devices(client),
                device_id=self.config.device_id,
                model_id=self.config.model_id,
                artifact_id=self.config.artifact_id,
            )
            samples.append(_telemetry_sample(device))
            await asyncio.sleep(self.config.poll_interval_seconds)
        response = await submit
        e2e_latency_ms = round((time.perf_counter() - request_started) * 1000)
        response.raise_for_status()
        submission = response.json()
        task_id = str(submission.get("task_id") or "")
        if not task_id:
            raise ValidationFailure("Brain response did not include a task id")
        task = await self._task(client, task_id)
        chosen, metrics = validate_execution_result(
            submission,
            task,
            device_id=self.config.device_id,
            model_id=self.config.model_id,
        )

        recovery = self.config.recovery_seconds if self.config.calibrate_memory else 0.0
        after_device, after_model = await self._wait_for_idle(client, recovery)
        available_after = _available_memory(after_device)
        during_memory = [int(sample["available_memory_mb"]) for sample in samples]
        minimum_available = min([available_before, *during_memory])
        heartbeat_updates = {
            sample["last_heartbeat"]
            for sample in samples
            if sample.get("last_heartbeat") != heartbeat_before
        }
        active_seen = any(bool(sample.get("active_tasks")) for sample in samples)
        telemetry_reliable = active_seen and len(heartbeat_updates) >= 2
        output = str((task.get("result") or {}).get("output_text") or "").strip()

        print(
            f"Run {run_number}: {task_id} -> {self.config.device_id} / "
            f"{self.config.model_id}, genie/htp, "
            f"{metrics.get('execution_latency_ms')} ms server, "
            f"{e2e_latency_ms} ms E2E"
        )
        return {
            "run": run_number,
            "task_id": task_id,
            "attempt_id": str(task.get("accepted_attempt_id")),
            "chosen_route": {
                "device_id": chosen.get("device_id"),
                "catalog_artifact_id": chosen.get("artifact_id"),
                "runtime": chosen.get("runtime"),
                "deployment_state": chosen.get("deployment_state"),
                "realization_mode": chosen.get("realization_mode"),
            },
            "scheduler_explanation": list(
                (submission.get("route_plan") or {}).get("explanation") or []
            ),
            "device_id": submission.get("device_id"),
            "model_id": metrics.get("model_id"),
            "artifact_id": before_model.get("artifact_id"),
            "runtime_name": metrics.get("runtime_name"),
            "runtime_version": metrics.get("runtime_version"),
            "accelerator": metrics.get("accelerator"),
            "success": True,
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "available_before_mb": available_before,
            "minimum_available_mb": minimum_available,
            "available_after_mb": available_after,
            "peak_available_memory_delta_mb": max(
                0, available_before - minimum_available
            ),
            "execution_latency_ms": int(metrics.get("execution_latency_ms") or 0),
            "e2e_latency_ms": e2e_latency_ms,
            "telemetry_sample_count": len(samples),
            "distinct_in_task_heartbeat_updates": len(heartbeat_updates),
            "active_task_observed": active_seen,
            "telemetry_capture_reliable": telemetry_reliable,
            "telemetry_before": _sanitized_telemetry(before_device),
            "telemetry_after": _sanitized_telemetry(after_device),
            "worker_capability_stable": before_model == after_model,
        }

    async def _devices(self, client: httpx.AsyncClient) -> list[Mapping[str, Any]]:
        response = await client.get("/api/devices")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValidationFailure("Brain /api/devices returned a non-list payload")
        return payload

    async def _task(
        self, client: httpx.AsyncClient, task_id: str
    ) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.config.timeout_seconds
        while time.monotonic() < deadline:
            response = await client.get(f"/api/tasks/{task_id}")
            response.raise_for_status()
            task = response.json()
            if task.get("state") in TERMINAL_TASK_STATES:
                return task
            await asyncio.sleep(self.config.poll_interval_seconds)
        raise ValidationFailure(f"task {task_id} did not reach a terminal state")


def _available_memory(device: Mapping[str, Any]) -> int:
    return int((device.get("health") or {}).get("available_memory_mb") or 0)


def _telemetry_sample(device: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "last_heartbeat": device.get("last_heartbeat"),
        "available_memory_mb": _available_memory(device),
        "active_tasks": list(device.get("active_tasks") or []),
    }


def _sanitized_telemetry(device: Mapping[str, Any]) -> dict[str, Any]:
    health = device.get("health") or {}
    return {
        "available_memory_mb": int(health.get("available_memory_mb") or 0),
        "thermal_level": health.get("thermal_level"),
        "battery_pct": health.get("battery_pct"),
        "charging": health.get("charging"),
        "accelerator_utilization": health.get("accelerator_utilization"),
        "npu_utilization": health.get("npu_utilization"),
        "network_rtt_ms": health.get("network_rtt_ms"),
        "reachable": health.get("reachable"),
        "status": device.get("status"),
    }


def _sanitized_worker(
    device: Mapping[str, Any], model: Mapping[str, Any]
) -> dict[str, Any]:
    hardware = device.get("hardware") or {}
    return {
        "device_id": device.get("device_id"),
        "transport": device.get("transport"),
        "manufacturer": hardware.get("manufacturer"),
        "model": hardware.get("model"),
        "soc_model": hardware.get("soc_model"),
        "os_version": hardware.get("os_version"),
        "cpu_abis": hardware.get("cpu_abis"),
        "model_id": model.get("model_id"),
        "artifact_id": model.get("artifact_id"),
        "runtime": model.get("runtime"),
        "runtime_version": model.get("runtime_version"),
        "accelerators": model.get("accelerators"),
        "warm": model.get("warm"),
        "supports_steering": model.get("supports_steering"),
        "steering_modes": model.get("steering_modes"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a true desktop-Brain to physical X Elite Genie/HTP round trip"
        )
    )
    parser.add_argument("--brain-http", default="http://127.0.0.1:8080")
    parser.add_argument("--device-id", default="pc-01")
    parser.add_argument("--model-id", default="qwen3-4b-genie")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=240.0, help="seconds")
    parser.add_argument("--poll-interval", type=float, default=0.25, help="seconds")
    parser.add_argument("--calibrate-memory", action="store_true")
    parser.add_argument("--idle-stabilization", type=float, default=3.0)
    parser.add_argument("--recovery", type=float, default=3.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "dragonnest-crosshost-xelite.json",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    config = HarnessConfig(
        brain_http=args.brain_http,
        device_id=args.device_id,
        model_id=args.model_id,
        runs=args.runs,
        timeout_seconds=args.timeout,
        poll_interval_seconds=args.poll_interval,
        idle_stabilization_seconds=args.idle_stabilization,
        recovery_seconds=args.recovery,
        calibrate_memory=args.calibrate_memory,
        output=args.output,
    )
    proof = await RemoteXEliteValidator(config).run()
    print(f"PASS: wrote secret-free proof to {config.output.resolve()}")
    print(json.dumps(proof["summary"], indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return asyncio.run(_async_main(build_parser().parse_args()))
    except (ValidationFailure, ValueError, httpx.HTTPError) as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

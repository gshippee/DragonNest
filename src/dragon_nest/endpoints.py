from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import (
    Device,
    HardwareInventory,
    HealthState,
    HealthStatus,
    ModelCapability,
    ModelSegment,
)


class EndpointError(ValueError):
    pass


@dataclass(frozen=True)
class HttpEndpoint:
    device: Device
    base_url: str
    credential_env: str = ""
    request_timeout_seconds: float = 30.0
    health_timeout_seconds: float = 5.0
    poll_interval_seconds: float = 5.0
    allow_profile_context: bool = False

    def __post_init__(self) -> None:
        if not self.device.device_id.strip():
            raise EndpointError("device_id is required")
        if not self.device.models:
            raise EndpointError("at least one model capability is required")
        if self.credential_env and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.credential_env
        ):
            raise EndpointError("credential_env must be an environment variable name")
        if self.request_timeout_seconds <= 0:
            raise EndpointError("request_timeout_seconds must be positive")
        if self.health_timeout_seconds <= 0:
            raise EndpointError("health_timeout_seconds must be positive")
        if self.poll_interval_seconds < 1:
            raise EndpointError("poll_interval_seconds must be at least 1")


class HttpEndpointStore:
    """SQLite-backed endpoint configuration without persisted credentials."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS http_endpoints (
                    device_id TEXT PRIMARY KEY,
                    base_url TEXT NOT NULL,
                    credential_env TEXT NOT NULL,
                    request_timeout_seconds REAL NOT NULL,
                    health_timeout_seconds REAL NOT NULL,
                    poll_interval_seconds REAL NOT NULL,
                    allow_profile_context INTEGER NOT NULL DEFAULT 0,
                    device_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def put(self, endpoint: HttpEndpoint) -> HttpEndpoint:
        now = time.time()
        payload = json.dumps(
            asdict(endpoint.device), separators=(",", ":"), sort_keys=True
        )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO http_endpoints (
                    device_id, base_url, credential_env,
                    request_timeout_seconds, health_timeout_seconds,
                    poll_interval_seconds, allow_profile_context, device_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    base_url=excluded.base_url,
                    credential_env=excluded.credential_env,
                    request_timeout_seconds=excluded.request_timeout_seconds,
                    health_timeout_seconds=excluded.health_timeout_seconds,
                    poll_interval_seconds=excluded.poll_interval_seconds,
                    allow_profile_context=excluded.allow_profile_context,
                    device_json=excluded.device_json,
                    updated_at=excluded.updated_at
                """,
                (
                    endpoint.device.device_id,
                    endpoint.base_url,
                    endpoint.credential_env,
                    endpoint.request_timeout_seconds,
                    endpoint.health_timeout_seconds,
                    endpoint.poll_interval_seconds,
                    int(endpoint.allow_profile_context),
                    payload,
                    now,
                    now,
                ),
            )
        return self.get(endpoint.device.device_id)

    def get(self, device_id: str) -> HttpEndpoint:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM http_endpoints WHERE device_id=?", (device_id,)
            ).fetchone()
        if row is None:
            raise EndpointError("HTTP endpoint not found")
        return _endpoint(row)

    def all(self) -> tuple[HttpEndpoint, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM http_endpoints ORDER BY device_id"
            ).fetchall()
        return tuple(_endpoint(row) for row in rows)

    def delete(self, device_id: str) -> bool:
        with self._lock, self._connection:
            deleted = self._connection.execute(
                "DELETE FROM http_endpoints WHERE device_id=?", (device_id,)
            )
        return deleted.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _endpoint(row: sqlite3.Row) -> HttpEndpoint:
    return HttpEndpoint(
        device=_device(json.loads(row["device_json"])),
        base_url=row["base_url"],
        credential_env=row["credential_env"],
        request_timeout_seconds=row["request_timeout_seconds"],
        health_timeout_seconds=row["health_timeout_seconds"],
        poll_interval_seconds=row["poll_interval_seconds"],
        allow_profile_context=bool(row["allow_profile_context"]),
    )


def _device(value: dict[str, Any]) -> Device:
    health_value = value.get("health") or {}
    health = HealthState(
        **{
            **health_value,
            "status": HealthStatus(
                health_value.get("status", HealthStatus.HEALTHY.value)
            ),
        }
    )
    models = []
    for model_value in value.get("models", []):
        segment_value = model_value.get("segment")
        models.append(
            ModelCapability(
                **{
                    **model_value,
                    "task_classes": tuple(model_value.get("task_classes", ())),
                    "steering_vector_ids": tuple(
                        model_value.get("steering_vector_ids", ())
                    ),
                    "supported_steering_layers": tuple(
                        model_value.get("supported_steering_layers", ())
                    ),
                    "supported_accelerators": tuple(
                        model_value.get("supported_accelerators", ("cpu",))
                    ),
                    "segment": (
                        ModelSegment(**segment_value) if segment_value else None
                    ),
                }
            )
        )
    hardware_value = value.get("hardware") or {}
    hardware = HardwareInventory(
        **{
            **hardware_value,
            "cpu_abis": tuple(hardware_value.get("cpu_abis", ())),
        }
    )
    return Device(
        device_id=str(value.get("device_id", "")),
        display_name=str(value.get("display_name", "")),
        device_type=str(value.get("device_type", "endpoint")),
        platform=str(value.get("platform", "")),
        total_memory_mb=int(value.get("total_memory_mb", 0)),
        health=health,
        models=tuple(models),
        hardware=hardware,
    )

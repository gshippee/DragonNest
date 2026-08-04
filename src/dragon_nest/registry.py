from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable

from .models import Device, HealthState, HealthStatus


@dataclass(frozen=True)
class RegistryConfig:
    stale_after_seconds: float = 10.0
    offline_after_seconds: float = 20.0
    unhealthy_thermal_level: float = 0.85
    unhealthy_battery_pct: float = 10.0
    degraded_memory_mb: int = 512
    degraded_accelerator_utilization: float = 0.85

    def __post_init__(self) -> None:
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        if self.offline_after_seconds <= self.stale_after_seconds:
            raise ValueError("offline_after_seconds must exceed stale_after_seconds")


@dataclass(frozen=True)
class RegistryEvent:
    device_id: str
    previous_status: HealthStatus | None
    status: HealthStatus
    reason: str
    timestamp: float


@dataclass(frozen=True)
class DeviceRecord:
    device: Device
    registered_at: float
    last_heartbeat: float
    stream_connected: bool
    departure_since: float | None = None
    active_task_ids: tuple[str, ...] = ()
    warm_model_ids: tuple[str, ...] = ()
    simulated_constraint: bool = False

    @property
    def status(self) -> HealthStatus:
        return self.device.health.status


class DeviceRegistry:
    def __init__(
        self,
        config: RegistryConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config or RegistryConfig()
        self._clock = clock
        self._records: dict[str, DeviceRecord] = {}
        self._events: list[RegistryEvent] = []
        self._lock = threading.RLock()

    def register(self, device: Device, now: float | None = None) -> DeviceRecord:
        timestamp = self._now(now)
        with self._lock:
            previous = self._records.get(device.device_id)
            status = self._status_for_health(device.health)
            if not device.health.reachable and status != HealthStatus.OFFLINE:
                status = HealthStatus.STALE
            normalized = replace(
                device,
                health=replace(device.health, status=status),
            )
            record = DeviceRecord(
                device=normalized,
                registered_at=(previous.registered_at if previous else timestamp),
                last_heartbeat=timestamp,
                stream_connected=True,
                departure_since=(timestamp if status == HealthStatus.STALE else None),
                active_task_ids=previous.active_task_ids if previous else (),
                warm_model_ids=tuple(
                    model.model_id for model in normalized.models if model.warm
                ),
                simulated_constraint=(
                    previous.simulated_constraint if previous else False
                ),
            )
            self._records[device.device_id] = record
            self._append_event(
                device.device_id,
                previous.status if previous else None,
                status,
                "reconnected" if previous else "registered",
                timestamp,
            )
            return record

    def heartbeat(
        self,
        device_id: str,
        health: HealthState,
        now: float | None = None,
        active_task_ids: Iterable[str] | None = None,
        warm_model_ids: Iterable[str] | None = None,
        simulated_constraint: bool = False,
    ) -> DeviceRecord:
        timestamp = self._now(now)
        with self._lock:
            current = self._require(device_id)
            status = self._status_for_health(health)
            departure_since = current.departure_since
            reason = "heartbeat"
            if not health.reachable and status != HealthStatus.OFFLINE:
                status = HealthStatus.STALE
                departure_since = (
                    departure_since if departure_since is not None else timestamp
                )
                reason = "reported_unreachable"
            elif status == HealthStatus.OFFLINE:
                departure_since = (
                    departure_since if departure_since is not None else timestamp
                )
                reason = "reported_offline"
            else:
                departure_since = None
            warm = (
                current.warm_model_ids
                if warm_model_ids is None
                else tuple(dict.fromkeys(warm_model_ids))
            )
            updated_device = replace(
                current.device,
                health=replace(health, status=status),
                models=tuple(
                    replace(model, warm=model.model_id in warm)
                    for model in current.device.models
                ),
            )
            updated = replace(
                current,
                device=updated_device,
                last_heartbeat=timestamp,
                stream_connected=True,
                departure_since=departure_since,
                active_task_ids=(
                    current.active_task_ids
                    if active_task_ids is None
                    else tuple(dict.fromkeys(active_task_ids))
                ),
                warm_model_ids=warm,
                simulated_constraint=simulated_constraint,
            )
            self._records[device_id] = updated
            self._append_transition(current, updated, reason, timestamp)
            return updated

    def stream_closed(
        self,
        device_id: str,
        unexpected: bool = True,
        now: float | None = None,
    ) -> DeviceRecord:
        if not unexpected:
            return self.mark_offline(device_id, reason="graceful_shutdown", now=now)
        timestamp = self._now(now)
        with self._lock:
            current = self._require(device_id)
            updated = self._with_status(
                current,
                HealthStatus.STALE,
                stream_connected=False,
                departure_since=(
                    current.departure_since
                    if current.departure_since is not None
                    else timestamp
                ),
            )
            self._records[device_id] = updated
            self._append_transition(current, updated, "stream_closed", timestamp)
            return updated

    def mark_offline(
        self,
        device_id: str,
        reason: str = "explicit_offline",
        now: float | None = None,
    ) -> DeviceRecord:
        timestamp = self._now(now)
        with self._lock:
            current = self._require(device_id)
            updated = self._with_status(
                current,
                HealthStatus.OFFLINE,
                stream_connected=False,
                departure_since=(
                    current.departure_since
                    if current.departure_since is not None
                    else timestamp
                ),
            )
            self._records[device_id] = updated
            self._append_transition(current, updated, reason, timestamp)
            return updated

    def sweep(self, now: float | None = None) -> tuple[RegistryEvent, ...]:
        timestamp = self._now(now)
        transitions: list[RegistryEvent] = []
        with self._lock:
            for device_id, current in tuple(self._records.items()):
                if current.status == HealthStatus.OFFLINE:
                    continue
                departed_for = (
                    timestamp - current.departure_since
                    if current.departure_since is not None
                    else 0.0
                )
                heartbeat_age = timestamp - current.last_heartbeat
                if (
                    departed_for >= self.config.offline_after_seconds
                    or heartbeat_age >= self.config.offline_after_seconds
                ):
                    status = HealthStatus.OFFLINE
                    reason = "heartbeat_timeout"
                    connected = False
                elif (
                    current.departure_since is not None
                    or heartbeat_age >= self.config.stale_after_seconds
                ):
                    status = HealthStatus.STALE
                    reason = "heartbeat_stale"
                    connected = current.stream_connected
                else:
                    continue
                updated = self._with_status(
                    current,
                    status,
                    stream_connected=connected,
                    departure_since=(
                        current.departure_since
                        if current.departure_since is not None
                        else timestamp
                    ),
                )
                self._records[device_id] = updated
                event = self._append_transition(current, updated, reason, timestamp)
                if event:
                    transitions.append(event)
        return tuple(transitions)

    def get(self, device_id: str) -> DeviceRecord:
        with self._lock:
            return self._require(device_id)

    def records(self) -> tuple[DeviceRecord, ...]:
        with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def events(self) -> tuple[RegistryEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def eligible(
        self,
        candidate_device_ids: Iterable[str] | None = None,
        allow_stale_fallback: bool = True,
    ) -> tuple[Device, ...]:
        candidates = (
            set(candidate_device_ids) if candidate_device_ids is not None else None
        )
        with self._lock:
            records = [
                record
                for device_id, record in self._records.items()
                if candidates is None or device_id in candidates
            ]
            normal = [
                record.device
                for record in records
                if record.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}
                and record.stream_connected
                and record.device.health.reachable
            ]
            if normal or not allow_stale_fallback:
                return tuple(sorted(normal, key=lambda device: device.device_id))
            stale = [
                record.device
                for record in records
                if record.status == HealthStatus.STALE
                and record.stream_connected
                and record.device.health.reachable
            ]
            return tuple(sorted(stale, key=lambda device: device.device_id))

    def _status_for_health(self, health: HealthState) -> HealthStatus:
        if health.status in {HealthStatus.OFFLINE, HealthStatus.STALE}:
            return health.status
        if health.thermal_level >= self.config.unhealthy_thermal_level:
            return HealthStatus.UNHEALTHY
        if (
            0 <= health.battery_pct < self.config.unhealthy_battery_pct
            and not health.charging
        ):
            return HealthStatus.UNHEALTHY
        if 0 < health.available_memory_mb < self.config.degraded_memory_mb:
            return HealthStatus.DEGRADED
        if (
            health.accelerator_utilization
            > self.config.degraded_accelerator_utilization
        ):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def _with_status(
        self,
        record: DeviceRecord,
        status: HealthStatus,
        stream_connected: bool,
        departure_since: float | None,
    ) -> DeviceRecord:
        return replace(
            record,
            device=replace(
                record.device,
                health=replace(record.device.health, status=status),
            ),
            stream_connected=stream_connected,
            departure_since=departure_since,
        )

    def _append_transition(
        self,
        previous: DeviceRecord,
        current: DeviceRecord,
        reason: str,
        timestamp: float,
    ) -> RegistryEvent | None:
        if previous.status == current.status:
            return None
        return self._append_event(
            current.device.device_id,
            previous.status,
            current.status,
            reason,
            timestamp,
        )

    def _append_event(
        self,
        device_id: str,
        previous_status: HealthStatus | None,
        status: HealthStatus,
        reason: str,
        timestamp: float,
    ) -> RegistryEvent:
        event = RegistryEvent(device_id, previous_status, status, reason, timestamp)
        self._events.append(event)
        return event

    def _require(self, device_id: str) -> DeviceRecord:
        try:
            return self._records[device_id]
        except KeyError as exc:
            raise KeyError(f"unknown device {device_id}") from exc

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

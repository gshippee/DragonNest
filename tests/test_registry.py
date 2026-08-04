from __future__ import annotations

from dataclasses import replace

from dragon_nest.models import Device, HealthState, HealthStatus
from dragon_nest.registry import DeviceRegistry, RegistryConfig


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _device(device_id: str, **health_changes) -> Device:
    health = HealthState(
        battery_pct=80,
        available_memory_mb=4096,
        network_rtt_ms=5,
    )
    health = replace(health, **health_changes)
    return Device(device_id, device_id, "pc", "windows", 8192, health, ())


def _registry(clock: Clock) -> DeviceRegistry:
    return DeviceRegistry(
        RegistryConfig(stale_after_seconds=10, offline_after_seconds=20),
        clock=clock,
    )


def test_unexpected_stream_close_transitions_stale_then_offline():
    clock = Clock()
    registry = _registry(clock)
    registry.register(_device("pc-01"))

    record = registry.stream_closed("pc-01")
    assert record.status == HealthStatus.STALE
    assert not record.stream_connected
    assert not registry.eligible(["pc-01"])

    clock.advance(20)
    registry.sweep()
    assert registry.get("pc-01").status == HealthStatus.OFFLINE


def test_missed_heartbeats_expire_and_reconnect_restores_health():
    clock = Clock()
    registry = _registry(clock)
    registry.register(_device("phone-01"))

    clock.advance(10)
    registry.sweep()
    assert registry.get("phone-01").status == HealthStatus.STALE

    clock.advance(10)
    registry.sweep()
    assert registry.get("phone-01").status == HealthStatus.OFFLINE

    registry.register(_device("phone-01"))
    assert registry.get("phone-01").status == HealthStatus.HEALTHY
    assert registry.get("phone-01").stream_connected


def test_reported_unreachable_starts_departure_timeout():
    clock = Clock()
    registry = _registry(clock)
    registry.register(_device("phone-01"))

    registry.heartbeat(
        "phone-01",
        _device("unused", reachable=False).health,
    )
    assert registry.get("phone-01").status == HealthStatus.STALE

    clock.advance(20)
    registry.sweep()
    assert registry.get("phone-01").status == HealthStatus.OFFLINE


def test_graceful_shutdown_is_immediately_offline():
    clock = Clock()
    registry = _registry(clock)
    registry.register(_device("pc-01"))

    registry.stream_closed("pc-01", unexpected=False)

    assert registry.get("pc-01").status == HealthStatus.OFFLINE
    assert not registry.eligible(["pc-01"])


def test_stale_device_is_only_returned_when_no_healthy_candidate_exists():
    clock = Clock()
    registry = _registry(clock)
    registry.register(_device("phone-01"))
    registry.register(_device("pc-01"))

    clock.advance(9)
    registry.heartbeat("pc-01", _device("unused").health)
    clock.advance(1)
    registry.sweep()

    assert registry.get("phone-01").status == HealthStatus.STALE
    assert [device.device_id for device in registry.eligible()] == ["pc-01"]

    registry.mark_offline("pc-01")
    assert [device.device_id for device in registry.eligible()] == ["phone-01"]


def test_real_telemetry_drives_unhealthy_and_degraded_states():
    clock = Clock()
    registry = _registry(clock)

    hot = registry.register(_device("hot", thermal_level=0.9))
    constrained = registry.register(_device("low-memory", available_memory_mb=256))

    assert hot.status == HealthStatus.UNHEALTHY
    assert constrained.status == HealthStatus.DEGRADED
    assert [device.device_id for device in registry.eligible()] == ["low-memory"]

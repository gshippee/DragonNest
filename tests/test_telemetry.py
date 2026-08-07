from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from dragon_nest.config import load_devices
from dragon_nest.models import HealthState, HealthStatus
from dragon_nest.telemetry import SimulatedTelemetry, SystemTelemetry, TelemetrySnapshot
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server


ROOT = Path(__file__).resolve().parents[1]


class MutableTelemetry:
    def __init__(self, snapshot: TelemetrySnapshot):
        self.snapshot = snapshot

    def sample(self) -> TelemetrySnapshot:
        return self.snapshot


def test_system_telemetry_reports_accelerator_utilization():
    device = load_devices(ROOT / "configs/dev-fabric.yaml")[0]
    snapshot = SystemTelemetry(device).sample()

    if sys.platform == "win32":
        # Real GPU/NPU utilization via the `\GPU Engine` performance counter
        # (the NPU enumerates as its own compute-only adapter on Copilot+
        # hardware) — either a real fraction, or -1 if this machine's driver
        # doesn't expose the counter category at all.
        assert snapshot.health.gpu_utilization == -1 or 0 <= snapshot.health.gpu_utilization <= 1
        assert snapshot.health.npu_utilization == -1 or 0 <= snapshot.health.npu_utilization <= 1
        assert snapshot.health.accelerator_utilization == max(
            snapshot.health.gpu_utilization, snapshot.health.npu_utilization
        )
    else:
        # Not implemented on this platform yet — must stay honestly unknown,
        # never a fabricated number.
        assert snapshot.health.accelerator_utilization == -1
        assert snapshot.health.gpu_utilization == -1
        assert snapshot.health.npu_utilization == -1
    assert snapshot.health.available_memory_mb >= 0
    assert snapshot.health.battery_pct >= -1
    assert snapshot.warm_model_ids


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only telemetry probe")
def test_system_telemetry_reports_real_memory_on_windows():
    device = load_devices(ROOT / "configs/dev-fabric.yaml")[0]
    snapshot = SystemTelemetry(device).sample()

    # Without a native probe the sampler reports 0 ("unknown") and the router
    # would exclude every model on the Snapdragon X Elite laptop.
    assert snapshot.health.available_memory_mb > 0


def test_simulated_telemetry_overrides_platform_sample():
    device = load_devices(ROOT / "configs/dev-fabric.yaml")[0]
    telemetry = SimulatedTelemetry(
        SystemTelemetry(device),
        thermal_level=0.95,
        accelerator_utilization=0.9,
    )

    snapshot = telemetry.sample()

    assert snapshot.health.thermal_level == 0.95
    assert snapshot.health.accelerator_utilization == 0.9
    assert snapshot.simulated_constraint


def test_network_change_triggers_immediate_telemetry_heartbeat():
    async def scenario() -> None:
        service = BrainService()
        server, port = await create_server(service, "127.0.0.1:0")
        device = load_devices(ROOT / "configs/dev-fabric.yaml")[0]
        telemetry = MutableTelemetry(
            TelemetrySnapshot(
                health=HealthState(
                    battery_pct=80,
                    thermal_level=0.2,
                    cpu_utilization=0.1,
                    accelerator_utilization=0.1,
                    available_memory_mb=4096,
                    network_rtt_ms=-1,
                ),
                warm_model_ids=(device.models[0].model_id,),
            )
        )
        agent = DeviceAgent(
            device,
            AgentClientConfig(
                brain_target=f"127.0.0.1:{port}",
                heartbeat_interval_seconds=60,
                reconnect_initial_seconds=0.01,
            ),
            telemetry=telemetry,
        )
        agent_task = asyncio.create_task(agent.run_forever())
        try:
            await asyncio.wait_for(agent.registered.wait(), timeout=3)
            for _ in range(50):
                initial = service.registry.get(device.device_id)
                if initial.device.health.thermal_level == pytest.approx(0.2):
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.02)

            telemetry.snapshot = TelemetrySnapshot(
                health=HealthState(
                    battery_pct=75,
                    thermal_level=0.95,
                    cpu_utilization=0.8,
                    accelerator_utilization=0.9,
                    available_memory_mb=2048,
                    network_rtt_ms=-1,
                ),
                active_task_ids=("external-task",),
                warm_model_ids=(device.models[-1].model_id,),
                simulated_constraint=True,
            )
            agent.notify_network_changed()

            for _ in range(100):
                record = service.registry.get(device.device_id)
                if record.status == HealthStatus.UNHEALTHY:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("network change did not trigger a heartbeat")

            assert record.device.health.thermal_level == pytest.approx(0.95)
            assert record.device.health.network_rtt_ms >= 0
            assert record.active_task_ids == ("external-task",)
            assert record.warm_model_ids == (device.models[-1].model_id,)
            assert record.simulated_constraint
        finally:
            await agent.stop()
            await asyncio.gather(agent_task, return_exceptions=True)
            await stop_server(server, service)

    asyncio.run(scenario())

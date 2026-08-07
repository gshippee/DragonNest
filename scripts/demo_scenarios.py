"""Deterministic behavior-routing demo: scenarios A-G on a simulated fleet.

Runs a Snapdragon X Elite laptop and a Galaxy S25 Ultra as in-process mock
agents against a real Brain (gRPC + scheduler + dispatch), then walks the
seven demo scenarios:

  A. Warm-device preference
  B. Behavior locality (runtime vector vs baked artifact)
  C. Thermal reroute
  D. Long-context memory rejection before dispatch
  E. Runtime steering unavailable -> baked fallback / policy rejection
  F. Mid-task disconnect -> fenced retry on the other device
  G. Missing profile -> provisioning -> routable

Every routing decision is printed with the scheduler's own explanation.
Exits non-zero if any scenario assertion fails.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dragon_nest.behavior import BehaviorProfileRegistry
from dragon_nest.config import load_devices
from dragon_nest.deployments import ArtifactCatalog, ArtifactState
from dragon_nest.scheduler import RequestSpec
from dragon_nest.steering import SteeringRegistry
from dragon_nest.telemetry import SimulatedTelemetry, SystemTelemetry
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server

ROOT = Path(__file__).resolve().parents[1]
LAPTOP = "x-elite-01"
PHONE = "s25-ultra-01"

# Deterministic Brain-side RTT overlays: agents report their real measured
# loopback RTT, which would make the network-cost comparison nondeterministic.
BASE_RTT = {LAPTOP: 6.0, PHONE: 18.0}


def _approx(actual: float, expected: float, tolerance: float = 1e-3) -> bool:
    return abs(actual - expected) < tolerance


def _print_plan(title: str, plan) -> None:
    print(f"\n=== {title} ===")
    for line in plan.explanation:
        print(f"  {line}")


def _check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise AssertionError(message)


async def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError("condition not reached")


async def main() -> None:
    service = BrainService(
        steering_registry=SteeringRegistry.from_yaml(
            ROOT / "configs/steering-vectors.yaml"
        ),
        artifact_catalog=ArtifactCatalog.from_yaml(
            ROOT / "configs/artifact-catalog.yaml"
        ),
        behavior_registry=BehaviorProfileRegistry.from_yaml(
            ROOT / "configs/behavior-profiles.yaml"
        ),
    )
    server, port = await create_server(service, "127.0.0.1:0")
    target = f"127.0.0.1:{port}"
    devices = load_devices(ROOT / "configs/demo-fleet.yaml")
    agents = []
    for device in devices:
        health = device.health
        telemetry = SimulatedTelemetry(
            SystemTelemetry(device),
            battery_pct=health.battery_pct,
            charging=health.charging,
            thermal_level=health.thermal_level,
            cpu_utilization=health.cpu_utilization,
            accelerator_utilization=health.accelerator_utilization,
            available_memory_mb=health.available_memory_mb,
            network_rtt_ms=health.network_rtt_ms,
        )
        agents.append(
            DeviceAgent(
                device,
                AgentClientConfig(
                    brain_target=target, heartbeat_interval_seconds=0.1
                ),
                telemetry=telemetry,
            )
        )
    agent_tasks = [asyncio.create_task(agent.run_forever()) for agent in agents]

    try:
        await asyncio.wait_for(
            asyncio.gather(*(agent.registered.wait() for agent in agents)),
            timeout=5,
        )
        def simulate(device_id: str, **extra) -> None:
            service.set_device_simulation(
                device_id, {"network_rtt_ms": BASE_RTT[device_id], **extra}
            )

        simulate(LAPTOP)
        simulate(PHONE)
        # wait until the deterministic simulated heartbeats have landed
        await _wait_for(
            lambda: (
                _approx(
                    service.registry.get(LAPTOP).device.health.thermal_level, 0.2
                )
                and _approx(
                    service.registry.get(PHONE).device.health.thermal_level, 0.3
                )
                and _approx(
                    service.registry.get(PHONE).device.health.network_rtt_ms, 18.0
                )
            )
        )

        # --- Scenario A: warm-device preference ---------------------------
        plan = service.build_route_plan(RequestSpec(base_model_family="mock"))
        _print_plan("Scenario A: warm-device preference", plan)
        _check(plan.chosen is not None, "a route was chosen")
        _check(
            plan.chosen.device_id == LAPTOP
            and plan.chosen.deployment.state == ArtifactState.WARM,
            "warm laptop deployment wins",
        )
        _check(
            any(c.device_id == PHONE and c.feasible for c in plan.candidates),
            "phone stays feasible but loses on cold-load cost",
        )

        # --- Scenario B: exact baked profile -------------------------------
        plan = service.build_route_plan(
            RequestSpec(base_model_family="mock", behavior_profile_id="concise")
        )
        _print_plan("Scenario B: exact baked profile (concise)", plan)
        modes = {c.realization_mode for c in plan.candidates if c.feasible}
        _check(
            modes == {"baked_profile"},
            "Concise admits only the exact baked realization",
        )
        _check(
            plan.chosen.realization_mode == "baked_profile"
            and plan.chosen.device_id == LAPTOP,
            "the installed baked deployment is selected",
        )
        _check(
            not plan.steering.enabled and not plan.steering.vector_id,
            "no runtime vector is sent for a baked profile",
        )

        # --- Scenario C: thermal reroute ----------------------------------
        service.set_deployment_simulation(
            LAPTOP, {"small-chat-v1": ArtifactState.ABSENT}
        )
        before = service.build_route_plan(RequestSpec(base_model_family="mock"))
        _print_plan("Scenario C1: phone preferred before thermal pressure", before)
        _check(before.chosen.device_id == PHONE, "phone wins before pressure")

        simulate(PHONE, thermal_level=0.92)
        await _wait_for(
            lambda: _approx(
                service.registry.get(PHONE).device.health.thermal_level, 0.92
            )
        )
        after = service.build_route_plan(RequestSpec(base_model_family="mock"))
        _print_plan("Scenario C2: thermal pressure reroutes to laptop", after)
        _check(after.chosen.device_id == LAPTOP, "laptop wins under phone thermal pressure")
        _check(
            any(
                "UNHEALTHY" in reason
                for c in after.candidates
                if c.device_id == PHONE
                for reason in c.rejection_reasons
            ),
            "phone rejection cites its thermal health state",
        )
        simulate(PHONE)
        service.deployment_overrides.clear()
        await _wait_for(
            lambda: _approx(
                service.registry.get(PHONE).device.health.thermal_level, 0.3
            )
        )

        # --- Scenario D: memory rejection ---------------------------------
        simulate(PHONE, available_memory_mb=1000)
        await _wait_for(
            lambda: service.registry.get(PHONE).device.health.available_memory_mb
            == 1000
        )
        plan = service.build_route_plan(
            RequestSpec(
                base_model_family="mock",
                estimated_input_tokens=2600,
                estimated_output_tokens=400,
            )
        )
        _print_plan("Scenario D: long-context memory rejection", plan)
        _check(plan.chosen.device_id == LAPTOP, "laptop absorbs the long context")
        _check(
            any(
                "MB" in reason
                for c in plan.candidates
                if c.device_id == PHONE
                for reason in c.rejection_reasons
            ),
            "phone was rejected before dispatch with concrete memory numbers",
        )
        simulate(PHONE)
        await _wait_for(
            lambda: service.registry.get(PHONE).device.health.available_memory_mb
            == 6144
        )

        # --- Scenario E: exact profile availability ------------------------
        service.set_runtime_steering_enabled(PHONE, False)
        plan = service.build_route_plan(
            RequestSpec(base_model_family="mock", behavior_profile_id="concise")
        )
        _print_plan("Scenario E1: Concise remains baked when runtime steering is disabled", plan)
        _check(
            plan.chosen.realization_mode == "baked_profile",
            "Concise uses the baked artifact and does not depend on runtime steering",
        )
        service.set_deployment_simulation(
            LAPTOP, {"small-chat-v1-concise-baked": ArtifactState.ABSENT}
        )
        strict = service.build_route_plan(
            RequestSpec(
                base_model_family="mock",
                behavior_profile_id="concise",
            )
        )
        _print_plan("Scenario E2: missing exact bake rejects", strict)
        _check(
            strict.chosen is None
            and strict.error_code == "BEHAVIOR_UNAVAILABLE",
            "missing baked profile rejects rather than substituting a prompt or vector",
        )
        service.deployment_overrides.clear()
        service.set_runtime_steering_enabled(PHONE, True)

        # --- Scenario F: mid-task disconnect and fenced retry --------------
        laptop_agent = next(a for a in agents if a.device.device_id == LAPTOP)
        laptop_agent.simulate_disconnect_on_next_task()
        plan, response = await service.submit_behavior_task(
            RequestSpec(
                request_text="Summarize today's routing decisions.",
                base_model_family="mock",
            ),
            timeout_ms=5000,
        )
        _print_plan("Scenario F: disconnect during dispatch", plan)
        _check(plan.chosen.device_id == LAPTOP, "laptop was the first choice")
        _check(response.success, "task still succeeded after the disconnect")
        _check(
            response.device_id == PHONE,
            "fenced retry completed on the phone",
        )
        task = service.tasks.get(response.task_id)
        _check(
            any(a.state.value == "DEVICE_OFFLINE" for a in task.attempts),
            "the laptop attempt was fenced as DEVICE_OFFLINE",
        )
        # let the laptop agent reconnect before the next scenario
        await _wait_for(
            lambda: service.registry.get(LAPTOP).status.value in {"HEALTHY", "DEGRADED"},
            timeout=10,
        )

        # --- Scenario G: missing profile -> provisioning -------------------
        plan = service.build_route_plan(
            RequestSpec(
                base_model_family="mock", behavior_profile_id="family-assistant"
            )
        )
        _print_plan("Scenario G1: family-assistant is not deployable", plan)
        _check(
            plan.chosen is None and plan.error_code == "BEHAVIOR_UNAVAILABLE",
            "request rejected instead of silently degrading",
        )
        _check(
            plan.provisioning_hint == "family-assistant",
            "scheduler suggests provisioning the profile",
        )
        job = service.provisioning.start(
            "family-assistant", LAPTOP, "family-assistant-v0-baked"
        )
        print("  provisioning:", job.state.value, end="")
        while job.state.value not in {"warm", "failed"}:
            job = service.provisioning.advance(job.job_id)
            print(" ->", job.state.value, end="")
        print()
        _check(job.state.value == "warm", "mock provisioning reached warm")
        _check(
            all(detail.startswith("[mock]") for _, detail in job.history[1:]),
            "every provisioning step is labeled [mock]; no false compile claim",
        )
        provisioned = service.build_route_plan(
            RequestSpec(
                base_model_family="mock", behavior_profile_id="family-assistant"
            )
        )
        _print_plan("Scenario G2: profile routable after provisioning", provisioned)
        _check(
            provisioned.chosen is not None
            and provisioned.chosen.artifact.artifact_id
            == "family-assistant-v0-baked",
            "provisioned baked artifact now serves the profile",
        )

        print("\nAll scenarios passed.")
    finally:
        await asyncio.gather(*(agent.stop() for agent in agents))
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        await stop_server(server, service)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\nDEMO SCENARIO FAILED: {exc}")
        sys.exit(1)

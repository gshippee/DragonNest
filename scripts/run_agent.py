from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dragon_nest.artifacts import ArtifactRegistry
from dragon_nest.config import load_device
from dragon_nest.telemetry import SimulatedTelemetry, SystemTelemetry
from dragon_nest.runtime.hardware_adapter import AdapterTelemetry, HardwareRuntimeAdapter
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent


async def run(args) -> None:
    artifacts = (
        ArtifactRegistry.from_yaml(args.artifact_manifest)
        if args.artifact_manifest
        else None
    )
    device = load_device(args.fabric, args.device_id)
    telemetry = SystemTelemetry(device)
    simulation = {
        "thermal_level": args.simulate_thermal,
        "battery_pct": args.simulate_battery,
        "accelerator_utilization": args.simulate_load,
        "network_rtt_ms": args.simulate_rtt,
    }
    if any(value is not None for value in simulation.values()):
        telemetry = SimulatedTelemetry(telemetry, **simulation)
    adapter = None
    if artifacts is not None and args.compatibility_key:
        adapter = HardwareRuntimeAdapter(
            artifacts,
            compatibility_key=args.compatibility_key,
            compatible_target_classes=tuple(args.compatible_target_class),
            runtime_name=args.runtime_name,
            runtime_version=args.runtime_version,
            accelerator_available=args.accelerator_available,
            telemetry=telemetry,
            artifact_store=args.artifact_store,
        )
        telemetry = AdapterTelemetry(adapter)
    pipeline_provider = None
    if args.enable_qwen17_pipeline:
        if artifacts is None:
            raise RuntimeError("--enable-qwen17-pipeline requires --artifact-manifest")
        from dragon_nest.runtime.qwen17_provider import Qwen17PipelineProvider

        pipeline_provider = Qwen17PipelineProvider(artifacts)
    agent = DeviceAgent(
        device,
        AgentClientConfig(
            brain_target=args.brain,
            enrollment_token=args.enrollment_token,
            tls_ca_path=str(args.tls_ca or ""),
            tls_client_certificate_path=str(args.tls_certificate or ""),
            tls_client_key_path=str(args.tls_key or ""),
        ),
        artifacts=artifacts,
        executor=adapter,
        telemetry=telemetry,
        pipeline_provider=pipeline_provider,
    )
    print(f"Starting {args.device_id}; connecting to {args.brain}")
    try:
        await agent.run_forever()
    finally:
        await agent.stop(graceful=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DragonNest Device Agent")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--brain", default="127.0.0.1:50051")
    parser.add_argument("--enrollment-token", default="dev-token")
    parser.add_argument("--tls-ca", type=Path)
    parser.add_argument("--tls-certificate", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument(
        "--fabric",
        type=Path,
        default=Path("configs/dev-fabric.yaml"),
    )
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--simulate-thermal", type=float)
    parser.add_argument("--simulate-battery", type=float)
    parser.add_argument("--simulate-load", type=float)
    parser.add_argument("--simulate-rtt", type=float)
    parser.add_argument(
        "--compatibility-key",
        help="exact target class, for example windows-arm64-x1e-v73-qairt-2.48",
    )
    parser.add_argument(
        "--compatible-target-class",
        action="append",
        default=[],
        help=(
            "Additional artifact target proven compatible with this runtime; "
            "repeatable and never inferred automatically."
        ),
    )
    parser.add_argument("--runtime-name", default="genie")
    parser.add_argument("--runtime-version", default="unknown")
    parser.add_argument(
        "--accelerator-available",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--artifact-store",
        type=Path,
        default=Path.home() / ".dragonnest" / "artifacts",
    )
    parser.add_argument(
        "--enable-qwen17-pipeline",
        action="store_true",
        help=(
            "Enable the fail-closed physical Qwen3-1.7B QNN stage provider; "
            "requires the four checksummed X Elite artifacts and QAIRT 2.45"
        ),
    )
    parser.set_defaults(artifact_manifest=Path("configs/model-artifacts.yaml"))
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

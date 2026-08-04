from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import uvicorn

from dragon_nest.dashboard import create_dashboard_app
from dragon_nest.steering import SteeringRegistry
from dragon_nest.transport.brain import (
    BrainService,
    BrainServiceConfig,
    create_server,
    stop_server,
)


async def run(args) -> None:
    service = BrainService(
        BrainServiceConfig(
            brain_id=args.brain_id,
            enrollment_token=args.enrollment_token,
            dev_mode=not args.production,
            tls_server_certificate_path=str(args.tls_certificate or ""),
            tls_server_key_path=str(args.tls_key or ""),
            tls_client_ca_path=str(args.tls_client_ca or ""),
            state_db_path=str(args.state_db),
        ),
        steering_registry=SteeringRegistry.from_yaml(args.steering_config),
    )
    server, port = await create_server(service, args.address)
    dashboard = uvicorn.Server(
        uvicorn.Config(
            create_dashboard_app(service),
            host=args.http_host,
            port=args.http_port,
            log_level="warning",
        )
    )
    dashboard_task = asyncio.create_task(dashboard.serve())
    print(f"DragonNest Brain listening on {args.address} (port {port})")
    print(f"DragonNest dashboard: http://{args.http_host}:{args.http_port}")
    try:
        await server.wait_for_termination()
    finally:
        dashboard.should_exit = True
        await dashboard_task
        await stop_server(server, service)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DragonNest gRPC Brain")
    parser.add_argument("--address", default="0.0.0.0:50051")
    parser.add_argument("--brain-id", default="dragon-nest-brain")
    parser.add_argument("--enrollment-token", default="dev-token")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--tls-certificate", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--tls-client-ca", type=Path)
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument(
        "--state-db",
        type=Path,
        default=Path("local/dragonnest-state.sqlite3"),
    )
    parser.add_argument(
        "--steering-config",
        type=Path,
        default=Path("configs/steering-vectors.yaml"),
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

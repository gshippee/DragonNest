from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import uvicorn

from dragon_nest.behavior import BehaviorProfileRegistry
from dragon_nest.dashboard import create_dashboard_app
from dragon_nest.deployments import ArtifactCatalog
from dragon_nest.steering import SteeringRegistry
from dragon_nest.transport.brain import (
    BrainService,
    BrainServiceConfig,
    create_server,
    stop_server,
)


async def run(args) -> None:
    http_endpoint_admin_token = os.environ.get(args.http_endpoint_admin_token_env, "")
    if (
        args.enable_http_endpoints
        and not http_endpoint_admin_token
        and not args.production
    ):
        http_endpoint_admin_token = args.enrollment_token
        print(
            f"{args.http_endpoint_admin_token_env} is not set; using --enrollment-token "
            f"({http_endpoint_admin_token!r}) as the HTTP endpoint admin token for local "
            f"dev. Set {args.http_endpoint_admin_token_env} to override, or pass "
            "--production to require it explicitly."
        )
    service = BrainService(
        BrainServiceConfig(
            brain_id=args.brain_id,
            enrollment_token=args.enrollment_token,
            dev_mode=not args.production,
            tls_server_certificate_path=str(args.tls_certificate or ""),
            tls_server_key_path=str(args.tls_key or ""),
            tls_client_ca_path=str(args.tls_client_ca or ""),
            state_db_path=str(args.state_db),
            http_endpoint_registration_enabled=args.enable_http_endpoints,
            http_endpoint_admin_token=http_endpoint_admin_token,
            http_endpoint_allowed_cidrs=tuple(
                args.http_endpoint_allow_cidr or ("127.0.0.0/8", "::1/128")
            ),
            http_endpoint_allowed_hosts=tuple(
                args.http_endpoint_allow_host or ("localhost",)
            ),
        ),
        steering_registry=SteeringRegistry.from_yaml(args.steering_config),
        artifact_catalog=ArtifactCatalog.from_yaml(args.artifact_catalog),
        behavior_registry=BehaviorProfileRegistry.from_yaml(args.behavior_profiles),
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
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument(
        "--enable-http-endpoints",
        dest="enable_http_endpoints",
        action="store_true",
        default=True,
        help=(
            "Enable the HTTP endpoint API (manual entry + auto-discovery in the "
            "admin dashboard, and re-registration of persisted endpoints on "
            "startup). Enabled by default."
        ),
    )
    parser.add_argument(
        "--disable-http-endpoints",
        dest="enable_http_endpoints",
        action="store_false",
        help="Disable the HTTP endpoint API.",
    )
    parser.add_argument(
        "--http-endpoint-admin-token-env",
        default="DRAGONNEST_HTTP_ENDPOINT_ADMIN_TOKEN",
    )
    parser.add_argument(
        "--http-endpoint-allow-cidr",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--http-endpoint-allow-host",
        action="append",
        default=[],
    )
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
    parser.add_argument(
        "--artifact-catalog",
        type=Path,
        default=Path("configs/artifact-catalog.yaml"),
    )
    parser.add_argument(
        "--behavior-profiles",
        type=Path,
        default=Path("configs/behavior-profiles.yaml"),
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

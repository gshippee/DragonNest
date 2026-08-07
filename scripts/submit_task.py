from __future__ import annotations

import argparse
import asyncio

import grpc

from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.proto import dragonnest_pb2_grpc as pb_grpc


async def run(args) -> int:
    request = pb.SubmitTaskRequest(
        request_text=args.request,
        preferred_mode=args.preferred_mode,
        execution_mode=args.execution_mode,
        origin_device_id=args.origin_device_id,
        reducer=args.reducer,
        timeout_ms=args.timeout_ms,
        persona_id=args.persona_id,
    )
    if args.steering_alpha is not None:
        # Calibration path: pin an explicit alpha instead of taking the one the
        # behavior profile carries. Used to sweep the vector on hardware before
        # committing a production value to configs/behavior-profiles.yaml.
        request.steering.CopyFrom(
            pb.SteeringSpec(
                enabled=True,
                mode="runtime_vector",
                vector_id=args.steering_vector_id,
                target_layer=args.steering_layer,
                alpha=args.steering_alpha,
                positions=args.steering_positions,
                model_family=args.steering_model_family,
                behavior_profile_id=args.persona_id,
            )
        )
    async with grpc.aio.insecure_channel(args.brain) as channel:
        stub = pb_grpc.BrainControlStub(channel)
        response = await stub.SubmitTask(request)
    print(f"task_id: {response.task_id}")
    print(f"state: {response.state}")
    print(f"device/model: {response.device_id}/{response.model_id}")
    steering = response.steering
    if steering.enabled:
        print(
            f"steering: mode={steering.mode} vector={steering.vector_id} "
            f"layer={steering.target_layer} alpha={steering.alpha:g} "
            f"positions={steering.positions or 'last'}"
        )
    if args.persona_id:
        print(f"persona requested: {args.persona_id}")
        print(
            "persona realized: "
            f"{response.steering.behavior_profile_id or 'none'} "
            f"(mode={response.steering.mode or 'none'})"
        )
    if response.success:
        print(response.output_text)
        return 0
    print(f"{response.error_code}: {response.error_message}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a task to DragonNest")
    parser.add_argument("request")
    parser.add_argument("--brain", default="127.0.0.1:50051")
    parser.add_argument("--preferred-mode", default="auto")
    parser.add_argument("--execution-mode", default="single")
    parser.add_argument("--origin-device-id", default="")
    parser.add_argument(
        "--persona-id",
        choices=("", "balanced", "concise", "detailed"),
        default="",
        help=(
            "PersonaCare profile to request (balanced/concise/detailed). "
            "Required to faithfully exercise Concise/Detailed Local "
            "acceptance from the CLI; omit to use the device's default "
            "persona (balanced)."
        ),
    )
    parser.add_argument(
        "--reducer",
        choices=("concat", "first_success", "mock_synthesis"),
        default="mock_synthesis",
    )
    parser.add_argument(
        "--steering-alpha",
        type=float,
        default=None,
        help=(
            "pin an explicit runtime-vector alpha instead of the profile's "
            "calibrated value. Alpha-sweep tooling for hardware calibration; "
            "PersonaCare never sends this."
        ),
    )
    parser.add_argument(
        "--steering-vector-id", default="concise-vs-verbose-layer-7"
    )
    parser.add_argument("--steering-layer", type=int, default=7)
    parser.add_argument("--steering-positions", default="all")
    parser.add_argument("--steering-model-family", default="qwen3")
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()

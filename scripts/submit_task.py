from __future__ import annotations

import argparse
import asyncio

import grpc

from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.proto import dragonnest_pb2_grpc as pb_grpc


async def run(args) -> int:
    async with grpc.aio.insecure_channel(args.brain) as channel:
        stub = pb_grpc.BrainControlStub(channel)
        response = await stub.SubmitTask(
            pb.SubmitTaskRequest(
                request_text=args.request,
                preferred_mode=args.preferred_mode,
                execution_mode=args.execution_mode,
                origin_device_id=args.origin_device_id,
                reducer=args.reducer,
                timeout_ms=args.timeout_ms,
            )
        )
    print(f"task_id: {response.task_id}")
    print(f"state: {response.state}")
    print(f"device/model: {response.device_id}/{response.model_id}")
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
        "--reducer",
        choices=("concat", "first_success", "mock_synthesis"),
        default="mock_synthesis",
    )
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()

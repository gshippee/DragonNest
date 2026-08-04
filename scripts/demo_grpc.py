from __future__ import annotations

import asyncio
from pathlib import Path

import grpc

from dragon_nest.config import load_devices
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.proto import dragonnest_pb2_grpc as pb_grpc
from dragon_nest.transport.agent import AgentClientConfig, DeviceAgent
from dragon_nest.transport.brain import BrainService, create_server, stop_server


async def main() -> None:
    service = BrainService()
    server, port = await create_server(service, "127.0.0.1:0")
    target = f"127.0.0.1:{port}"
    devices = load_devices(Path("configs/dev-fabric.yaml"))
    agents = [
        DeviceAgent(
            device,
            AgentClientConfig(
                brain_target=target,
                heartbeat_interval_seconds=0.1,
            ),
        )
        for device in devices
    ]
    agent_tasks = [asyncio.create_task(agent.run_forever()) for agent in agents]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(agent.registered.wait() for agent in agents)), timeout=5
        )
        agents[1].simulate_disconnect_on_next_task()
        async with grpc.aio.insecure_channel(target) as channel:
            stub = pb_grpc.BrainControlStub(channel)
            response = await stub.SubmitTask(
                pb.SubmitTaskRequest(
                    request_text="Compare both options and recommend one.",
                    execution_mode="single",
                    timeout_ms=5_000,
                )
            )
            for _ in range(50):
                pc_session = await service.sessions.get("pc-01")
                if pc_session is not None and not pc_session.closed:
                    break
                await asyncio.sleep(0.02)
            parallel = await stub.SubmitTask(
                pb.SubmitTaskRequest(
                    request_text="Summarize sections, then give key points.",
                    execution_mode="data_parallel",
                    timeout_ms=5_000,
                )
            )
            pipeline = await stub.SubmitTask(
                pb.SubmitTaskRequest(
                    request_text="Analyze this complex trade-off.",
                    preferred_mode="quality",
                    execution_mode="layer_pipeline",
                    timeout_ms=5_000,
                )
            )
        print(f"task {response.task_id}: {response.state}")
        print(f"accepted device/model: {response.device_id}/{response.model_id}")
        print(response.output_text)
        print(f"parallel task {parallel.task_id}: {parallel.state}")
        print(parallel.output_text)
        print(f"pipeline task {pipeline.task_id}: {pipeline.state}")
        print(pipeline.output_text)
    finally:
        await asyncio.gather(*(agent.stop() for agent in agents))
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        await stop_server(server, service)


if __name__ == "__main__":
    asyncio.run(main())

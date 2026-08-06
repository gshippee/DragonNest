from __future__ import annotations

import asyncio
from typing import Protocol

from ..proto import dragonnest_pb2 as pb


class SessionConflictError(ValueError):
    pass


class DeviceSession(Protocol):
    device_id: str
    transport: str
    allow_profile_context: bool
    closed: bool

    async def execute(
        self, command: pb.ExecuteTask, timeout_seconds: float
    ) -> pb.TaskResult: ...

    async def execute_shard(
        self, command: pb.ExecuteShard, timeout_seconds: float
    ) -> pb.PartialTaskResult: ...

    async def execute_pipeline_stage(
        self, command: pb.ExecutePipelineStage, timeout_seconds: float
    ) -> pb.PipelineStageResult: ...

    async def cancel(self, task_id: str, attempt_id: str, reason: str) -> None: ...

    async def close(self, graceful: bool = False) -> None: ...


class SessionRegistry:
    def __init__(self):
        self._sessions: dict[str, DeviceSession] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, session: DeviceSession, *, replace_same_transport: bool = False
    ) -> DeviceSession | None:
        async with self._lock:
            previous = self._sessions.get(session.device_id)
            if previous is not None and (
                not replace_same_transport or previous.transport != session.transport
            ):
                raise SessionConflictError(
                    f"device {session.device_id!r} is already connected via "
                    f"{previous.transport}"
                )
            self._sessions[session.device_id] = session
        if previous is not None:
            await previous.close()
        return previous

    async def get(self, device_id: str) -> DeviceSession | None:
        async with self._lock:
            return self._sessions.get(device_id)

    async def remove(self, session: DeviceSession) -> bool:
        async with self._lock:
            if self._sessions.get(session.device_id) is not session:
                return False
            del self._sessions[session.device_id]
            return True

    async def close_device(self, device_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(device_id, None)
        if session is not None:
            await session.close()

    async def close_all(self) -> None:
        async with self._lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            await session.close(graceful=True)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class PipelineSessionKey:
    task_id: str
    pipeline_id: str
    stage_index: int


@dataclass
class PipelineStageSession:
    key: PipelineSessionKey
    prefill_complete: bool = False
    decode_steps: int = 0
    kv_inputs: dict[str, object] = field(default_factory=dict)


class PipelineSessionStore:
    """Agent-local lifecycle for stage KV state; never serialized over gRPC."""

    def __init__(self) -> None:
        self._sessions: dict[PipelineSessionKey, PipelineStageSession] = {}

    def begin_prefill(self, key: PipelineSessionKey) -> PipelineStageSession:
        session = PipelineStageSession(key=key)
        self._sessions[key] = session
        return session

    def require_decode(self, key: PipelineSessionKey) -> PipelineStageSession:
        session = self._sessions.get(key)
        if session is None or not session.prefill_complete:
            raise RuntimeError(
                f"decode requested before prefill for {key.pipeline_id}/S{key.stage_index}"
            )
        session.decode_steps += 1
        return session

    def complete_prefill(self, key: PipelineSessionKey) -> None:
        self._sessions[key].prefill_complete = True

    def retain_outputs(
        self, key: PipelineSessionKey, outputs: Mapping[str, object]
    ) -> None:
        session = self._sessions[key]
        for name, value in outputs.items():
            if name.startswith("past_key_") and name.endswith("_out"):
                session.kv_inputs[name.removesuffix("_out") + "_in"] = value
            elif name.startswith("past_value_") and name.endswith("_out"):
                session.kv_inputs[name.removesuffix("_out") + "_in"] = value

    def get(self, key: PipelineSessionKey) -> PipelineStageSession | None:
        return self._sessions.get(key)

    def release(self, key: PipelineSessionKey) -> bool:
        return self._sessions.pop(key, None) is not None

    def cleanup_task(self, task_id: str) -> int:
        keys = [key for key in self._sessions if key.task_id == task_id]
        for key in keys:
            del self._sessions[key]
        return len(keys)

    def clear(self) -> int:
        count = len(self._sessions)
        self._sessions.clear()
        return count

    def __len__(self) -> int:
        return len(self._sessions)

"""Production Agent provider for the physically proven X Elite Qwen3-1.7B stages."""

from __future__ import annotations

import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..artifacts import ArtifactRegistry, ModelArtifact
from ..pipeline_sessions import PipelineSessionKey
from ..proto import dragonnest_pb2 as pb
from .qwen17_kv import StageKVBuffer

PIPELINE_ID = "qwen3-1.7b-w4a16-demo-v1"
EOS_IDS = frozenset((151645, 151643))


@dataclass
class _StageSession:
    key: PipelineSessionKey
    num_prompt_tokens: int
    kv: StageKVBuffer | None
    decode_steps: int = 0


@dataclass(frozen=True)
class Qwen17StageResult:
    boundary: np.ndarray | None = None
    next_token_id: int | None = None
    token_text: str = ""
    eos: bool = False
    latency_ms: int = 0


class Qwen17PipelineProvider:
    """Own stage-local Qwen KV and execute one PREFILL/DECODE command.

    For the first production acceptance all four stages live in this provider,
    so S0 is the single tokenizer owner and records the prompt-token count for
    S1-S3. A future cross-host S25 stage provider will need an explicit prompt
    metadata transport before real 3+1/2+2 is enabled.
    """

    def __init__(
        self,
        artifacts: ArtifactRegistry,
        *,
        engine=None,
        tokenizer_factory: Callable[[], object] | None = None,
        metadata_cache: str | Path | None = None,
    ):
        self.artifacts = artifacts
        if engine is None:
            from . import qwen17_stage_engine as engine_module

            engine = engine_module
        self.engine = engine
        self.tokenizer_factory = tokenizer_factory or self._default_tokenizer
        self.metadata_cache = Path(
            metadata_cache
            or os.environ.get(
                "DRAGONNEST_QWEN17_METADATA_CACHE",
                str(Path(tempfile.gettempdir()) / "dragon_nest" / "qwen17-metadata"),
            )
        )
        self._tokenizer = None
        self._metadata: dict[str, dict] = {}
        self._sessions: dict[PipelineSessionKey, _StageSession] = {}
        self._task_prompt_tokens: dict[str, int] = {}
        self._cancelled_tasks: OrderedDict[str, None] = OrderedDict()
        self._active_tasks: dict[str, int] = {}
        self._cleanup_pending: set[str] = set()
        self._lock = threading.RLock()

    def supports(self, artifact: ModelArtifact) -> bool:
        split = artifact.split_boundary
        return bool(split and split.pipeline_id == PIPELINE_ID)

    def execute(
        self,
        command: pb.ExecutePipelineStage,
        artifact: ModelArtifact,
        boundary: np.ndarray | None,
    ) -> Qwen17StageResult:
        if not self.supports(artifact):
            raise RuntimeError(f"unsupported physical pipeline: {artifact.model_id}")
        split = artifact.split_boundary
        assert split is not None
        if command.pipeline_id and command.pipeline_id != PIPELINE_ID:
            raise RuntimeError(f"pipeline id mismatch: {command.pipeline_id}")
        if command.stage_index != split.stage_index:
            raise RuntimeError(
                f"stage index mismatch for {artifact.model_id}: "
                f"command S{command.stage_index}, artifact S{split.stage_index}"
            )
        self._begin_execution(command.task_id)
        try:
            key = PipelineSessionKey(command.task_id, PIPELINE_ID, command.stage_index)
            if command.operation == pb.PIPELINE_PREFILL:
                return self._prefill(key, command, artifact, boundary)
            if command.operation == pb.PIPELINE_DECODE:
                return self._decode(key, command, artifact, boundary)
            raise RuntimeError(
                "unsupported Qwen3 physical operation "
                f"{pb.PipelineOperation.Name(command.operation)}"
            )
        finally:
            self._end_execution(command.task_id)

    def release(
        self,
        task_id: str,
        pipeline_id: str,
        stage_index: int,
        *,
        cancelled: bool = False,
    ) -> bool:
        if pipeline_id and pipeline_id != PIPELINE_ID:
            return False
        with self._lock:
            if cancelled:
                self._mark_cancelled(task_id)
            if self._active_tasks.get(task_id, 0):
                self._cleanup_pending.add(task_id)
                return False
            key = PipelineSessionKey(task_id, PIPELINE_ID, stage_index)
            session = self._sessions.pop(key, None)
            if session and session.kv:
                session.kv.release()
            if not any(candidate.task_id == task_id for candidate in self._sessions):
                self._task_prompt_tokens.pop(task_id, None)
            return session is not None

    def cleanup_task(self, task_id: str, *, cancelled: bool = False) -> int:
        with self._lock:
            if cancelled:
                self._mark_cancelled(task_id)
            if self._active_tasks.get(task_id, 0):
                self._cleanup_pending.add(task_id)
                return 0
            return self._cleanup_task_locked(task_id)

    def clear(self) -> int:
        with self._lock:
            task_ids = {
                *(key.task_id for key in self._sessions),
                *self._active_tasks,
            }
            count = 0
            for task_id in task_ids:
                self._mark_cancelled(task_id)
                if self._active_tasks.get(task_id, 0):
                    self._cleanup_pending.add(task_id)
                else:
                    count += self._cleanup_task_locked(task_id)
            if not self._active_tasks:
                self._task_prompt_tokens.clear()
                self._cleanup_pending.clear()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _begin_execution(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._cancelled_tasks:
                raise RuntimeError(f"pipeline task {task_id} was cancelled")
            self._active_tasks[task_id] = self._active_tasks.get(task_id, 0) + 1

    def _end_execution(self, task_id: str) -> None:
        with self._lock:
            remaining = self._active_tasks.get(task_id, 0) - 1
            if remaining > 0:
                self._active_tasks[task_id] = remaining
                return
            self._active_tasks.pop(task_id, None)
            if task_id in self._cleanup_pending or task_id in self._cancelled_tasks:
                self._cleanup_task_locked(task_id)

    def _cleanup_task_locked(self, task_id: str) -> int:
        keys = [key for key in self._sessions if key.task_id == task_id]
        for key in keys:
            session = self._sessions.pop(key)
            if session.kv:
                session.kv.release()
        self._task_prompt_tokens.pop(task_id, None)
        self._cleanup_pending.discard(task_id)
        return len(keys)

    def _mark_cancelled(self, task_id: str) -> None:
        # Cancellation tombstones prevent a late gRPC command from recreating
        # state. Keep them bounded because task IDs are unique and long-lived
        # Agents may process many cancellations.
        self._cancelled_tasks[task_id] = None
        self._cancelled_tasks.move_to_end(task_id)
        while len(self._cancelled_tasks) > 256:
            self._cancelled_tasks.popitem(last=False)

    def _prefill(
        self,
        key: PipelineSessionKey,
        command: pb.ExecutePipelineStage,
        artifact: ModelArtifact,
        boundary: np.ndarray | None,
    ) -> Qwen17StageResult:
        stage = key.stage_index
        path, metadata = self._artifact_context(artifact)
        if stage == 0:
            if not command.request_text:
                raise RuntimeError("Qwen3 S0 PREFILL requires request text")
            input_ids, num_tokens = self._tokenize(command.request_text)
            self._task_prompt_tokens[command.task_id] = num_tokens
            output, elapsed = self.engine.run_s0_prompt(path, metadata, input_ids)
            self._sessions[key] = _StageSession(key, num_tokens, None)
            self._assert_not_cancelled(command.task_id)
            return Qwen17StageResult(output, latency_ms=_milliseconds(elapsed))

        if boundary is None:
            raise RuntimeError(f"Qwen3 S{stage} PREFILL requires its named activation")
        num_tokens = self._task_prompt_tokens.get(command.task_id)
        if num_tokens is None:
            raise RuntimeError(
                "S0 prompt metadata is unavailable on this Agent; real cross-device "
                "Qwen3 execution is not enabled yet"
            )
        layers = _layers(stage)
        output_name = _output_name(stage)
        result = self.engine.run_stage_prompt(
            path,
            metadata,
            _prompt_graph(stage),
            artifact.split_boundary.input_tensor,
            np.asarray(boundary, dtype=np.uint16),
            output_name,
            _output_shape(stage, prompt=True),
            layers,
            num_tokens,
            has_steering=(stage == 1),
        )
        kv = StageKVBuffer(layers)
        for layer in layers:
            kv.seed_from_prompt(
                layer,
                result["past_key_in"][layer],
                result["raw_outputs"][f"past_key_{layer}_out"],
                True,
            )
            kv.seed_from_prompt(
                layer,
                result["past_value_in"][layer],
                result["raw_outputs"][f"past_value_{layer}_out"],
                False,
            )
        self._sessions[key] = _StageSession(key, num_tokens, kv)
        self._assert_not_cancelled(command.task_id)
        return self._result_from_output(stage, result)

    def _decode(
        self,
        key: PipelineSessionKey,
        command: pb.ExecutePipelineStage,
        artifact: ModelArtifact,
        boundary: np.ndarray | None,
    ) -> Qwen17StageResult:
        session = self._sessions.get(key)
        if session is None:
            raise RuntimeError(
                f"decode requested before prefill for {PIPELINE_ID}/S{key.stage_index}"
            )
        stage = key.stage_index
        path, metadata = self._artifact_context(artifact)
        if stage == 0:
            output, elapsed = self.engine.run_s0_decode(
                path, metadata, command.token_id
            )
            self._assert_not_cancelled(command.task_id)
            session.decode_steps += 1
            return Qwen17StageResult(output, latency_ms=_milliseconds(elapsed))
        if boundary is None:
            raise RuntimeError(f"Qwen3 S{stage} DECODE requires its named activation")
        assert session.kv is not None
        result = self.engine.run_stage_decode(
            path,
            metadata,
            _decode_graph(stage),
            artifact.split_boundary.input_tensor,
            np.asarray(boundary, dtype=np.uint16),
            _output_name(stage),
            _output_shape(stage, prompt=False),
            session.kv,
            _layers(stage),
            session.num_prompt_tokens + session.decode_steps,
            session.num_prompt_tokens + session.decode_steps + 1,
            has_steering=(stage == 1),
        )
        self._assert_not_cancelled(command.task_id)
        session.decode_steps += 1
        return self._result_from_output(stage, result)

    def _result_from_output(self, stage: int, result: dict) -> Qwen17StageResult:
        latency = _milliseconds(result["elapsed_sec"])
        if stage != 3:
            return Qwen17StageResult(result["output_u16"], latency_ms=latency)
        final_logits = np.asarray(result["output_u16"])[0, -1, :]
        logits = self.engine.dequantize(
            final_logits,
            result["output_scale"],
            result["output_offset"],
        )
        token_id = int(np.argmax(logits))
        return Qwen17StageResult(
            next_token_id=token_id,
            token_text=str(self._tokenizer_instance().decode([token_id])),
            eos=token_id in EOS_IDS,
            latency_ms=latency,
        )

    def _artifact_context(self, artifact: ModelArtifact) -> tuple[Path, dict]:
        path = self.artifacts.validate(artifact.model_id)
        metadata = self._metadata.get(artifact.model_id)
        if metadata is None:
            cache = self.metadata_cache / f"{artifact.model_id}.json"
            metadata = self.engine.ensure_binary_info(path, cache)
            self._metadata[artifact.model_id] = metadata
        return path, metadata

    def _tokenize(self, request_text: str) -> tuple[np.ndarray, int]:
        tokenizer = self._tokenizer_instance()
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": request_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            max_length=self.engine.CONTEXT_LENGTH,
        )
        num_tokens = int(
            min(encoded["attention_mask"].sum(), self.engine.SEQUENCE_LENGTH)
        )
        return encoded["input_ids"].astype(np.int32)[
            :, -self.engine.SEQUENCE_LENGTH :
        ], num_tokens

    def _tokenizer_instance(self):
        if self._tokenizer is None:
            self._tokenizer = self.tokenizer_factory()
        return self._tokenizer

    @staticmethod
    def _default_tokenizer():
        from transformers import AutoTokenizer

        source = os.environ.get(
            "DRAGONNEST_QWEN17_TOKENIZER", "Qwen/Qwen3-1.7B"
        )
        tokenizer = AutoTokenizer.from_pretrained(source, is_fast=False)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left"
        return tokenizer

    def _assert_not_cancelled(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._cancelled_tasks:
                raise RuntimeError(f"pipeline task {task_id} was cancelled")


def _layers(stage: int) -> list[int]:
    return {
        1: list(range(0, 10)),
        2: list(range(10, 20)),
        3: list(range(20, 28)),
    }[stage]


def _prompt_graph(stage: int) -> str:
    return f"prompt_ar128_cl512_{stage + 1}_of_4"


def _decode_graph(stage: int) -> str:
    return f"token_ar1_cl512_{stage + 1}_of_4"


def _output_name(stage: int) -> str:
    return {1: "add_21844", 2: "add_42314", 3: "logits"}[stage]


def _output_shape(stage: int, *, prompt: bool) -> tuple[int, ...]:
    sequence = 128 if prompt else 1
    width = 151936 if stage == 3 else 2048
    return (1, sequence, width)


def _milliseconds(seconds: float) -> int:
    return max(1, round(float(seconds) * 1000))

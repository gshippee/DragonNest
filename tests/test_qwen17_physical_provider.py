from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from dragon_nest.artifacts import ArtifactNotFoundError, ModelArtifact, SplitBoundary
from dragon_nest.config import load_device
from dragon_nest.models import RuntimeName
from dragon_nest.proto import dragonnest_pb2 as pb
from dragon_nest.runtime.qwen17_kv import CONTEXT_LENGTH, StageKVBuffer
from dragon_nest.runtime.qwen17_provider import (
    PIPELINE_ID,
    Qwen17PipelineProvider,
)
from dragon_nest.transport.agent import DeviceAgent


ROOT = Path(__file__).resolve().parents[1]


def _artifact(stage: int) -> ModelArtifact:
    starts = (None, 0, 10, 20)
    ends = (None, 9, 19, 27)
    inputs = ("input_ids", "embedding", "add_21844", "add_42314")
    outputs = ("embedding", "add_21844", "add_42314", "logits")
    return ModelArtifact(
        model_id=f"qwen3-1.7b-s{stage}-xelite",
        artifact_id=f"artifact-s{stage}",
        model_version="demo-v1",
        runtime=RuntimeName.QNN,
        artifact_path=f"s{stage}.bin",
        checksum=f"sha256:{'0' * 64}",
        tokenizer_id="Qwen/Qwen3-1.7B",
        precision="w4a16-name-w8a16-compile-observed",
        supported_accelerators=("htp",),
        min_memory_mb=1,
        max_context_tokens=512,
        supports_steering=False,
        supports_data_parallel=False,
        supports_layer_pipeline=True,
        split_boundary=SplitBoundary(
            pipeline_id=PIPELINE_ID,
            stage_index=stage,
            stage_count=4,
            transformer_start_layer=starts[stage],
            transformer_end_layer=ends[stage],
            total_layers=28,
            input_tensor=inputs[stage],
            output_tensor=outputs[stage],
            includes_embedding=stage == 0,
            includes_lm_head=stage == 3,
            boundary_format="qnn-raw-tensor-v1",
        ),
        runtime_options={"runtime_version": "QAIRT-2.45"},
    )


class _Registry:
    def __init__(self, root: Path):
        self.root = root
        self.artifacts = {item.model_id: item for item in map(_artifact, range(4))}

    def validate(self, model_id: str) -> Path:
        return self.root / f"{model_id}.bin"

    def get(self, model_id: str) -> ModelArtifact:
        try:
            return self.artifacts[model_id]
        except KeyError as exc:
            raise ArtifactNotFoundError(model_id) from exc

    def all(self):
        return tuple(self.artifacts.values())

    def is_available(self, model_id: str) -> bool:
        return model_id in self.artifacts


class _Tokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    padding_side = "left"
    pad_token = "<eos>"
    pad_token_id = 0
    truncation_side = "left"

    def apply_chat_template(self, *args, **kwargs):
        return "formatted"

    def __call__(self, *args, **kwargs):
        input_ids = np.zeros((1, 512), dtype=np.int64)
        input_ids[0, -4:] = (1, 2, 3, 4)
        mask = np.zeros((1, 512), dtype=np.int64)
        mask[0, -4:] = 1
        return {"input_ids": input_ids, "attention_mask": mask}

    def decode(self, token_ids):
        return f"token-{token_ids[0]}"


class _Engine:
    CONTEXT_LENGTH = 512
    SEQUENCE_LENGTH = 128

    @staticmethod
    def ensure_binary_info(path, cache):
        return {"path": str(path)}

    @staticmethod
    def run_s0_prompt(path, metadata, input_ids):
        return np.zeros((1, 128, 2048), dtype=np.uint16), 0.001

    @staticmethod
    def run_s0_decode(path, metadata, token_id):
        return np.zeros((1, 1, 2048), dtype=np.uint16), 0.001

    @staticmethod
    def run_stage_prompt(
        path,
        metadata,
        graph,
        input_name,
        boundary,
        output_name,
        output_shape,
        layers,
        num_tokens,
        has_steering=False,
    ):
        past_key = {
            layer: np.full((8, 1, 128, 384), 128, dtype=np.uint8)
            for layer in layers
        }
        past_value = {
            layer: np.full((8, 1, 384, 128), 128, dtype=np.uint8)
            for layer in layers
        }
        outputs = {}
        for layer in layers:
            outputs[f"past_key_{layer}_out"] = np.full(
                (8, 1, 128, 128), layer + 1, dtype=np.uint8
            )
            outputs[f"past_value_{layer}_out"] = np.full(
                (8, 1, 128, 128), layer + 1, dtype=np.uint8
            )
        output = np.zeros(output_shape, dtype=np.uint16)
        if output_name == "logits":
            output[0, -1, 42] = 10
        return {
            "output_u16": output,
            "output_scale": 1.0,
            "output_offset": 0,
            "raw_outputs": outputs,
            "past_key_in": past_key,
            "past_value_in": past_value,
            "elapsed_sec": 0.001,
        }

    @staticmethod
    def run_stage_decode(
        path,
        metadata,
        graph,
        input_name,
        boundary,
        output_name,
        output_shape,
        kv,
        layers,
        rope_position,
        num_tokens,
        has_steering=False,
    ):
        for layer in layers:
            kv.update(
                layer,
                np.full((8, 1, 128, 1), 222, dtype=np.uint8),
                np.full((8, 1, 1, 128), 223, dtype=np.uint8),
            )
        output = np.zeros(output_shape, dtype=np.uint16)
        if output_name == "logits":
            output[0, -1, 43] = 10
        return {
            "output_u16": output,
            "output_scale": 1.0,
            "output_offset": 0,
            "elapsed_sec": 0.001,
        }

    @staticmethod
    def dequantize(raw, scale, offset):
        return raw.astype(np.float32)


def _command(stage: int, operation: int, *, token_id: int = 0) -> pb.ExecutePipelineStage:
    return pb.ExecutePipelineStage(
        task_id="task-physical",
        attempt_id=f"attempt-{stage}-{operation}",
        stage_id=f"stage-{stage}",
        stage_index=stage,
        stage_count=4,
        pipeline_id=PIPELINE_ID,
        model_id=f"qwen3-1.7b-s{stage}-xelite",
        request_text="What is gravity?",
        operation=operation,
        token_id=token_id,
        final_stage=stage == 3,
    )


def test_decode_delta_appends_without_replacing_511_token_history():
    kv = StageKVBuffer([10])
    kv.key[10][..., -1] = 77
    kv.value[10][:, :, -1, :] = 78

    kv.update(
        10,
        np.full((8, 1, 128, 1), 201, dtype=np.uint8),
        np.full((8, 1, 1, 128), 202, dtype=np.uint8),
    )

    assert kv.key[10].shape[3] == CONTEXT_LENGTH
    assert kv.value[10].shape[2] == CONTEXT_LENGTH
    assert np.all(kv.key[10][..., -2] == 77)
    assert np.all(kv.value[10][:, :, -2, :] == 78)
    assert np.all(kv.key[10][..., -1] == 201)
    assert np.all(kv.value[10][:, :, -1, :] == 202)


def test_provider_requires_prefill_and_cleans_reset_cancel(tmp_path: Path):
    provider = Qwen17PipelineProvider(
        _Registry(tmp_path), engine=_Engine, tokenizer_factory=_Tokenizer
    )
    with pytest.raises(RuntimeError, match="before prefill"):
        provider.execute(_command(0, pb.PIPELINE_DECODE), _artifact(0), None)

    boundaries: dict[int, np.ndarray] = {}
    for stage in range(4):
        result = provider.execute(
            _command(stage, pb.PIPELINE_PREFILL),
            _artifact(stage),
            boundaries.get(stage - 1),
        )
        if result.boundary is not None:
            boundaries[stage] = result.boundary
    assert len(provider) == 4
    assert result.next_token_id == 42

    for stage in range(4):
        provider.release("task-physical", PIPELINE_ID, stage)
    assert len(provider) == 0

    provider.execute(_command(0, pb.PIPELINE_PREFILL), _artifact(0), None)
    assert provider.cleanup_task("task-physical", cancelled=True) == 1
    assert len(provider) == 0
    with pytest.raises(RuntimeError, match="cancelled"):
        provider.execute(_command(0, pb.PIPELINE_DECODE), _artifact(0), None)


def test_registered_physical_pipeline_error_never_becomes_mock(tmp_path: Path):
    class _FailingProvider:
        def supports(self, artifact):
            return True

        def execute(self, command, artifact, boundary):
            raise RuntimeError("physical QNN failure")

    registry = _Registry(tmp_path)
    device = load_device(ROOT / "configs/hardware-fabric.yaml", "pc-01")
    agent = DeviceAgent(
        device,
        artifacts=registry,
        pipeline_provider=_FailingProvider(),
    )

    with pytest.raises(RuntimeError, match="physical QNN failure"):
        asyncio.run(
            agent._run_pipeline_stage(_command(0, pb.PIPELINE_PREFILL))
        )


def test_agent_stop_clears_physical_provider_sessions(tmp_path: Path):
    class _Provider:
        def __init__(self):
            self.clear_calls = 0

        def clear(self):
            self.clear_calls += 1
            return 4

    provider = _Provider()
    agent = DeviceAgent(
        load_device(ROOT / "configs/hardware-fabric.yaml", "pc-01"),
        artifacts=_Registry(tmp_path),
        pipeline_provider=provider,
    )

    asyncio.run(agent.stop())

    assert provider.clear_calls == 1


def test_cancel_waits_for_active_qnn_call_before_releasing_session(tmp_path: Path):
    entered = threading.Event()
    unblock = threading.Event()

    class _BlockingEngine(_Engine):
        @staticmethod
        def run_s0_prompt(path, metadata, input_ids):
            entered.set()
            assert unblock.wait(timeout=5)
            return np.zeros((1, 128, 2048), dtype=np.uint16), 0.001

    provider = Qwen17PipelineProvider(
        _Registry(tmp_path), engine=_BlockingEngine, tokenizer_factory=_Tokenizer
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            provider.execute,
            _command(0, pb.PIPELINE_PREFILL),
            _artifact(0),
            None,
        )
        assert entered.wait(timeout=5)
        assert provider.cleanup_task("task-physical", cancelled=True) == 0
        unblock.set()
        with pytest.raises(RuntimeError, match="cancelled"):
            future.result(timeout=5)

    assert len(provider) == 0

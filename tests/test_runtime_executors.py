from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import numpy as np

from dragon_nest.artifacts import ArtifactRegistry
from dragon_nest.executors import ExecutorDispatcher
from dragon_nest.models import (
    ExecutionMode,
    ExecutionPlan,
    PipelineStage,
    PlannedTask,
)
from dragon_nest.runtime.executors import (
    GenieExecutor,
    QnnExecutor,
    QnnPipelineExecutor,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_registry(tmp_path: Path) -> ArtifactRegistry:
    genie = tmp_path / "genie"
    genie.mkdir()
    (genie / "genie-t2t-run.exe").write_bytes(b"fake executable")
    (genie / "genie_config.json").write_text("{}", encoding="utf-8")
    part_a = tmp_path / "part_a.bin"
    part_b = tmp_path / "part_b.bin"
    part_a.write_bytes(b"part-a")
    part_b.write_bytes(b"part-b")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""
models:
  - model_id: genie-model
    model_version: genie-v1
    runtime: genie
    artifact_path: {genie}
    checksum: sha256-tree:{_tree_sha256(genie)}
    tokenizer_id: qwen-tokenizer
    precision: w4a16
    supported_accelerators: [htp]
    min_memory_mb: 1
    max_context_tokens: 128
    supports_steering: false
    supports_data_parallel: true
    supports_layer_pipeline: false
    runtime_options:
      backend: htp
      runtime_version: test-genie
      executable: genie-t2t-run.exe
      config: genie_config.json

  - model_id: part-a
    model_version: split-v1
    runtime: qnn
    artifact_path: {part_a}
    checksum: sha256:{_sha256(part_a)}
    tokenizer_id: split-tokenizer
    precision: fp16
    supported_accelerators: [htp]
    min_memory_mb: 1
    max_context_tokens: 8
    supports_steering: false
    supports_data_parallel: false
    supports_layer_pipeline: true
    split_boundary:
      pipeline_id: split
      start_layer: 0
      end_layer: 1
      total_layers: 2
      input_tensor: input_ids
      output_tensor: hidden
      includes_embedding: true
      includes_lm_head: false
      boundary_format: raw-fp32
    runtime_options:
      artifact_kind: context_binary
      backend: htp
      runtime_version: test-qnn
      outputs:
        - name: hidden
          shape: [1, 2]
          dtype: float32

  - model_id: part-b
    model_version: split-v1
    runtime: qnn
    artifact_path: {part_b}
    checksum: sha256:{_sha256(part_b)}
    tokenizer_id: split-tokenizer
    precision: fp16
    supported_accelerators: [htp]
    min_memory_mb: 1
    max_context_tokens: 8
    supports_steering: false
    supports_data_parallel: false
    supports_layer_pipeline: true
    split_boundary:
      pipeline_id: split
      start_layer: 1
      end_layer: 2
      total_layers: 2
      input_tensor: hidden
      output_tensor: logits
      includes_embedding: false
      includes_lm_head: true
      boundary_format: raw-fp32
    runtime_options:
      artifact_kind: context_binary
      backend: htp
      runtime_version: test-qnn
      outputs:
        - name: logits
          shape: [1, 3]
          dtype: float32
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return ArtifactRegistry.from_yaml(manifest)


def _single_plan(model_id: str) -> ExecutionPlan:
    return ExecutionPlan(
        task_id="task-1",
        execution_mode=ExecutionMode.SINGLE,
        request_text="Explain local AI.",
        tasks=(
            PlannedTask(
                shard_id="shard-1",
                request_text="Explain local AI.",
                selected_device_id="pc-01",
                selected_model_id=model_id,
            ),
        ),
    )


def _pipeline_plan() -> ExecutionPlan:
    return ExecutionPlan(
        task_id="task-pipeline",
        execution_mode=ExecutionMode.LAYER_PIPELINE,
        request_text="Complete this prompt.",
        stages=(
            PipelineStage("stage-1", 0, "split", "phone-01", "part-a", 0, 1),
            PipelineStage("stage-2", 1, "split", "pc-01", "part-b", 1, 2),
        ),
    )


def test_genie_executor_uses_manifest_bundle_and_reports_metrics(tmp_path: Path):
    registry = _runtime_registry(tmp_path)
    calls = []

    def fake_genie(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return "Local inference response"

    executor = GenieExecutor(registry, runner=fake_genie)
    result = asyncio.run(executor.execute(_single_plan("genie-model"), "attempt-fixed"))

    assert result.success
    assert result.output_text == "Local inference response"
    assert result.attempt_id == "attempt-fixed"
    assert result.metrics is not None
    assert result.metrics.runtime_name == "genie"
    assert result.metrics.model_version == "genie-v1"
    assert calls[0][1]["genie_dir"] == tmp_path / "genie"
    assert "Explain local AI." in calls[0][0]


def test_qnn_pipeline_passes_real_boundary_array_between_stages(tmp_path: Path):
    registry = _runtime_registry(tmp_path)
    calls: list[tuple[str, dict[str, np.ndarray]]] = []

    def fake_context(
        path, inputs, output_names, output_shapes, output_dtypes, **kwargs
    ):
        calls.append((Path(path).name, inputs))
        if Path(path).name == "part_a.bin":
            return {"hidden": np.array([[2.0, 3.0]], dtype=np.float32)}
        np.testing.assert_array_equal(
            inputs["hidden"], np.array([[2.0, 3.0]], dtype=np.float32)
        )
        return {"logits": np.array([[0.1, 0.2, 0.9]], dtype=np.float32)}

    qnn = QnnExecutor(registry, context_runner=fake_context)
    pipeline = QnnPipelineExecutor(
        qnn,
        output_formatter=lambda outputs, artifact: str(int(outputs["logits"].argmax())),
    )
    result = asyncio.run(
        pipeline.execute(
            _pipeline_plan(),
            initial_inputs={"input_ids": np.array([[4, 5]], dtype=np.int32)},
        )
    )

    assert result.success
    assert result.output_text == "2"
    assert [name for name, _ in calls] == ["part_a.bin", "part_b.bin"]
    assert result.metrics is not None
    assert result.metrics.runtime_name == "qnn"


def test_dispatcher_selects_registered_genie_runtime(tmp_path: Path):
    registry = _runtime_registry(tmp_path)

    def fake_genie(prompt, **kwargs):
        return "Dispatched through Genie"

    genie = GenieExecutor(registry, runner=fake_genie)
    qnn = QnnExecutor(registry)
    dispatcher = ExecutorDispatcher(
        artifacts=registry,
        genie=genie,
        qnn=qnn,
        qnn_pipeline=QnnPipelineExecutor(qnn),
    )

    result = asyncio.run(dispatcher.execute(_single_plan("genie-model")))

    assert result.success
    assert result.output_text == "Dispatched through Genie"
    assert result.metrics is not None
    assert result.metrics.runtime_name == "genie"


def test_qnn_executor_fails_cleanly_without_input_adapter(tmp_path: Path):
    registry = _runtime_registry(tmp_path)
    executor = QnnExecutor(registry)

    result = asyncio.run(executor.execute(_single_plan("part-a")))

    assert not result.success
    assert result.error_code == "MISSING_INPUT_ADAPTER"
    assert result.metrics is not None
    assert result.metrics.error_code == "MISSING_INPUT_ADAPTER"

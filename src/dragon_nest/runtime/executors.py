from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from ..artifacts import ArtifactError, ArtifactRegistry, ModelArtifact
from ..models import (
    ExecutionMetrics,
    ExecutionMode,
    ExecutionPlan,
    RuntimeName,
    TaskResult,
)
from . import genie_runner, qnn_runner


class InputBuilder(Protocol):
    def __call__(
        self, request_text: str, artifact: ModelArtifact
    ) -> Mapping[str, np.ndarray]: ...


class OutputFormatter(Protocol):
    def __call__(
        self, outputs: Mapping[str, np.ndarray], artifact: ModelArtifact
    ) -> str: ...


@dataclass(frozen=True)
class QnnGraphResult:
    model_id: str
    success: bool
    outputs: Mapping[str, np.ndarray] = field(default_factory=dict)
    error_code: str = ""
    error_message: str = ""
    attempt_id: str = ""
    metrics: ExecutionMetrics | None = None


class GenieExecutor:
    def __init__(
        self,
        artifacts: ArtifactRegistry,
        runner: Callable[..., str] = genie_runner.run_genie,
        verify_checksum: bool = True,
    ):
        self.artifacts = artifacts
        self.runner = runner
        self.verify_checksum = verify_checksum

    async def execute(
        self, plan: ExecutionPlan, attempt_id: str | None = None
    ) -> TaskResult:
        attempt_id = attempt_id or _new_attempt_id()
        if plan.execution_mode != ExecutionMode.SINGLE or not plan.tasks:
            return _task_failure(
                plan,
                attempt_id,
                "UNSUPPORTED_EXECUTION_MODE",
                "GenieExecutor requires one single-device task",
                runtime=RuntimeName.GENIE,
            )
        task = plan.tasks[0]
        try:
            artifact = self.artifacts.get(task.selected_model_id)
            if artifact.runtime != RuntimeName.GENIE:
                raise ArtifactError(
                    f"{artifact.model_id}: expected genie runtime, got {artifact.runtime}"
                )
            bundle = self.artifacts.validate(
                artifact.model_id, verify_checksum=self.verify_checksum
            )
        except ArtifactError as exc:
            return _task_failure(
                plan,
                attempt_id,
                "ARTIFACT_INVALID",
                str(exc),
                model_id=task.selected_model_id,
                runtime=RuntimeName.GENIE,
                device_id=task.selected_device_id,
            )

        options = artifact.runtime_options
        system_prompt = str(
            options.get("system_prompt", "You are a helpful AI assistant.")
        )
        prompt = genie_runner.build_chatml_prompt(system_prompt, task.request_text)
        start = time.perf_counter()
        try:
            output = await asyncio.to_thread(
                self.runner,
                prompt,
                backend=str(options.get("backend", "htp")),
                timeout_sec=float(options.get("timeout_sec", 180)),
                genie_dir=bundle,
                genie_executable=_optional_bundle_path(
                    bundle, options.get("executable")
                ),
                genie_config=options.get("config", "genie_config.json"),
            )
        except Exception as exc:
            latency = _elapsed_ms(start)
            return _task_failure(
                plan,
                attempt_id,
                "RUNTIME_ERROR",
                str(exc),
                artifact=artifact,
                runtime=RuntimeName.GENIE,
                latency_ms=latency,
                device_id=task.selected_device_id,
            )
        latency = _elapsed_ms(start)
        metrics = _metrics(artifact, RuntimeName.GENIE, latency)
        return TaskResult(
            task_id=plan.task_id,
            attempt_id=attempt_id,
            success=True,
            output_text=output,
            device_id=task.selected_device_id,
            latency_ms=latency,
            metrics=metrics,
        )


class QnnExecutor:
    def __init__(
        self,
        artifacts: ArtifactRegistry,
        dlc_runner: Callable[..., Mapping[str, np.ndarray]] = qnn_runner.run_dlc,
        context_runner: Callable[
            ..., Mapping[str, np.ndarray]
        ] = qnn_runner.run_context_binary,
        input_builder: InputBuilder | None = None,
        output_formatter: OutputFormatter | None = None,
        verify_checksum: bool = True,
    ):
        self.artifacts = artifacts
        self.dlc_runner = dlc_runner
        self.context_runner = context_runner
        self.input_builder = input_builder
        self.output_formatter = output_formatter
        self.verify_checksum = verify_checksum

    async def execute(
        self, plan: ExecutionPlan, attempt_id: str | None = None
    ) -> TaskResult:
        attempt_id = attempt_id or _new_attempt_id()
        if plan.execution_mode != ExecutionMode.SINGLE or not plan.tasks:
            return _task_failure(
                plan,
                attempt_id,
                "UNSUPPORTED_EXECUTION_MODE",
                "QnnExecutor requires one single-device task",
                runtime=RuntimeName.QNN,
            )
        task = plan.tasks[0]
        try:
            artifact = self.artifacts.get(task.selected_model_id)
            if artifact.runtime != RuntimeName.QNN:
                raise ArtifactError(
                    f"{artifact.model_id}: expected qnn runtime, got {artifact.runtime}"
                )
            self.artifacts.validate(
                artifact.model_id, verify_checksum=self.verify_checksum
            )
        except ArtifactError as exc:
            return _task_failure(
                plan,
                attempt_id,
                "ARTIFACT_INVALID",
                str(exc),
                model_id=task.selected_model_id,
                runtime=RuntimeName.QNN,
                device_id=task.selected_device_id,
            )
        if self.input_builder is None:
            return _task_failure(
                plan,
                attempt_id,
                "MISSING_INPUT_ADAPTER",
                f"{artifact.model_id}: no text-to-tensor input adapter is configured",
                artifact=artifact,
                runtime=RuntimeName.QNN,
                device_id=task.selected_device_id,
            )
        try:
            inputs = self.input_builder(task.request_text, artifact)
        except Exception as exc:
            return _task_failure(
                plan,
                attempt_id,
                "INPUT_ADAPTER_ERROR",
                str(exc),
                artifact=artifact,
                runtime=RuntimeName.QNN,
                device_id=task.selected_device_id,
            )
        graph_result = await self.execute_graph(artifact.model_id, inputs, attempt_id)
        if not graph_result.success:
            return TaskResult(
                task_id=plan.task_id,
                attempt_id=attempt_id,
                success=False,
                output_text="",
                device_id=task.selected_device_id,
                error_code=graph_result.error_code,
                error_message=graph_result.error_message,
                latency_ms=(
                    graph_result.metrics.execution_latency_ms
                    if graph_result.metrics
                    else 0
                ),
                metrics=graph_result.metrics,
            )
        try:
            output = (
                self.output_formatter(graph_result.outputs, artifact)
                if self.output_formatter
                else _describe_outputs(graph_result.outputs)
            )
        except Exception as exc:
            return _task_failure(
                plan,
                attempt_id,
                "OUTPUT_ADAPTER_ERROR",
                str(exc),
                artifact=artifact,
                runtime=RuntimeName.QNN,
                latency_ms=(
                    graph_result.metrics.execution_latency_ms
                    if graph_result.metrics
                    else 0
                ),
                device_id=task.selected_device_id,
            )
        return TaskResult(
            task_id=plan.task_id,
            attempt_id=attempt_id,
            success=True,
            output_text=output,
            device_id=task.selected_device_id,
            latency_ms=(
                graph_result.metrics.execution_latency_ms if graph_result.metrics else 0
            ),
            metrics=graph_result.metrics,
        )

    async def execute_graph(
        self,
        model_id: str,
        inputs: Mapping[str, np.ndarray],
        attempt_id: str | None = None,
    ) -> QnnGraphResult:
        attempt_id = attempt_id or _new_attempt_id()
        try:
            artifact = self.artifacts.get(model_id)
            if artifact.runtime != RuntimeName.QNN:
                raise ArtifactError(
                    f"{artifact.model_id}: expected qnn runtime, got {artifact.runtime}"
                )
            artifact_path = self.artifacts.validate(
                model_id, verify_checksum=self.verify_checksum
            )
            output_names, output_shapes, output_dtypes = _qnn_outputs(artifact)
            runner = _qnn_runner_for(artifact, self.dlc_runner, self.context_runner)
        except (ArtifactError, TypeError, ValueError) as exc:
            return QnnGraphResult(
                model_id=model_id,
                attempt_id=attempt_id,
                success=False,
                error_code="ARTIFACT_INVALID",
                error_message=str(exc),
                metrics=_unknown_metrics(
                    model_id, RuntimeName.QNN, "ARTIFACT_INVALID", str(exc)
                ),
            )

        options = artifact.runtime_options
        start = time.perf_counter()
        try:
            outputs = await asyncio.to_thread(
                runner,
                artifact_path,
                dict(inputs),
                output_names,
                output_shapes,
                output_dtypes,
                backend=str(options.get("backend", "htp")),
                timeout_sec=float(options.get("timeout_sec", 120)),
            )
        except Exception as exc:
            latency = _elapsed_ms(start)
            message = str(exc)
            return QnnGraphResult(
                model_id=model_id,
                attempt_id=attempt_id,
                success=False,
                error_code="RUNTIME_ERROR",
                error_message=message,
                metrics=_metrics(
                    artifact, RuntimeName.QNN, latency, "RUNTIME_ERROR", message
                ),
            )
        latency = _elapsed_ms(start)
        return QnnGraphResult(
            model_id=model_id,
            attempt_id=attempt_id,
            success=True,
            outputs=outputs,
            metrics=_metrics(artifact, RuntimeName.QNN, latency),
        )


class QnnPipelineExecutor:
    def __init__(
        self,
        qnn: QnnExecutor,
        input_builder: InputBuilder | None = None,
        output_formatter: OutputFormatter | None = None,
    ):
        self.qnn = qnn
        self.input_builder = input_builder
        self.output_formatter = output_formatter

    async def execute(
        self,
        plan: ExecutionPlan,
        initial_inputs: Mapping[str, np.ndarray] | None = None,
    ) -> TaskResult:
        if plan.execution_mode != ExecutionMode.LAYER_PIPELINE or not plan.stages:
            return _task_failure(
                plan,
                _new_attempt_id(),
                "UNSUPPORTED_EXECUTION_MODE",
                "QnnPipelineExecutor requires a routed layer-pipeline plan",
                runtime=RuntimeName.QNN,
            )

        try:
            artifacts = [
                self.qnn.artifacts.get(stage.selected_model_id) for stage in plan.stages
            ]
            _validate_pipeline_artifacts(artifacts)
        except ArtifactError as exc:
            return _task_failure(
                plan,
                _new_attempt_id(),
                "INCOMPATIBLE_PIPELINE",
                str(exc),
                runtime=RuntimeName.QNN,
            )

        if initial_inputs is None:
            if self.input_builder is None:
                return _task_failure(
                    plan,
                    _new_attempt_id(),
                    "MISSING_INPUT_ADAPTER",
                    "layer pipeline requires initial tensors or an input builder",
                    artifact=artifacts[0],
                    runtime=RuntimeName.QNN,
                )
            initial_inputs = self.input_builder(plan.request_text, artifacts[0])

        stage_inputs: Mapping[str, np.ndarray] = initial_inputs
        total_latency = 0
        last_outputs: Mapping[str, np.ndarray] = {}
        attempt_ids: list[str] = []
        for index, (stage, artifact) in enumerate(
            zip(plan.stages, artifacts, strict=True)
        ):
            attempt_id = _new_attempt_id()
            attempt_ids.append(attempt_id)
            result = await self.qnn.execute_graph(
                artifact.model_id, stage_inputs, attempt_id
            )
            if result.metrics:
                total_latency += result.metrics.execution_latency_ms
            if not result.success:
                return TaskResult(
                    task_id=plan.task_id,
                    attempt_id=attempt_id,
                    success=False,
                    output_text="",
                    device_id=stage.selected_device_id,
                    error_code=result.error_code,
                    error_message=result.error_message,
                    latency_ms=total_latency,
                    metrics=result.metrics,
                )
            last_outputs = result.outputs
            if index + 1 < len(artifacts):
                current = artifact.split_boundary
                following = artifacts[index + 1].split_boundary
                assert current is not None and following is not None
                try:
                    boundary = result.outputs[current.output_tensor]
                except KeyError:
                    return _task_failure(
                        plan,
                        attempt_id,
                        "BOUNDARY_TENSOR_MISSING",
                        f"{artifact.model_id} did not return {current.output_tensor}",
                        artifact=artifact,
                        runtime=RuntimeName.QNN,
                        latency_ms=total_latency,
                        device_id=stage.selected_device_id,
                    )
                stage_inputs = {following.input_tensor: boundary}

        final_artifact = artifacts[-1]
        try:
            output_text = (
                self.output_formatter(last_outputs, final_artifact)
                if self.output_formatter
                else _describe_outputs(last_outputs)
            )
        except Exception as exc:
            return _task_failure(
                plan,
                attempt_ids[-1],
                "OUTPUT_ADAPTER_ERROR",
                str(exc),
                artifact=final_artifact,
                runtime=RuntimeName.QNN,
                latency_ms=total_latency,
                device_id=plan.stages[-1].selected_device_id,
            )
        metrics = _metrics(final_artifact, RuntimeName.QNN, total_latency)
        return TaskResult(
            task_id=plan.task_id,
            attempt_id=attempt_ids[-1],
            success=True,
            output_text=output_text,
            device_id=plan.stages[-1].selected_device_id,
            latency_ms=total_latency,
            metrics=metrics,
        )


def _validate_pipeline_artifacts(artifacts: list[ModelArtifact]) -> None:
    if len(artifacts) < 2:
        raise ArtifactError("layer pipeline requires at least two model artifacts")
    previous: ModelArtifact | None = None
    indexed = all(
        artifact.split_boundary is not None
        and artifact.split_boundary.stage_count > 0
        for artifact in artifacts
    )
    if indexed:
        splits = [artifact.split_boundary for artifact in artifacts]
        assert all(split is not None for split in splits)
        concrete = [split for split in splits if split is not None]
        stage_count = concrete[0].stage_count
        if len(artifacts) != stage_count or [
            split.stage_index for split in concrete
        ] != list(range(stage_count)):
            raise ArtifactError("pipeline must contain every indexed stage exactly once")
    for artifact in artifacts:
        split = artifact.split_boundary
        if (
            artifact.runtime != RuntimeName.QNN
            or not artifact.supports_layer_pipeline
            or not split
        ):
            raise ArtifactError(
                f"{artifact.model_id}: not a QNN layer-pipeline artifact"
            )
        if previous:
            left = previous.split_boundary
            assert left is not None
            if left.pipeline_id != split.pipeline_id:
                raise ArtifactError(
                    f"{previous.model_id} and {artifact.model_id}: non-contiguous pipeline"
                )
            if indexed:
                if left.output_tensor != split.input_tensor:
                    raise ArtifactError(
                        f"{previous.model_id} and {artifact.model_id}: boundary tensor mismatch"
                    )
            elif left.end_layer != split.start_layer:
                raise ArtifactError(
                    f"{previous.model_id} and {artifact.model_id}: non-contiguous pipeline"
                )
            if left.total_layers != split.total_layers:
                raise ArtifactError("pipeline total layer counts do not match")
            if previous.model_version != artifact.model_version:
                raise ArtifactError("pipeline model versions do not match")
            if previous.tokenizer_id != artifact.tokenizer_id:
                raise ArtifactError("pipeline tokenizers do not match")
            if previous.precision != artifact.precision:
                raise ArtifactError("pipeline precisions do not match")
            if left.boundary_format != split.boundary_format:
                raise ArtifactError("pipeline boundary formats do not match")
        previous = artifact
    first = artifacts[0].split_boundary
    last = artifacts[-1].split_boundary
    assert first is not None and last is not None
    if not first.includes_embedding or (
        not indexed and first.start_layer != 0
    ):
        raise ArtifactError("pipeline does not start with embeddings")
    if not last.includes_lm_head or (
        not indexed and last.end_layer != last.total_layers
    ):
        raise ArtifactError("pipeline does not end with the LM head")


def _qnn_outputs(
    artifact: ModelArtifact,
) -> tuple[list[str], dict[str, tuple[int, ...]], dict[str, np.dtype[Any]]]:
    raw_outputs = artifact.runtime_options.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ArtifactError(f"{artifact.model_id}: runtime_options.outputs is required")
    names: list[str] = []
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, np.dtype[Any]] = {}
    for item in raw_outputs:
        if not isinstance(item, Mapping):
            raise ArtifactError(
                f"{artifact.model_id}: invalid output tensor specification"
            )
        name = str(item["name"])
        names.append(name)
        shapes[name] = tuple(int(value) for value in item["shape"])
        dtypes[name] = np.dtype(str(item["dtype"]))
    return names, shapes, dtypes


def _qnn_runner_for(
    artifact: ModelArtifact,
    dlc_runner: Callable[..., Mapping[str, np.ndarray]],
    context_runner: Callable[..., Mapping[str, np.ndarray]],
) -> Callable[..., Mapping[str, np.ndarray]]:
    kind = str(artifact.runtime_options.get("artifact_kind", "")).lower()
    if kind == "dlc" or (not kind and artifact.artifact_path.lower().endswith(".dlc")):
        return dlc_runner
    if kind in {"context_binary", "bin"} or (
        not kind and artifact.artifact_path.lower().endswith(".bin")
    ):
        return context_runner
    raise ArtifactError(
        f"{artifact.model_id}: runtime_options.artifact_kind must be dlc or context_binary"
    )


def _metrics(
    artifact: ModelArtifact,
    runtime: RuntimeName,
    latency_ms: int,
    error_code: str = "",
    error_message: str = "",
) -> ExecutionMetrics:
    options = artifact.runtime_options
    return ExecutionMetrics(
        model_id=artifact.model_id,
        model_version=artifact.model_version,
        runtime_name=runtime.value,
        runtime_version=str(options.get("runtime_version", "unknown")),
        accelerator=str(options.get("backend", artifact.supported_accelerators[0])),
        execution_latency_ms=latency_ms,
        error_code=error_code,
        error_message=error_message,
    )


def _unknown_metrics(
    model_id: str,
    runtime: RuntimeName,
    error_code: str,
    error_message: str,
) -> ExecutionMetrics:
    return ExecutionMetrics(
        model_id=model_id,
        model_version="unknown",
        runtime_name=runtime.value,
        runtime_version="unknown",
        accelerator="unknown",
        execution_latency_ms=0,
        error_code=error_code,
        error_message=error_message,
    )


def _task_failure(
    plan: ExecutionPlan,
    attempt_id: str,
    error_code: str,
    error_message: str,
    artifact: ModelArtifact | None = None,
    model_id: str = "",
    runtime: RuntimeName = RuntimeName.MOCK,
    latency_ms: int = 0,
    device_id: str = "",
) -> TaskResult:
    metrics = (
        _metrics(artifact, runtime, latency_ms, error_code, error_message)
        if artifact
        else _unknown_metrics(model_id, runtime, error_code, error_message)
    )
    return TaskResult(
        task_id=plan.task_id,
        attempt_id=attempt_id,
        success=False,
        output_text="",
        device_id=device_id,
        error_code=error_code,
        error_message=error_message,
        latency_ms=latency_ms,
        metrics=metrics,
    )


def _optional_bundle_path(bundle: Path, value: object) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else bundle / path


def _describe_outputs(outputs: Mapping[str, np.ndarray]) -> str:
    tensors = ", ".join(
        f"{name}:shape={tuple(value.shape)},dtype={value.dtype}"
        for name, value in outputs.items()
    )
    return f"QNN execution completed ({tensors})"


def _new_attempt_id() -> str:
    return f"attempt-{uuid.uuid4().hex}"


def _elapsed_ms(start: float) -> int:
    return max(1, round((time.perf_counter() - start) * 1000))

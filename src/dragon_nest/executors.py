from __future__ import annotations

import asyncio
import uuid

from .artifacts import ArtifactNotFoundError, ArtifactRegistry
from .models import (
    ExecutionMetrics,
    ExecutionMode,
    ExecutionPlan,
    RuntimeName,
    TaskResult,
)
from .runtime.executors import GenieExecutor, QnnExecutor, QnnPipelineExecutor


class MockExecutor:
    async def execute(self, plan: ExecutionPlan) -> TaskResult:
        if plan.execution_mode == "data_parallel":
            return await self._execute_data_parallel(plan)
        if plan.execution_mode == "layer_pipeline":
            return await self._execute_pipeline(plan)
        return await self._execute_single(plan)

    async def _execute_single(self, plan: ExecutionPlan) -> TaskResult:
        task = plan.tasks[0]
        await asyncio.sleep(0.01)
        steering = self._steering_text(plan)
        attempt_id = _attempt_id()
        return TaskResult(
            task_id=plan.task_id,
            attempt_id=attempt_id,
            success=True,
            device_id=task.selected_device_id,
            output_text=(
                f"[Mock single result from {task.selected_device_id}/{task.selected_model_id}]\n"
                f"{steering}\n"
                "Response: execution completed successfully."
            ),
            latency_ms=10,
            metrics=_mock_metrics(task.selected_model_id, 10),
        )

    async def _execute_data_parallel(self, plan: ExecutionPlan) -> TaskResult:
        await asyncio.sleep(0.01)
        attempt_id = _attempt_id()
        parts = [
            f"{task.shard_id}:{task.selected_device_id}/{task.selected_model_id}"
            for task in plan.tasks
        ]
        return TaskResult(
            task_id=plan.task_id,
            attempt_id=attempt_id,
            success=True,
            device_id="reducer",
            output_text=(
                "[Mock data-parallel result]\n"
                f"Reducer: {plan.reducer}\n"
                f"Shards: {', '.join(parts)}\n"
                f"{self._steering_text(plan)}"
            ),
            latency_ms=10,
            metrics=_mock_metrics("mock-reducer", 10),
        )

    async def _execute_pipeline(self, plan: ExecutionPlan) -> TaskResult:
        await asyncio.sleep(0.01)
        attempt_id = _attempt_id()
        stages = [
            f"{stage.stage_id}:{stage.selected_device_id}/{stage.selected_model_id}"
            f"[{stage.start_layer}..{stage.end_layer}]"
            for stage in plan.stages
        ]
        return TaskResult(
            task_id=plan.task_id,
            attempt_id=attempt_id,
            success=True,
            device_id=plan.stages[-1].selected_device_id if plan.stages else "",
            output_text=(
                "[Mock layer-pipeline result]\n"
                f"Stages: {' -> '.join(stages)}\n"
                f"{self._steering_text(plan)}\n"
                "Boundary: mock hidden tensor metadata exchanged."
            ),
            latency_ms=10,
            metrics=_mock_metrics(
                plan.stages[-1].selected_model_id if plan.stages else "mock-pipeline",
                10,
            ),
        )

    def _steering_text(self, plan: ExecutionPlan) -> str:
        if not plan.steering.enabled:
            return "Steering: disabled"
        return (
            "Steering: "
            f"{plan.steering.vector_id} alpha={plan.steering.alpha} "
            f"layer={plan.steering.target_layer} positions={plan.steering.positions}"
        )


class ExecutorDispatcher:
    """Select a concrete executor from the routed model's artifact manifest."""

    def __init__(
        self,
        artifacts: ArtifactRegistry | None = None,
        mock: MockExecutor | None = None,
        genie: GenieExecutor | None = None,
        qnn: QnnExecutor | None = None,
        qnn_pipeline: QnnPipelineExecutor | None = None,
    ):
        self.artifacts = artifacts
        self.mock = mock or MockExecutor()
        self.genie = genie or (GenieExecutor(artifacts) if artifacts else None)
        self.qnn = qnn or (QnnExecutor(artifacts) if artifacts else None)
        self.qnn_pipeline = qnn_pipeline or (
            QnnPipelineExecutor(self.qnn) if self.qnn else None
        )

    async def execute(self, plan: ExecutionPlan) -> TaskResult:
        if self.artifacts is None:
            return await self.mock.execute(plan)
        if plan.execution_mode == ExecutionMode.LAYER_PIPELINE:
            if plan.stages and self._all_registered(
                stage.selected_model_id for stage in plan.stages
            ):
                assert self.qnn_pipeline is not None
                return await self.qnn_pipeline.execute(plan)
            return await self.mock.execute(plan)
        if plan.execution_mode == ExecutionMode.DATA_PARALLEL:
            return await self.mock.execute(plan)
        if not plan.tasks:
            return await self.mock.execute(plan)
        try:
            artifact = self.artifacts.get(plan.tasks[0].selected_model_id)
        except ArtifactNotFoundError:
            return await self.mock.execute(plan)
        if artifact.runtime == RuntimeName.GENIE:
            assert self.genie is not None
            return await self.genie.execute(plan)
        if artifact.runtime == RuntimeName.QNN:
            assert self.qnn is not None
            return await self.qnn.execute(plan)
        return await self.mock.execute(plan)

    def _all_registered(self, model_ids) -> bool:
        try:
            for model_id in model_ids:
                self.artifacts.get(model_id)
        except ArtifactNotFoundError:
            return False
        return True


def _attempt_id() -> str:
    return f"attempt-{uuid.uuid4().hex}"


def _mock_metrics(model_id: str, latency_ms: int) -> ExecutionMetrics:
    return ExecutionMetrics(
        model_id=model_id,
        model_version="mock-v1",
        runtime_name=RuntimeName.MOCK.value,
        runtime_version="dragon-nest-0.1.0",
        accelerator="cpu",
        execution_latency_ms=latency_ms,
    )

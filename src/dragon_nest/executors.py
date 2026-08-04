from __future__ import annotations

import asyncio

from .models import ExecutionPlan, TaskResult


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
        return TaskResult(
            task_id=plan.task_id,
            success=True,
            device_id=task.selected_device_id,
            output_text=(
                f"[Mock single result from {task.selected_device_id}/{task.selected_model_id}]\n"
                f"{steering}\n"
                "Response: execution completed successfully."
            ),
            latency_ms=10,
        )

    async def _execute_data_parallel(self, plan: ExecutionPlan) -> TaskResult:
        await asyncio.sleep(0.01)
        parts = [
            f"{task.shard_id}:{task.selected_device_id}/{task.selected_model_id}"
            for task in plan.tasks
        ]
        return TaskResult(
            task_id=plan.task_id,
            success=True,
            device_id="reducer",
            output_text=(
                "[Mock data-parallel result]\n"
                f"Reducer: {plan.reducer}\n"
                f"Shards: {', '.join(parts)}\n"
                f"{self._steering_text(plan)}"
            ),
            latency_ms=10,
        )

    async def _execute_pipeline(self, plan: ExecutionPlan) -> TaskResult:
        await asyncio.sleep(0.01)
        stages = [
            f"{stage.stage_id}:{stage.selected_device_id}/{stage.selected_model_id}"
            f"[{stage.start_layer}..{stage.end_layer}]"
            for stage in plan.stages
        ]
        return TaskResult(
            task_id=plan.task_id,
            success=True,
            device_id=plan.stages[-1].selected_device_id if plan.stages else "",
            output_text=(
                "[Mock layer-pipeline result]\n"
                f"Stages: {' -> '.join(stages)}\n"
                f"{self._steering_text(plan)}\n"
                "Boundary: mock hidden tensor metadata exchanged."
            ),
            latency_ms=10,
        )

    def _steering_text(self, plan: ExecutionPlan) -> str:
        if not plan.steering.enabled:
            return "Steering: disabled"
        return (
            "Steering: "
            f"{plan.steering.vector_id} alpha={plan.steering.alpha} "
            f"layer={plan.steering.target_layer} positions={plan.steering.positions}"
        )


from __future__ import annotations

import itertools
import re
import uuid

from .models import ExecutionMode, ExecutionPlan, PlannedTask, SteeringSpec, TaskProfile


class ExecutionPlanner:
    def plan(
        self,
        request_text: str,
        profile: TaskProfile,
        preferred_mode: str = "auto",
        requested_execution_mode: str = "auto",
        steering: SteeringSpec | None = None,
    ) -> ExecutionPlan:
        mode = self._choose_mode(profile, preferred_mode, requested_execution_mode)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        steering = steering or SteeringSpec(enabled=False)
        reasons: list[str] = []

        if mode == ExecutionMode.DATA_PARALLEL:
            shards = self._split_shards(request_text)
            reasons.append(f"Selected data_parallel: request split into {len(shards)} shard(s).")
            tasks = tuple(
                PlannedTask(shard_id=f"shard-{idx + 1}", request_text=shard)
                for idx, shard in enumerate(shards)
            )
            return ExecutionPlan(
                task_id=task_id,
                execution_mode=mode,
                request_text=request_text,
                tasks=tasks,
                steering=steering,
                reducer="mock_synthesis",
                reasons=tuple(reasons),
            )

        if mode == ExecutionMode.LAYER_PIPELINE:
            reasons.append("Selected layer_pipeline: request prefers quality and can use split model segments.")
            return ExecutionPlan(
                task_id=task_id,
                execution_mode=mode,
                request_text=request_text,
                steering=steering,
                reasons=tuple(reasons),
            )

        reasons.append("Selected single: request does not require parallel execution.")
        return ExecutionPlan(
            task_id=task_id,
            execution_mode=ExecutionMode.SINGLE,
            request_text=request_text,
            tasks=(PlannedTask(shard_id="shard-1", request_text=request_text),),
            steering=steering,
            reasons=tuple(reasons),
        )

    def _choose_mode(
        self,
        profile: TaskProfile,
        preferred_mode: str,
        requested_execution_mode: str,
    ) -> ExecutionMode:
        if requested_execution_mode != "auto":
            return ExecutionMode(requested_execution_mode)
        if preferred_mode == "private":
            return ExecutionMode.SINGLE
        if profile.data_parallelizable:
            return ExecutionMode.DATA_PARALLEL
        if profile.layer_parallel_candidate:
            return ExecutionMode.LAYER_PIPELINE
        return ExecutionMode.SINGLE

    def _split_shards(self, request_text: str) -> list[str]:
        section_refs = re.findall(r"section\s+\d+", request_text, flags=re.IGNORECASE)
        if len(section_refs) > 1:
            return [f"{request_text} [{section_ref}]" for section_ref in section_refs]
        numbered = [
            part.strip()
            for part in re.split(r"(?:^|\n)\s*\d+[.)]\s*", request_text)
            if part.strip()
        ]
        if len(numbered) > 1:
            return numbered
        for delimiter in ("; then ", ";", " and also "):
            if delimiter in request_text:
                return [part.strip() for part in request_text.split(delimiter) if part.strip()]
        if "sections" in request_text.lower():
            return [f"{request_text} [section {idx}]" for idx in range(1, 4)]
        return list(itertools.islice([request_text], 1))

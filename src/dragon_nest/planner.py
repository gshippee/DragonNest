from __future__ import annotations

import itertools
import re
import uuid

from .models import (
    ComputePreference,
    ExecutionMode,
    ExecutionPlan,
    PlannedTask,
    ReducerMode,
    SteeringSpec,
    TaskProfile,
)


ELASTIC_PIPELINE_ID = "qwen3-1.7b-w4a16-demo-v1"


class ExecutionPlanner:
    def plan(
        self,
        request_text: str,
        profile: TaskProfile,
        preferred_mode: str = "auto",
        requested_execution_mode: str = "auto",
        steering: SteeringSpec | None = None,
        origin_device_id: str = "",
        behavior_profile_id: str = "",
        profile_realization: str = "none",
        reducer: str = ReducerMode.MOCK_SYNTHESIS.value,
    ) -> ExecutionPlan:
        mode = self._choose_mode(profile, preferred_mode, requested_execution_mode)
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        steering = steering or SteeringSpec(enabled=False)
        reasons: list[str] = []

        if mode == ExecutionMode.DATA_PARALLEL:
            if reducer == ReducerMode.FIRST_SUCCESS:
                shards = [request_text]
                reasons.append(
                    "Selected replica_race: the first successful replica result wins."
                )
            else:
                shards = self._split_shards(request_text)
                reasons.append(
                    f"Selected data_parallel: request split into {len(shards)} shard(s)."
                )
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
                origin_device_id=origin_device_id,
                preferred_mode=preferred_mode,
                behavior_profile_id=behavior_profile_id,
                profile_realization=profile_realization,
                reducer=reducer,
                reasons=tuple(reasons),
            )

        if mode == ExecutionMode.LAYER_PIPELINE:
            pipeline_id = (
                ELASTIC_PIPELINE_ID
                if preferred_mode == ComputePreference.ELASTIC.value
                else ""
            )
            reasons.append(
                f"Selected layer_pipeline: {preferred_mode} requested the "
                f"{pipeline_id or 'compatible'} distributed pipeline."
            )
            return ExecutionPlan(
                task_id=task_id,
                execution_mode=mode,
                request_text=request_text,
                steering=steering,
                origin_device_id=origin_device_id,
                preferred_mode=preferred_mode,
                pipeline_id=pipeline_id,
                behavior_profile_id=behavior_profile_id,
                profile_realization=profile_realization,
                reducer=reducer,
                reasons=tuple(reasons),
            )

        reasons.append("Selected single: request does not require parallel execution.")
        return ExecutionPlan(
            task_id=task_id,
            execution_mode=ExecutionMode.SINGLE,
            request_text=request_text,
            tasks=(PlannedTask(shard_id="shard-1", request_text=request_text),),
            steering=steering,
            origin_device_id=origin_device_id,
            preferred_mode=preferred_mode,
            behavior_profile_id=behavior_profile_id,
            profile_realization=profile_realization,
            reducer=reducer,
            reasons=tuple(reasons),
        )

    def _choose_mode(
        self,
        profile: TaskProfile,
        preferred_mode: str,
        requested_execution_mode: str,
    ) -> ExecutionMode:
        if preferred_mode in {"private", "local", "quality"}:
            return ExecutionMode.SINGLE
        if preferred_mode == "elastic":
            return ExecutionMode.LAYER_PIPELINE
        if preferred_mode == "parallel":
            return ExecutionMode.DATA_PARALLEL
        if requested_execution_mode != "auto":
            return ExecutionMode(requested_execution_mode)
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

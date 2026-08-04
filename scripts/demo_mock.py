from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dragon_nest.classifier import RuleBasedTaskClassifier
from dragon_nest.executors import MockExecutor
from dragon_nest.models import Device, HealthState, ModelCapability, ModelSegment
from dragon_nest.planner import ExecutionPlanner
from dragon_nest.router import DeterministicRouter
from dragon_nest.steering import SteeringRegistry


def load_devices() -> list[Device]:
    raw = yaml.safe_load((ROOT / "configs/dev-fabric.yaml").read_text(encoding="utf-8"))
    devices = []
    for item in raw["devices"]:
        models = []
        for model in item["models"]:
            segment = None
            if "segment" in model:
                segment = ModelSegment(**model["segment"])
            models.append(
                ModelCapability(
                    model_id=model["model_id"],
                    model_family=model["model_family"],
                    role=model["role"],
                    task_classes=tuple(model["task_classes"]),
                    max_context_tokens=int(model["max_context_tokens"]),
                    warm=bool(model["warm"]),
                    quality_score=float(model["quality_score"]),
                    steering_vector_ids=tuple(model.get("steering_vector_ids", [])),
                    supported_steering_layers=tuple(model.get("supported_steering_layers", [])),
                    segment=segment,
                )
            )
        devices.append(
            Device(
                device_id=item["device_id"],
                display_name=item["display_name"],
                device_type=item["device_type"],
                platform=item["platform"],
                total_memory_mb=int(item["total_memory_mb"]),
                health=HealthState(**item["health"]),
                models=tuple(models),
            )
        )
    return devices


async def main() -> None:
    classifier = RuleBasedTaskClassifier()
    planner = ExecutionPlanner()
    steering = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    router = DeterministicRouter(steering)
    executor = MockExecutor()
    devices = load_devices()

    request = "Summarize section 1, section 2, and section 3, then give concise key points."
    profile = classifier.classify(request, preferred_mode="parallel")
    steering_spec = steering.default_spec("concise-vs-verbose-layer-7")
    plan = planner.plan(
        request,
        profile,
        preferred_mode="parallel",
        requested_execution_mode="data_parallel",
        steering=steering_spec,
    )
    routed_plan, decision = router.route(plan, profile, devices)
    result = await executor.execute(routed_plan)

    print("DragonNest mock demo")
    print("====================")
    print(f"Task class: {profile.task_class}")
    print(f"Execution mode: {decision.execution_mode}")
    print("Reasons:")
    for reason in decision.reasons:
        print(f"- {reason}")
    print()
    print(result.output_text)


if __name__ == "__main__":
    asyncio.run(main())

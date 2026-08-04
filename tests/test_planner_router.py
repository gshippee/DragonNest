from dragon_nest.classifier import RuleBasedTaskClassifier
from dragon_nest.models import Device, HealthState, HealthStatus, ModelCapability, ModelSegment, SteeringSpec
from dragon_nest.planner import ExecutionPlanner
from dragon_nest.router import DeterministicRouter


def _devices():
    phone = Device(
        device_id="phone-01",
        display_name="Phone",
        device_type="phone",
        platform="android",
        total_memory_mb=8192,
        health=HealthState(thermal_level=0.30, available_memory_mb=4096, network_rtt_ms=20),
        models=(
            ModelCapability(
                model_id="small-chat-v1",
                model_family="mock",
                role="small_chat",
                task_classes=("chat_qa", "summarization", "reasoning_analysis"),
                max_context_tokens=4096,
                warm=True,
                quality_score=0.65,
            ),
        ),
    )
    pc = Device(
        device_id="pc-01",
        display_name="PC",
        device_type="pc",
        platform="windows",
        total_memory_mb=32768,
        health=HealthState(thermal_level=0.10, available_memory_mb=16000, network_rtt_ms=5),
        models=(
            ModelCapability(
                model_id="large-reasoning-v1",
                model_family="mock",
                role="large_reasoning",
                task_classes=("chat_qa", "summarization", "reasoning_analysis"),
                max_context_tokens=8192,
                warm=True,
                quality_score=0.92,
            ),
        ),
    )
    return [phone, pc]


def test_router_prefers_pc_for_reasoning():
    request = "Compare both options and recommend one."
    profile = RuleBasedTaskClassifier().classify(request)
    plan = ExecutionPlanner().plan(request, profile)
    routed, decision = DeterministicRouter().route(plan, profile, _devices())
    assert decision.selected_device_id == "pc-01"
    assert routed.tasks[0].selected_model_id == "large-reasoning-v1"


def test_data_parallel_routes_all_shards():
    request = "Summarize sections, then give key points."
    profile = RuleBasedTaskClassifier().classify(request, preferred_mode="parallel")
    plan = ExecutionPlanner().plan(request, profile, requested_execution_mode="data_parallel")
    routed, decision = DeterministicRouter().route(plan, profile, _devices())
    assert decision.execution_mode == "data_parallel"
    assert len(routed.tasks) == 3
    assert {task.selected_device_id for task in routed.tasks} == {"pc-01", "phone-01"}


def test_unhealthy_device_is_excluded():
    devices = _devices()
    bad_pc = Device(
        **{
            **devices[1].__dict__,
            "health": HealthState(
                thermal_level=0.95,
                available_memory_mb=16000,
                network_rtt_ms=5,
                status=HealthStatus.UNHEALTHY,
            ),
        }
    )
    request = "Compare both options and recommend one."
    profile = RuleBasedTaskClassifier().classify(request)
    plan = ExecutionPlanner().plan(request, profile)
    _, decision = DeterministicRouter().route(plan, profile, [devices[0], bad_pc])
    assert decision.selected_device_id == "phone-01"


def test_layer_pipeline_uses_contiguous_segments():
    devices = [
        Device(
            "phone-01",
            "Phone",
            "phone",
            "android",
            8192,
            HealthState(available_memory_mb=4096),
            (
                ModelCapability(
                    "part-a",
                    "qwen3",
                    "pipeline_segment",
                    ("reasoning_analysis",),
                    2048,
                    True,
                    0.7,
                    segment=ModelSegment("pipe", 0, 14, 28, True, False),
                ),
            ),
        ),
        Device(
            "pc-01",
            "PC",
            "pc",
            "windows",
            32768,
            HealthState(available_memory_mb=16000),
            (
                ModelCapability(
                    "part-b",
                    "qwen3",
                    "pipeline_segment",
                    ("reasoning_analysis",),
                    2048,
                    True,
                    0.7,
                    segment=ModelSegment("pipe", 14, 28, 28, False, True),
                ),
            ),
        ),
    ]
    request = "Analyze this complex trade-off."
    profile = RuleBasedTaskClassifier().classify(request, preferred_mode="quality")
    plan = ExecutionPlanner().plan(request, profile, requested_execution_mode="layer_pipeline")
    routed, decision = DeterministicRouter().route(plan, profile, devices)
    assert decision.execution_mode == "layer_pipeline"
    assert [stage.selected_device_id for stage in routed.stages] == ["phone-01", "pc-01"]


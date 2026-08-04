from dataclasses import replace

from dragon_nest.models import Device, HealthState, ModelCapability
from dragon_nest.transport.conversion import (
    device_from_registration,
    registration_from_device,
)


def test_registration_round_trips_runtime_and_artifact_capabilities():
    capability = ModelCapability(
        model_id="qnn-model",
        model_family="qwen3",
        role="reasoning",
        task_classes=("reasoning_analysis",),
        max_context_tokens=4096,
        warm=True,
        quality_score=0.9,
        model_version="v2",
        tokenizer_id="qwen-tokenizer",
        precision="fp16",
        runtime_name="qnn",
        runtime_version="QAIRT-2.48",
        supported_accelerators=("htp",),
        min_memory_mb=2048,
        supports_steering=True,
        supports_data_parallel=True,
        supports_layer_pipeline=False,
    )
    device = Device(
        device_id="phone-01",
        display_name="Phone",
        device_type="phone",
        platform="android",
        total_memory_mb=8192,
        health=HealthState(available_memory_mb=4096),
        models=(capability,),
    )

    registration = registration_from_device(device, "token", "agent-v1")
    restored = device_from_registration(registration)

    assert replace(
        restored.models[0], quality_score=capability.quality_score
    ) == capability

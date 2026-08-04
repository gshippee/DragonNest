from pathlib import Path

from dragon_nest.models import ModelCapability
from dragon_nest.steering import SteeringRegistry


ROOT = Path(__file__).resolve().parents[1]


def test_registry_loads_default_spec():
    registry = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    spec = registry.default_spec("concise-vs-verbose-layer-7")
    assert spec.enabled
    assert spec.alpha == -2.0
    assert spec.target_layer == 7


def test_steering_validation_accepts_compatible_model():
    registry = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    spec = registry.default_spec("concise-vs-verbose-layer-7")
    model = ModelCapability(
        model_id="small-chat-v1",
        model_family="mock",
        role="small_chat",
        task_classes=("chat_qa",),
        max_context_tokens=4096,
        warm=True,
        quality_score=0.6,
        steering_vector_ids=("concise-vs-verbose-layer-7",),
        supported_steering_layers=(7,),
    )
    ok, reason = registry.validate(spec, model)
    assert ok, reason


def test_steering_validation_rejects_bad_alpha():
    registry = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    spec = registry.default_spec("concise-vs-verbose-layer-7")
    spec = spec.__class__(**{**spec.__dict__, "alpha": 9.0})
    model = ModelCapability(
        model_id="small-chat-v1",
        model_family="mock",
        role="small_chat",
        task_classes=("chat_qa",),
        max_context_tokens=4096,
        warm=True,
        quality_score=0.6,
        steering_vector_ids=("concise-vs-verbose-layer-7",),
        supported_steering_layers=(7,),
    )
    ok, reason = registry.validate(spec, model)
    assert not ok
    assert "outside range" in reason


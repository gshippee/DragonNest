from pathlib import Path

from dragon_nest.models import ModelCapability, SteeringSpec
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
        # supports_steering is what marks a deployment as able to bind a vector
        # at request time; listing vector ids alone is not enough.
        supports_steering=True,
        steering_vector_ids=("concise-vs-verbose-layer-7",),
        supported_steering_layers=(7,),
    )
    ok, reason = registry.validate(spec, model)
    assert ok, reason


def test_vector_records_carry_lifecycle_metadata():
    registry = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    vector = next(
        v for v in registry.vectors() if v.vector_id == "concise-vs-verbose-layer-7"
    )
    assert vector.status == "validated"
    assert vector.extraction_method == "mean_difference"
    assert "mock" in vector.validated_runtimes
    assert vector.dtype == "fp32"
    assert vector.source_layer == 7


def test_runtime_compatible_enforces_validation_boundaries():
    registry = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    revision = next(
        v for v in registry.vectors()
        if v.vector_id == "concise-vs-verbose-layer-7"
    ).model_revision

    ok, _ = registry.runtime_compatible(
        "concise-vs-verbose-layer-7",
        model_family="qwen3",
        model_revision=revision,
        runtime="genie",
        quantization="w4a16",
        injection_layer=7,
    )
    assert ok

    ok, reason = registry.runtime_compatible(
        "concise-vs-verbose-layer-7",
        model_family="qwen3",
        model_revision=revision,
        runtime="qnn",
        quantization="w4a16",
        injection_layer=7,
    )
    assert not ok
    assert "runtime" in reason

    ok, reason = registry.runtime_compatible(
        "concise-vs-verbose-layer-7",
        model_family="qwen3",
        model_revision=revision,
        runtime="genie",
        quantization="int4-unknown",
        injection_layer=7,
    )
    assert not ok
    assert "quantization" in reason

    ok, reason = registry.runtime_compatible(
        "concise-vs-verbose-layer-7",
        model_family="llama",
        model_revision=revision,
        runtime="genie",
        quantization="w4a16",
        injection_layer=7,
    )
    assert not ok
    assert "family" in reason

    ok, reason = registry.runtime_compatible(
        "concise-vs-verbose-layer-7",
        model_family="qwen3",
        model_revision=revision,
        runtime="genie",
        quantization="w4a16",
        injection_layer=3,
    )
    assert not ok
    assert "layer" in reason


def test_runtime_compatible_rejects_non_routable_status():
    registry = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    ok, reason = registry.runtime_compatible(
        "friendly-warmth-layer-7",
        model_family="mock",
        model_revision="",
        runtime="genie",  # calibrated for mock runtime only
        quantization="none",
        injection_layer=7,
    )
    assert not ok

    ok, _ = registry.runtime_compatible(
        "friendly-warmth-layer-7",
        model_family="mock",
        model_revision="",
        runtime="mock",
        quantization="none",
        injection_layer=7,
    )
    assert ok


def test_steering_validation_rejects_bad_alpha():
    registry = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    spec = registry.default_spec("concise-vs-verbose-layer-7")
    # The runtime-vector realization widened this vector's bound to +/-10;
    # 12.0 is still outside it, so the range check must still bite.
    spec = spec.__class__(**{**spec.__dict__, "alpha": 12.0})
    model = ModelCapability(
        model_id="small-chat-v1",
        model_family="mock",
        role="small_chat",
        task_classes=("chat_qa",),
        max_context_tokens=4096,
        warm=True,
        quality_score=0.6,
        # supports_steering is what marks a deployment as able to bind a vector
        # at request time; listing vector ids alone is not enough.
        supports_steering=True,
        steering_vector_ids=("concise-vs-verbose-layer-7",),
        supported_steering_layers=(7,),
    )
    ok, reason = registry.validate(spec, model)
    assert not ok
    assert "outside range" in reason


def test_baked_profile_routes_only_to_matching_artifact():
    registry = SteeringRegistry.from_yaml(ROOT / "configs/steering-vectors.yaml")
    spec = SteeringSpec(
        enabled=True,
        mode="baked_profile",
        behavior_profile_id="concise",
    )
    baked = ModelCapability(
        model_id="qwen3-0.6b-s25-concise",
        model_family="qwen3",
        role="small_chat",
        task_classes=("chat_qa",),
        max_context_tokens=512,
        warm=False,
        quality_score=0.7,
        steering_modes=("baked_profile",),
        behavior_profile_ids=("concise",),
    )
    base = baked.__class__(
        **{
            **baked.__dict__,
            "model_id": "qwen3-0.6b-s25-base",
            "steering_modes": ("none",),
            "behavior_profile_ids": (),
        }
    )

    assert registry.validate(spec, baked)[0]
    ok, reason = registry.validate(spec, base)
    assert not ok
    assert "does not advertise baked_profile" in reason


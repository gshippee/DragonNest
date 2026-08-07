from __future__ import annotations

import time
from pathlib import Path

import pytest

from dragon_nest.artifacts import ArtifactRegistry
from dragon_nest.config import load_device
from dragon_nest.deployments import (
    ArtifactCatalog,
    ArtifactState,
    DeploymentIndex,
    device_compatibility_classes,
)
from dragon_nest.models import (
    Device,
    HardwareInventory,
    HealthState,
    ModelCapability,
)
from dragon_nest.registry import DeviceRegistry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def catalog() -> ArtifactCatalog:
    return ArtifactCatalog.from_yaml(ROOT / "configs/artifact-catalog.yaml")


def _device(device_id: str, models: tuple[ModelCapability, ...], soc: str = "") -> Device:
    return Device(
        device_id=device_id,
        display_name=device_id,
        device_type="pc",
        platform="windows",
        total_memory_mb=16384,
        health=HealthState(available_memory_mb=8192),
        models=models,
        hardware=HardwareInventory(soc_model=soc),
    )


def _capability(model_id: str, warm: bool) -> ModelCapability:
    return ModelCapability(
        model_id=model_id,
        model_family="mock",
        role="small_chat",
        task_classes=("chat_qa",),
        max_context_tokens=4096,
        warm=warm,
        quality_score=0.6,
    )


def test_catalog_loads_full_models_and_pipeline_stages(catalog):
    small = catalog.get("small-chat-v1")
    assert small.topology == "full_model"
    assert small.kv_cache_bytes_per_token > 0
    memory, is_estimate = small.memory_mb()
    assert memory > 0
    assert is_estimate  # no measured memory recorded yet

    stage = catalog.get("qwen3-0.6b-part-a")
    assert stage.topology == "pipeline_stage"
    assert (stage.start_layer, stage.end_layer) == (0, 14)
    assert stage.boundary_schema


def test_catalog_finds_baked_artifacts_by_profile(catalog):
    baked = catalog.baked_for("concise", "mock")
    assert [artifact.artifact_id for artifact in baked] == [
        "small-chat-v1-concise-baked"
    ]
    assert baked[0].behavior_profile_id == "concise"
    # family-assistant only has a provisioning *target*, not a built artifact
    family = catalog.baked_for("family-assistant", "mock")
    assert [artifact.readiness for artifact in family] == ["unvalidated"]


def test_real_s25_capability_ids_resolve_to_catalog(catalog):
    manifest = ArtifactRegistry.from_yaml(ROOT / "configs/model-artifacts.yaml")
    phone = load_device(ROOT / "configs/hardware-fabric.yaml", "phone-01")
    capability_ids = {model.model_id for model in phone.models}

    assert {
        "qwen3-0.6b-s25-base",
        "qwen3-0.6b-s25-concise",
    }.issubset(capability_ids)
    for model_id in capability_ids & {
        "qwen3-0.6b-s25-base",
        "qwen3-0.6b-s25-concise",
    }:
        catalog_artifact = catalog.get(model_id)
        runtime_artifact = manifest.get(model_id)
        assert catalog_artifact.model_version == runtime_artifact.model_version
        assert catalog_artifact.runtime == runtime_artifact.runtime.value


def test_real_s25_concise_artifact_is_a_baked_profile(catalog):
    concise = catalog.get("qwen3-0.6b-s25-concise")

    assert concise.steering_realization == "baked_profile"
    assert concise.behavior_profile_id == "concise"
    assert concise.vector_id == "concise-vs-verbose-layer-7"


def test_s25_detailed_artifact_is_physically_validated(catalog):
    detailed = catalog.get("qwen3-0.6b-s25-detailed")

    assert detailed.steering_realization == "baked_profile"
    assert detailed.behavior_profile_id == "detailed"
    assert detailed.vector_id == "concise-vs-verbose-layer-7"
    assert detailed.readiness == "ready"


def test_compatibility_class_mapping():
    x_elite = _device("pc", (), soc="Snapdragon X Elite X1E-80-100")
    s25 = _device("phone", (), soc="Qualcomm SM8750 Snapdragon 8 Elite")
    unknown = _device("dev", (), soc="")

    assert "snapdragon-x-elite" in device_compatibility_classes(x_elite)
    assert "snapdragon-8-elite" in device_compatibility_classes(s25)
    # every device can run portable mock artifacts
    for device in (x_elite, s25, unknown):
        assert "mock" in device_compatibility_classes(device)


def test_deployment_index_derives_states_from_advertisement(catalog):
    registry = DeviceRegistry(clock=time.monotonic)
    registry.register(
        _device(
            "pc-01",
            (
                _capability("small-chat-v1", warm=True),
                _capability("large-reasoning-v1", warm=False),
            ),
        )
    )
    index = DeploymentIndex.build(registry.records(), catalog, overrides={})

    assert index.state("pc-01", "small-chat-v1").state == ArtifactState.WARM
    assert index.state("pc-01", "large-reasoning-v1").state == ArtifactState.INSTALLED
    # catalog artifact the device never advertised
    assert (
        index.state("pc-01", "small-chat-v1-concise-baked").state
        == ArtifactState.ABSENT
    )


def test_deployment_overrides_win(catalog):
    registry = DeviceRegistry(clock=time.monotonic)
    registry.register(_device("pc-01", (_capability("small-chat-v1", warm=True),)))
    overrides = {
        ("pc-01", "small-chat-v1"): ArtifactState.ABSENT,
        ("pc-01", "large-reasoning-v1"): ArtifactState.WARM,
    }
    index = DeploymentIndex.build(registry.records(), catalog, overrides=overrides)

    assert index.state("pc-01", "small-chat-v1").state == ArtifactState.ABSENT
    assert index.state("pc-01", "large-reasoning-v1").state == ArtifactState.WARM

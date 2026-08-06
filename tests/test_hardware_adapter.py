from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from dragon_nest.artifacts import ArtifactRegistry, calculate_checksum
from dragon_nest.config import load_device
from dragon_nest.deployments import ArtifactCatalog
from dragon_nest.models import (
    ExecutionMode,
    ExecutionPlan,
    HealthState,
    PlannedTask,
    TaskResult,
)
from dragon_nest.runtime.hardware_adapter import (
    HardwareRuntimeAdapter,
    RuntimeSteeringUnavailableError,
)
from dragon_nest.telemetry import TelemetrySnapshot
from dragon_nest.transport.agent import DeviceAgent


ROOT = Path(__file__).resolve().parents[1]


class _Telemetry:
    def sample(self):
        return TelemetrySnapshot(HealthState(available_memory_mb=4096))


class _Dispatcher:
    qnn = None

    async def execute(self, plan):
        return TaskResult(
            task_id=plan.task_id,
            success=True,
            output_text="real-adapter-delegation",
        )


def _registry(tmp_path: Path, *, mode: str = "none") -> ArtifactRegistry:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.bin").write_bytes(b"model")
    digest = hashlib.sha256()
    digest.update(b"model.bin\0model\0")
    manifest = tmp_path / "manifest.yaml"
    profile = (
        "behavior_profile_id: concise\n    vector_id: vector-7"
        if mode == "baked_profile"
        else 'behavior_profile_id: ""\n    vector_id: ""'
    )
    manifest.write_text(
        f"""
models:
  - model_id: model
    artifact_id: artifact-v1
    model_version: revision
    base_model: Qwen/test
    base_model_revision: revision
    runtime: genie
    artifact_path: {bundle}
    artifact_format: genie_bundle
    checksum: sha256-tree:{digest.hexdigest()}
    tokenizer_id: tokenizer
    tokenizer_fingerprint: hash
    precision: w4a16
    quantization: w4a16
    steering_mode: {mode}
    {profile}
    target_compatibility_class: test-device
    supported_accelerators: [htp]
    min_memory_mb: 1
    max_context_tokens: 16
    supports_steering: false
    supports_data_parallel: true
    supports_layer_pipeline: false
    size_bytes: 5
    verification_status: verified locally without hardware
    runtime_options:
      persistent_load_supported: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return ArtifactRegistry.from_yaml(manifest)


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        task_id="task",
        execution_mode=ExecutionMode.SINGLE,
        request_text="hello",
        tasks=(PlannedTask("shard", "hello", "device", "model"),),
    )


def _adapter(tmp_path: Path, *, mode: str = "none") -> HardwareRuntimeAdapter:
    return HardwareRuntimeAdapter(
        _registry(tmp_path, mode=mode),
        compatibility_key="test-device",
        runtime_name="genie",
        runtime_version="test",
        accelerator_available=True,
        telemetry=_Telemetry(),
        artifact_store=tmp_path / "store",
        dispatcher=_Dispatcher(),
    )


def test_adapter_reports_installed_but_not_warm_for_cli_runtime(tmp_path: Path):
    adapter = _adapter(tmp_path)

    loaded = adapter.load_artifact("artifact-v1")
    capabilities = adapter.capabilities()

    assert loaded.warm is False
    assert capabilities.installed_artifact_ids == ("artifact-v1",)
    assert capabilities.warm_artifact_ids == ()
    assert capabilities.supported_steering_modes == ("none",)


def test_real_xelite_advertised_model_id_resolves_to_brain_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = tmp_path / "xelite-genie"
    bundle.mkdir()
    (bundle / "genie-t2t-run.exe").write_bytes(b"runner")
    (bundle / "genie_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GENIE_DIR", str(bundle))
    monkeypatch.setenv("QWEN3_4B_GENIE_SHA256_TREE", calculate_checksum(bundle))

    artifacts = ArtifactRegistry.from_yaml(ROOT / "configs/model-artifacts.yaml")
    adapter = HardwareRuntimeAdapter(
        artifacts,
        compatibility_key="windows-arm64-x1e-v73-qairt-2.48",
        runtime_name="genie",
        runtime_version="QAIRT-2.48",
        accelerator_available=True,
        telemetry=_Telemetry(),
        artifact_store=tmp_path / "store",
        dispatcher=_Dispatcher(),
    )
    device = load_device(ROOT / "configs/hardware-fabric.yaml", "pc-01")
    advertised = DeviceAgent(
        device,
        artifacts=artifacts,
        executor=adapter,
        telemetry=_Telemetry(),
    ).device.models
    capability = next(
        model for model in advertised if model.model_id == "qwen3-4b-genie"
    )
    catalog = ArtifactCatalog.from_yaml(ROOT / "configs/artifact-catalog.yaml")
    catalog_artifact = catalog.get(capability.model_id)

    assert capability.artifact_id == "qwen3-4b-w4a16-xelite-v73-qairt248"
    assert catalog_artifact.artifact_id == capability.model_id
    assert catalog_artifact.runtime == "genie"


def test_adapter_delegates_existing_execution_plan(tmp_path: Path):
    adapter = _adapter(tmp_path)

    result = asyncio.run(adapter.execute(_plan(), attempt_id="attempt"))

    assert result.success
    assert result.output_text == "real-adapter-delegation"


def test_adapter_rejects_fake_runtime_steering_for_baked_profile(tmp_path: Path):
    adapter = _adapter(tmp_path, mode="baked_profile")

    with pytest.raises(RuntimeSteeringUnavailableError, match="baked_profile"):
        asyncio.run(adapter.execute_runtime_steered(_plan(), object()))


def test_content_addressed_install_verifies_checksum(tmp_path: Path):
    adapter = _adapter(tmp_path)
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    checksum = f"sha256:{hashlib.sha256(b'payload').hexdigest()}"

    installed = adapter.install_artifact(source, checksum)

    assert installed.read_bytes() == b"payload"
    assert installed.parent == (tmp_path / "store").resolve()

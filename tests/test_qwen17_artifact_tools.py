from pathlib import Path

from scripts.artifact_tools.stage_android_artifacts import android_record, locate_context
from scripts.artifact_tools.verify_demo_cache import find_candidate


def test_android_provisioning_manifest_keeps_s0_layerless():
    record = android_record(0, "qwen3-1.7b-s0-s25.bin", "a" * 64)

    boundary = record["split_boundary"]
    assert boundary["stage_index"] == 0
    assert boundary["stage_count"] == 4
    assert boundary["includes_embedding"] is True
    assert "transformer_start_layer" not in boundary
    assert record["checksum"] == f"sha256:{'a' * 64}"


def test_cache_lookup_is_target_and_stage_specific(tmp_path: Path):
    s25 = tmp_path / "qwen3-1.7b/s25/stage-0/model.bin"
    xelite = tmp_path / "qwen3-1.7b/xelite/stage-0/model.bin"
    s25.parent.mkdir(parents=True)
    xelite.parent.mkdir(parents=True)
    s25.write_bytes(b"phone")
    xelite.write_bytes(b"laptop")

    assert locate_context(tmp_path, 0) == s25
    assert find_candidate(tmp_path, "qwen3-1.7b-s0-s25") == s25
    assert find_candidate(tmp_path, "qwen3-1.7b-s0-xelite") == xelite


def test_s25_deployment_wrapper_guards_phone_and_runtime_readiness():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "deploy_s25_demo_artifacts.ps1"
    ).read_text(encoding="utf-8")

    assert "Expected exactly one authorized Android device" in source
    assert "DEBUGGABLE" in source
    assert "run-as" in source
    assert "ARTIFACTS INSTALLED" in source
    assert "RUNTIME NOT YET EXECUTABLE" in source
    assert "qwen3-0.6b-s25-base" in source
    assert "qwen3-0.6b-s25-concise" in source

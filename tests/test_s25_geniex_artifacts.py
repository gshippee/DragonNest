from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _stager():
    path = ROOT / "scripts/artifact_tools/stage_s25_geniex_artifacts.py"
    spec = importlib.util.spec_from_file_location("stage_s25_geniex_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s25_geniex_inventory_maps_profiles_to_exact_artifacts():
    inventory = json.loads(
        (ROOT / "docs/results/s25_geniex_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    records = {item["model_id"]: item for item in inventory["artifacts"]}
    assert set(records) == {
        "qwen3-0.6b-s25-base",
        "qwen3-0.6b-s25-concise",
        "qwen3-0.6b-s25-detailed",
    }
    assert records["qwen3-0.6b-s25-base"]["behavior_profile_id"] == ""
    assert records["qwen3-0.6b-s25-concise"]["behavior_profile_id"] == "concise"
    assert records["qwen3-0.6b-s25-detailed"]["behavior_profile_id"] == "detailed"
    for record in records.values():
        assert record["geniex_autoregressive_ready"] is True
        assert set(record["graph_names"]) == {
            "prompt_ar128_cl512_1_of_2",
            "token_ar1_cl512_1_of_2",
            "prompt_ar128_cl512_2_of_2",
            "token_ar1_cl512_2_of_2",
        }
        assert len(record["sha256_tree"]) == 64
        assert set(record["files"]) >= {
            "genie_config.json",
            "htp_backend_ext_config.json",
            "metadata.json",
            "part1_of_2.bin",
            "part2_of_2.bin",
            "tokenizer.json",
        }
        assert all(len(value) == 64 for value in record["files"].values())


def test_s25_manifest_is_fail_closed_and_never_claims_runtime_steering():
    stager = _stager()
    inventory = json.loads(
        (ROOT / "docs/results/s25_geniex_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    entries = {
        item["model_id"]: stager.manifest_entry(item, item["sha256_tree"])
        for item in inventory["artifacts"]
    }
    assert stager.PACKAGE == "com.dragonnest.agent"
    assert stager.MODEL_ROOT.startswith(
        "/sdcard/Android/data/com.dragonnest.agent/files/"
    )
    for entry in entries.values():
        assert entry["runtime"] == "genie"
        assert entry["runtime_version"] == "GenieX-0.3.5 / QAIRT-2.45"
        assert entry["supported_accelerators"] == ["htp"]
        assert entry["supports_steering"] is False
        assert entry["steering_vector_ids"] == []
        assert entry["checksum"].startswith("sha256-tree:")
    assert entries["qwen3-0.6b-s25-base"]["steering_mode"] == "none"
    assert entries["qwen3-0.6b-s25-concise"]["steering_mode"] == "baked_profile"
    assert entries["qwen3-0.6b-s25-detailed"]["steering_mode"] == "baked_profile"


def test_s25_stager_rejects_prompt_only_inventory():
    stager = _stager()
    record = {
        "model_id": "qwen3-0.6b-s25-base",
        "artifact_id": "qwen3-0.6b-base",
        "source_directory": "base",
        "geniex_autoregressive_ready": False,
        "graph_names": [
            "prompt_ar128_cl512_1_of_2",
            "prompt_ar128_cl512_2_of_2",
        ],
    }
    try:
        stager.assert_safe(record)
    except RuntimeError as failure:
        assert "not certified" in str(failure)
    else:
        raise AssertionError("prompt-only inventory was accepted")


def test_s25_stager_model_id_defaults_to_all_three():
    stager = _stager()
    inventory = json.loads(
        (ROOT / "docs/results/s25_geniex_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    records = inventory["artifacts"]
    selected = stager.select_records(records, None)
    assert selected == records
    assert {record["model_id"] for record in selected} == {
        "qwen3-0.6b-s25-base",
        "qwen3-0.6b-s25-concise",
        "qwen3-0.6b-s25-detailed",
    }


def test_s25_stager_model_id_default_rejects_incomplete_catalog():
    stager = _stager()
    inventory = json.loads(
        (ROOT / "docs/results/s25_geniex_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    incomplete = [
        record
        for record in inventory["artifacts"]
        if record["model_id"] == "qwen3-0.6b-s25-base"
    ]
    try:
        stager.select_records(incomplete, None)
    except RuntimeError as failure:
        assert "exactly Base, Concise, and Detailed" in str(failure)
    else:
        raise AssertionError("incomplete default catalog was accepted")


def test_s25_stager_model_id_subset_selects_only_requested_profile():
    stager = _stager()
    inventory = json.loads(
        (ROOT / "docs/results/s25_geniex_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    selected = stager.select_records(
        inventory["artifacts"], ["qwen3-0.6b-s25-base"]
    )
    assert [record["model_id"] for record in selected] == ["qwen3-0.6b-s25-base"]


def test_s25_stager_model_id_subset_rejects_unknown_model():
    stager = _stager()
    inventory = json.loads(
        (ROOT / "docs/results/s25_geniex_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    try:
        stager.select_records(inventory["artifacts"], ["qwen3-0.6b-s25-nonexistent"])
    except RuntimeError as failure:
        assert "qwen3-0.6b-s25-nonexistent" in str(failure)
    else:
        raise AssertionError("unknown model_id was silently accepted")


def test_s25_stager_parses_repeated_model_id_flag():
    stager = _stager()
    with patch.object(
        sys,
        "argv",
        [
            "stage_s25_geniex_artifacts.py",
            "--cache-root",
            "cache",
            "--inventory",
            "inventory.json",
            "--verify-only",
            "--model-id",
            "qwen3-0.6b-s25-base",
        ],
    ):
        args = stager.parse_args()
    assert args.model_id == ["qwen3-0.6b-s25-base"]


def test_s25_stager_allows_desktop_only_cache_verification():
    stager = _stager()
    with patch.object(
        sys,
        "argv",
        [
            "stage_s25_geniex_artifacts.py",
            "--cache-root",
            "cache",
            "--inventory",
            "inventory.json",
            "--verify-only",
        ],
    ):
        args = stager.parse_args()
    assert args.verify_only is True
    assert args.serial is None
    inventory = json.loads(
        (ROOT / "docs/results/s25_geniex_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    for record in inventory["artifacts"]:
        stager.assert_safe(record)
        entry = stager.manifest_entry(record, record["sha256_tree"])
        assert entry["target_compatibility_class"].startswith(
            "android-arm64-v8a-sm8750-v79-"
        )

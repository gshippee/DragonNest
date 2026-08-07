from __future__ import annotations

import json

import pytest

from scripts.validate_remote_xelite import (
    EXPECTED_ARTIFACT_ID,
    ValidationFailure,
    save_secret_free_proof,
    summarize_runs,
    validate_execution_result,
    validate_worker_snapshot,
)


def _worker(*, runtime: str = "genie", model_id: str = "qwen3-4b-genie"):
    return {
        "device_id": "pc-01",
        "connected": True,
        "transport": "grpc_stream",
        "status": "HEALTHY",
        "health": {"reachable": True, "available_memory_mb": 12000},
        "active_tasks": [],
        "models": [
            {
                "model_id": model_id,
                "runtime": runtime,
                "runtime_version": "QAIRT-2.48",
                "accelerators": ["htp"],
                "artifact_id": EXPECTED_ARTIFACT_ID,
                "warm": False,
                "supports_steering": False,
                "steering_modes": ["none"],
                "steering_vectors": [],
            }
        ],
        "deployments": [{"artifact_id": model_id, "state": "installed"}],
    }


def _successful_result(*, device_id: str = "pc-01", model_id: str = "qwen3-4b-genie"):
    submission = {
        "task_id": "task-1",
        "accepted_attempt_id": "attempt-1",
        "device_id": device_id,
        "model_id": model_id,
        "success": True,
        "route_plan": {
            "chosen": {
                "device_id": device_id,
                "artifact_id": model_id,
                "runtime": "genie",
                "deployment_state": "installed",
            }
        },
    }
    task = {
        "accepted_attempt_id": "attempt-1",
        "result": {
            "success": True,
            "output_text": "DRAGONNEST_CROSSHOST_OK",
            "metrics": {
                "model_id": model_id,
                "runtime_name": "genie",
                "runtime_version": "QAIRT-2.48",
                "accelerator": "htp",
                "execution_latency_ms": 100,
            },
        },
    }
    return submission, task


def test_refuses_absent_worker():
    with pytest.raises(ValidationFailure, match="worker 'pc-01' is absent"):
        validate_worker_snapshot(
            [], device_id="pc-01", model_id="qwen3-4b-genie"
        )


def test_refuses_mock_runtime():
    with pytest.raises(ValidationFailure, match="requires genie"):
        validate_worker_snapshot(
            [_worker(runtime="mock")],
            device_id="pc-01",
            model_id="qwen3-4b-genie",
        )


def test_refuses_wrong_advertised_model():
    with pytest.raises(ValidationFailure, match="does not advertise"):
        validate_worker_snapshot(
            [_worker(model_id="not-qwen")],
            device_id="pc-01",
            model_id="qwen3-4b-genie",
        )


def test_refuses_wrong_result_device():
    submission, task = _successful_result(device_id="other-device")
    with pytest.raises(ValidationFailure, match="expected 'pc-01'"):
        validate_execution_result(
            submission, task, device_id="pc-01", model_id="qwen3-4b-genie"
        )


def test_refuses_wrong_result_model():
    submission, task = _successful_result(model_id="wrong-model")
    with pytest.raises(ValidationFailure, match="expected 'qwen3-4b-genie'"):
        validate_execution_result(
            submission, task, device_id="pc-01", model_id="qwen3-4b-genie"
        )


def test_refuses_non_htp_result():
    submission, task = _successful_result()
    task["result"]["metrics"]["accelerator"] = "cpu"
    with pytest.raises(ValidationFailure, match="expected 'htp'"):
        validate_execution_result(
            submission, task, device_id="pc-01", model_id="qwen3-4b-genie"
        )


def test_saves_secret_free_proof(tmp_path):
    output = tmp_path / "proof.json"
    proof = {
        "task_id": "task-1",
        "output_sha256": "abc",
        "runtime_name": "genie",
    }

    save_secret_free_proof(proof, output)

    assert json.loads(output.read_text(encoding="utf-8")) == proof
    with pytest.raises(ValidationFailure, match="sensitive key"):
        save_secret_free_proof({"enrollment_token": "secret"}, output)


def test_memory_sampling_summary_is_correct():
    summary = summarize_runs(
        [
            {
                "peak_available_memory_delta_mb": 8000,
                "execution_latency_ms": 20000,
                "telemetry_capture_reliable": True,
            },
            {
                "peak_available_memory_delta_mb": 9000,
                "execution_latency_ms": 22000,
                "telemetry_capture_reliable": True,
            },
            {
                "peak_available_memory_delta_mb": 10000,
                "execution_latency_ms": 24000,
                "telemetry_capture_reliable": False,
            },
        ]
    )

    assert summary == {
        "median_peak_available_memory_delta_mb": 9000,
        "max_peak_available_memory_delta_mb": 10000,
        "median_execution_latency_ms": 22000,
        "max_execution_latency_ms": 24000,
        "brain_telemetry_memory_capture_reliable": False,
    }

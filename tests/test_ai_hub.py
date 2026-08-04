from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from dragon_nest.runtime.ai_hub import AiHubDeviceLab, AiHubError


@dataclass
class _Device:
    name: str
    os: str = "test-os"


class _Model:
    model_id = "mq-test"


class _Job:
    def __init__(self, job_id: str, outputs=None, profile=None):
        self.job_id = job_id
        self.url = f"https://example.test/jobs/{job_id}"
        self._outputs = outputs
        self._profile = profile

    def get_target_model(self):
        return _Model()

    def download_output_data(self):
        return self._outputs

    def download_profile(self):
        return self._profile


class _Client:
    def __init__(self):
        self.devices = [_Device("Snapdragon X Elite CRD"), _Device("Phone QRD")]
        self.inference_calls = []

    def get_devices(self):
        return self.devices

    def submit_compile_and_link_jobs(self, **kwargs):
        return [_Job("compile")], _Job("link")

    def submit_inference_job(self, **kwargs):
        self.inference_calls.append(kwargs)
        inputs = kwargs["inputs"]
        if "boundary" in inputs:
            outputs = {"output": [inputs["boundary"][0] + 1]}
        else:
            outputs = {"boundary": [inputs["hidden"][0] * 2]}
        return _Job(f"infer-{len(self.inference_calls)}", outputs=outputs)

    def submit_profile_job(self, **kwargs):
        return _Job(
            "profile",
            profile={
                "execution_summary": {
                    "estimated_inference_time": 2250,
                    "estimated_inference_peak_memory": 10 * 1024 * 1024,
                },
                "execution_detail": [
                    {"compute_unit": "NPU"},
                    {"compute_unit": "NPU"},
                ],
            },
        )


def test_device_lab_compiles_runs_pipeline_and_normalizes_metrics(tmp_path: Path):
    client = _Client()
    lab = AiHubDeviceLab(client, "test-sdk")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"source")

    compilation = lab.compile_and_link(
        model_path,
        device_name="X Elite",
        name="compile-test",
    )
    run = lab.run_pipeline(
        compilation.target_model,
        compilation.target_model,
        first_device_name="Phone QRD",
        second_device_name="Snapdragon X Elite CRD",
        initial_inputs={"hidden": np.array([[2]], dtype=np.float32)},
        boundary_output="boundary",
        boundary_input="boundary",
        name="pipeline-test",
    )
    profile = lab.profile(
        compilation.target_model,
        device_name="Phone",
        name="profile-test",
    )
    metrics = profile.execution_metrics(
        model_id="model",
        model_version="v1",
        runtime_version="qairt-test",
    )

    assert compilation.target_model_id == "mq-test"
    assert compilation.compile_jobs[0].job_id == "compile"
    np.testing.assert_array_equal(run.boundary, [[4]])
    np.testing.assert_array_equal(run.second_stage.outputs["output"], [[5]])
    assert profile.ops_by_compute_unit == {"NPU": 2}
    assert metrics.accelerator == "NPU"
    assert metrics.execution_latency_ms == 2
    assert metrics.observed_memory_delta_mb == 10


def test_device_lab_reports_missing_device_and_boundary(tmp_path: Path):
    lab = AiHubDeviceLab(_Client())

    with pytest.raises(AiHubError, match="no AI Hub device"):
        lab.resolve_device("does-not-exist")

    inference = lab.inference(
        _Model(),
        device_name="Phone",
        inputs={"hidden": np.array([[1]], dtype=np.float32)},
        name="single-output",
    )
    np.testing.assert_array_equal(inference.output("renamed-by-qnn"), [[2]])

    with pytest.raises(AiHubError, match="available"):
        inference.__class__(
            outputs={"first": np.array([1]), "second": np.array([2])},
            job=inference.job,
            device_name=inference.device_name,
            wall_latency_ms=0,
        ).output("missing")

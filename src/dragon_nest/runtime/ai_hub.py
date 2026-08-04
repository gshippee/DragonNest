from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..models import ExecutionMetrics, RuntimeName


class AiHubError(RuntimeError):
    """Raised when an AI Hub device-lab operation cannot produce a result."""


@dataclass(frozen=True)
class AiHubJobRef:
    job_id: str
    url: str


@dataclass(frozen=True)
class AiHubCompilation:
    target_model: Any
    target_model_id: str
    device_name: str
    compile_jobs: tuple[AiHubJobRef, ...]
    link_job: AiHubJobRef | None


@dataclass(frozen=True)
class AiHubInference:
    outputs: Mapping[str, np.ndarray]
    job: AiHubJobRef
    device_name: str
    wall_latency_ms: int

    def output(self, requested_name: str) -> np.ndarray:
        if requested_name in self.outputs:
            return self.outputs[requested_name]
        if len(self.outputs) == 1:
            # Linked QNN context binaries commonly expose the sole ONNX output
            # as ``output_0``. A single output remains unambiguous.
            return next(iter(self.outputs.values()))
        available = ", ".join(sorted(self.outputs))
        raise AiHubError(
            f"inference did not return {requested_name!r}; available: {available}"
        )


@dataclass(frozen=True)
class AiHubProfile:
    job: AiHubJobRef
    device_name: str
    estimated_latency_us: int | None
    peak_memory_bytes: int | None
    ops_by_compute_unit: Mapping[str, int]

    def execution_metrics(
        self,
        *,
        model_id: str,
        model_version: str,
        runtime_version: str,
        fallback_latency_ms: int = 0,
    ) -> ExecutionMetrics:
        latency_ms = fallback_latency_ms
        if self.estimated_latency_us is not None:
            latency_ms = max(1, round(self.estimated_latency_us / 1000))
        memory_mb = None
        if self.peak_memory_bytes is not None:
            memory_mb = round(self.peak_memory_bytes / (1024 * 1024))
        accelerator = "+".join(sorted(self.ops_by_compute_unit)) or "unknown"
        return ExecutionMetrics(
            model_id=model_id,
            model_version=model_version,
            runtime_name=RuntimeName.QNN,
            runtime_version=runtime_version,
            accelerator=accelerator,
            execution_latency_ms=latency_ms,
            observed_memory_delta_mb=memory_mb,
        )


@dataclass(frozen=True)
class AiHubPipelineRun:
    first_stage: AiHubInference
    second_stage: AiHubInference
    boundary: np.ndarray


class AiHubDeviceLab:
    """Opt-in adapter for running QNN validation jobs on Qualcomm AI Hub.

    AI Hub is a remote hardware lab, not an Agent deployment runtime. This
    adapter is therefore kept separate from ``QnnExecutor`` and never submits
    work unless one of its compile, inference, or profile methods is called.
    """

    def __init__(self, client: Any, sdk_version: str = "unknown"):
        self.client = client
        self.sdk_version = sdk_version

    @classmethod
    def from_config(cls) -> "AiHubDeviceLab":
        try:
            import qai_hub
        except ImportError as exc:
            raise AiHubError(
                "qai-hub is not installed; use the isolated "
                "requirements-ai-hub.txt environment"
            ) from exc
        try:
            client = qai_hub.Client()
        except Exception as exc:
            raise AiHubError(f"AI Hub client configuration failed: {exc}") from exc
        return cls(client, str(qai_hub.__version__))

    def devices(self) -> tuple[Any, ...]:
        try:
            return tuple(self.client.get_devices())
        except Exception as exc:
            raise AiHubError(f"AI Hub device lookup failed: {exc}") from exc

    def resolve_device(self, name: str) -> Any:
        devices = self.devices()
        exact = [device for device in devices if device.name.casefold() == name.casefold()]
        matches = exact or [
            device for device in devices if name.casefold() in device.name.casefold()
        ]
        if not matches:
            available = ", ".join(sorted({device.name for device in devices}))
            raise AiHubError(f"no AI Hub device matching {name!r}; available: {available}")
        return matches[-1]

    def compile_and_link(
        self,
        model_path: str | Path,
        *,
        device_name: str,
        name: str,
        input_specs: Mapping[str, tuple[tuple[int, ...], str]] | None = None,
    ) -> AiHubCompilation:
        path = Path(model_path)
        if not path.is_file():
            raise AiHubError(f"source model does not exist: {path}")
        device = self.resolve_device(device_name)
        try:
            submitted = self.client.submit_compile_and_link_jobs(
                models=str(path),
                device=device,
                name=name,
                input_specs=input_specs,
            )
            compile_jobs, link_job = submitted[0], submitted[1]
            target_model = (
                link_job.get_target_model()
                if link_job is not None
                else compile_jobs[0].get_target_model()
            )
        except Exception as exc:
            raise AiHubError(f"AI Hub compile/link failed for {name}: {exc}") from exc
        if target_model is None:
            refs = ", ".join(job.url for job in map(_job_ref, compile_jobs))
            raise AiHubError(f"AI Hub compile/link produced no model; jobs: {refs}")
        return AiHubCompilation(
            target_model=target_model,
            target_model_id=str(target_model.model_id),
            device_name=device.name,
            compile_jobs=tuple(_job_ref(job) for job in compile_jobs),
            link_job=_job_ref(link_job) if link_job is not None else None,
        )

    def inference(
        self,
        model: Any,
        *,
        device_name: str,
        inputs: Mapping[str, np.ndarray],
        name: str,
    ) -> AiHubInference:
        device = self.resolve_device(device_name)
        samples = {key: [np.asarray(value)] for key, value in inputs.items()}
        start = time.perf_counter()
        try:
            job = self.client.submit_inference_job(
                model=model,
                device=device,
                inputs=samples,
                name=name,
            )
            raw_outputs = job.download_output_data()
        except Exception as exc:
            raise AiHubError(f"AI Hub inference failed for {name}: {exc}") from exc
        latency_ms = max(0, round((time.perf_counter() - start) * 1000))
        if not raw_outputs:
            raise AiHubError(f"AI Hub inference returned no outputs: {_job_ref(job).url}")
        outputs: dict[str, np.ndarray] = {}
        for output_name, output_samples in raw_outputs.items():
            if not output_samples:
                raise AiHubError(f"AI Hub output {output_name!r} has no samples")
            outputs[str(output_name)] = np.asarray(output_samples[0])
        return AiHubInference(
            outputs=outputs,
            job=_job_ref(job),
            device_name=device.name,
            wall_latency_ms=latency_ms,
        )

    def profile(
        self,
        model: Any,
        *,
        device_name: str,
        name: str,
    ) -> AiHubProfile:
        device = self.resolve_device(device_name)
        try:
            job = self.client.submit_profile_job(
                model=model,
                device=device,
                name=name,
            )
            report = job.download_profile()
        except Exception as exc:
            raise AiHubError(f"AI Hub profile failed for {name}: {exc}") from exc
        summary = report.get("execution_summary", {})
        compute_units: dict[str, int] = {}
        for operation in report.get("execution_detail", []):
            unit = str(operation.get("compute_unit", "unknown"))
            compute_units[unit] = compute_units.get(unit, 0) + 1
        return AiHubProfile(
            job=_job_ref(job),
            device_name=device.name,
            estimated_latency_us=_optional_int(summary.get("estimated_inference_time")),
            peak_memory_bytes=_optional_int(
                summary.get("estimated_inference_peak_memory")
            ),
            ops_by_compute_unit=compute_units,
        )

    def run_pipeline(
        self,
        first_model: Any,
        second_model: Any,
        *,
        first_device_name: str,
        second_device_name: str,
        initial_inputs: Mapping[str, np.ndarray],
        boundary_output: str,
        boundary_input: str,
        name: str,
    ) -> AiHubPipelineRun:
        first = self.inference(
            first_model,
            device_name=first_device_name,
            inputs=initial_inputs,
            name=f"{name}-stage-0",
        )
        boundary = first.output(boundary_output)
        second = self.inference(
            second_model,
            device_name=second_device_name,
            inputs={boundary_input: boundary},
            name=f"{name}-stage-1",
        )
        return AiHubPipelineRun(first, second, boundary)


def _job_ref(job: Any) -> AiHubJobRef:
    return AiHubJobRef(job_id=str(job.job_id), url=str(job.url))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)

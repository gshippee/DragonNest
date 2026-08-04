from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dragon_nest.runtime.ai_hub import (  # noqa: E402
    AiHubCompilation,
    AiHubDeviceLab,
    AiHubJobRef,
    AiHubProfile,
)


def _make_steering_model(path: Path, seq_len: int, hidden_size: int) -> None:
    import onnx
    from onnx import TensorProto, helper

    inputs = [
        helper.make_tensor_value_info(
            "hidden", TensorProto.FLOAT, [1, seq_len, hidden_size]
        ),
        helper.make_tensor_value_info(
            "steering", TensorProto.FLOAT, [1, 1, hidden_size]
        ),
        helper.make_tensor_value_info("alpha", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, seq_len, 1]),
    ]
    output = helper.make_tensor_value_info(
        "boundary", TensorProto.FLOAT, [1, seq_len, hidden_size]
    )
    nodes = [
        helper.make_node("Mul", ["alpha", "mask"], ["weighted_mask"]),
        helper.make_node(
            "Mul", ["weighted_mask", "steering"], ["steering_delta"]
        ),
        helper.make_node("Add", ["hidden", "steering_delta"], ["boundary"]),
    ]
    graph = helper.make_graph(nodes, "dragonnest-steering-stage", inputs, [output])
    model = helper.make_model(
        graph,
        producer_name="DragonNest",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _make_finish_model(path: Path, seq_len: int, hidden_size: int) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    boundary = helper.make_tensor_value_info(
        "boundary", TensorProto.FLOAT, [1, seq_len, hidden_size]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, seq_len, hidden_size]
    )
    gain = numpy_helper.from_array(np.array([0.5], dtype=np.float32), "gain")
    bias = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "bias")
    nodes = [
        helper.make_node("Mul", ["boundary", "gain"], ["scaled"]),
        helper.make_node("Add", ["scaled", "bias"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dragonnest-finish-stage",
        [boundary],
        [output],
        initializer=[gain, bias],
    )
    model = helper.make_model(
        graph,
        producer_name="DragonNest",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _job(job: AiHubJobRef | None) -> dict[str, str] | None:
    if job is None:
        return None
    return {"id": job.job_id, "url": job.url}


def _compilation(
    compilation: AiHubCompilation | None, model_id: str, model: Any
) -> dict[str, Any]:
    if compilation is None:
        record: dict[str, Any] = {"reused_model_id": model_id}
        producer = model.get_producer()
        if producer is not None:
            record["link"] = _job(
                AiHubJobRef(job_id=str(producer.job_id), url=str(producer.url))
            )
            compile_jobs = []
            for source_model in getattr(producer, "models", []):
                compile_job = source_model.get_producer()
                if compile_job is not None:
                    compile_jobs.append(
                        {
                            "id": str(compile_job.job_id),
                            "url": str(compile_job.url),
                        }
                    )
            if compile_jobs:
                record["compile"] = compile_jobs
        metadata = {
            getattr(key, "name", str(key)): str(value)
            for key, value in (getattr(model, "metadata", None) or {}).items()
        }
        if metadata:
            record["runtime_metadata"] = metadata
        return record
    return {
        "model_id": compilation.target_model_id,
        "compile": [_job(job) for job in compilation.compile_jobs],
        "link": _job(compilation.link_job),
    }


def _profile(profile: AiHubProfile) -> dict[str, Any]:
    return {
        "job": _job(profile.job),
        "estimated_inference_time_us": profile.estimated_latency_us,
        "peak_memory_bytes": profile.peak_memory_bytes,
        "ops_by_compute_unit": dict(profile.ops_by_compute_unit),
        "all_ops_on_npu": bool(
            profile.ops_by_compute_unit
            and set(profile.ops_by_compute_unit) == {"NPU"}
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _error(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    absolute = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
    }


def _get_or_compile(
    lab: AiHubDeviceLab,
    *,
    model_id: str | None,
    model_path: Path,
    device_name: str,
    job_name: str,
) -> tuple[Any, str, AiHubCompilation | None]:
    if model_id:
        try:
            model = lab.client.get_model(model_id)
        except Exception as exc:
            raise SystemExit(f"cannot load AI Hub model {model_id}: {exc}") from exc
        return model, str(model.model_id), None
    compilation = lab.compile_and_link(
        model_path,
        device_name=device_name,
        name=job_name,
    )
    return compilation.target_model, compilation.target_model_id, compilation


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an opt-in steering and two-device boundary proof on real "
            "Snapdragon hardware hosted by Qualcomm AI Hub."
        )
    )
    parser.add_argument("--submit", action="store_true", help="consume AI Hub quota")
    parser.add_argument("--phone", default="Snapdragon 8 Elite QRD")
    parser.add_argument("--pc", default="Snapdragon X Elite CRD")
    parser.add_argument("--stage-0-model-id", help="reuse an existing linked model")
    parser.add_argument("--stage-1-model-id", help="reuse an existing linked model")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/dragonnest-aihub"))
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/dragonnest-aihub-proof.json")
    )
    parser.add_argument("--tolerance", type=float, default=1e-2)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    stage0_path = args.work_dir / "steering_stage.onnx"
    stage1_path = args.work_dir / "finish_stage.onnx"
    seq_len, hidden_size = 16, 64
    _make_steering_model(stage0_path, seq_len, hidden_size)
    _make_finish_model(stage1_path, seq_len, hidden_size)

    lab = AiHubDeviceLab.from_config()
    phone = lab.resolve_device(args.phone)
    pc = lab.resolve_device(args.pc)
    print(f"Stage 0 target: {phone.name} (os={phone.os})")
    print(f"Stage 1 target: {pc.name} (os={pc.os})")
    print(f"Models: {stage0_path}, {stage1_path}")
    if not args.submit:
        print("Dry run only; pass --submit to compile, infer, profile, and validate.")
        return 0

    rng = np.random.default_rng(20260804)
    hidden = rng.standard_normal((1, seq_len, hidden_size)).astype(np.float32)
    steering = rng.standard_normal((1, 1, hidden_size)).astype(np.float32)
    steering /= np.linalg.norm(steering)
    mask = np.zeros((1, seq_len, 1), dtype=np.float32)
    mask[:, seq_len // 2 :, :] = 1
    alpha = 4.0
    base_inputs = {"hidden": hidden, "steering": steering, "mask": mask}

    print("Preparing QNN stage 0 on phone-class Snapdragon...")
    model0, model0_id, compile0 = _get_or_compile(
        lab,
        model_id=args.stage_0_model_id,
        model_path=stage0_path,
        device_name=phone.name,
        job_name="dragonnest-steering-stage",
    )
    print("Preparing QNN stage 1 on PC-class Snapdragon...")
    model1, model1_id, compile1 = _get_or_compile(
        lab,
        model_id=args.stage_1_model_id,
        model_path=stage1_path,
        device_name=pc.name,
        job_name="dragonnest-finish-stage",
    )

    print("Running alpha=0 steering control on stage 0...")
    control = lab.inference(
        model0,
        device_name=phone.name,
        inputs={**base_inputs, "alpha": np.array([0.0], dtype=np.float32)},
        name="dragonnest-steering-control",
    )
    print("Running steered stage 0 -> stage 1 boundary chain...")
    pipeline = lab.run_pipeline(
        model0,
        model1,
        first_device_name=phone.name,
        second_device_name=pc.name,
        initial_inputs={
            **base_inputs,
            "alpha": np.array([alpha], dtype=np.float32),
        },
        boundary_output="boundary",
        boundary_input="boundary",
        name="dragonnest-live-pipeline",
    )
    print("Profiling both QNN stages...")
    profile0 = lab.profile(
        model0, device_name=phone.name, name="dragonnest-stage-0-profile"
    )
    profile1 = lab.profile(
        model1, device_name=pc.name, name="dragonnest-stage-1-profile"
    )

    control_output = control.output("boundary")
    expected_boundary = hidden + alpha * mask * steering
    final_output = pipeline.second_stage.output("output")
    expected_final = expected_boundary * np.float32(0.5) + np.float32(1.0)
    control_check = _error(control_output, hidden)
    boundary_check = _error(pipeline.boundary, expected_boundary)
    final_check = _error(final_output, expected_final)
    passed = all(
        check["max_abs_error"] <= args.tolerance
        for check in (control_check, boundary_check, final_check)
    )

    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qai_hub_version": lab.sdk_version,
        "scope": (
            "Remote device-lab proof; validates QNN steering and sequential "
            "boundary compatibility, not live device-to-device transport."
        ),
        "devices": {
            "stage_0": {"name": phone.name, "os": phone.os},
            "stage_1": {"name": pc.name, "os": pc.os},
        },
        "models": {
            "stage_0_sha256": _sha256(stage0_path),
            "stage_1_sha256": _sha256(stage1_path),
            "stage_0": _compilation(compile0, model0_id, model0),
            "stage_1": _compilation(compile1, model1_id, model1),
        },
        "jobs": {
            "steering_control": _job(control.job),
            "pipeline_stage_0": _job(pipeline.first_stage.job),
            "pipeline_stage_1": _job(pipeline.second_stage.job),
        },
        "checks": {
            "alpha_zero_identity": control_check,
            "steering_boundary": boundary_check,
            "cross_device_chain": final_check,
            "tolerance": args.tolerance,
            "passed": passed,
        },
        "profiles": {"stage_0": _profile(profile0), "stage_1": _profile(profile1)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Proof: {'PASS' if passed else 'FAIL'}")
    print(f"Result: {args.output}")
    print(f"Reusable model IDs: stage0={model0_id} stage1={model1_id}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

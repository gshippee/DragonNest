"""Provision the four S25 QNN stages after installing the debuggable APK.

The 1.67 GB of model bytes remains outside the APK. This helper verifies every
external context against the canonical inventory, uses ``adb push`` only as a
transport, copies it into app-private storage with ``run-as``, verifies the
installed hashes, installs the Android runtime manifest, and stops the app so
the next launch rebuilds the runtime catalog.

Usage:
  python scripts/artifact_tools/stage_android_artifacts.py CACHE_ROOT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs/results/demo_artifact_inventory.json"
PIPELINE_ID = "qwen3-1.7b-w4a16-demo-v1"
PACKAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]+$")
MEMORY_MB = (1024, 768, 768, 1152)
TENSORS = (
    ("input_ids", "embedding"),
    ("embedding", "add_21844"),
    ("add_21844", "add_42314"),
    ("add_42314", "logits"),
)
LAYERS = ((None, None), (0, 9), (10, 19), (20, 27))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*command: str, capture: bool = False) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def locate_context(cache_root: Path, stage_index: int) -> Path:
    stage_dir = cache_root / f"qwen3-1.7b/s25/stage-{stage_index}"
    candidates = sorted(stage_dir.rglob("*.bin")) if stage_dir.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one .bin under {stage_dir}, found {len(candidates)}"
        )
    return candidates[0]


def android_record(stage_index: int, filename: str, checksum: str) -> dict:
    start, end = LAYERS[stage_index]
    input_tensor, output_tensor = TENSORS[stage_index]
    boundary = {
        "pipeline_id": PIPELINE_ID,
        "stage_index": stage_index,
        "stage_count": 4,
        "total_layers": 28,
        "input_tensor": input_tensor,
        "output_tensor": output_tensor,
        "includes_embedding": stage_index == 0,
        "includes_lm_head": stage_index == 3,
        "boundary_format": "qnn-raw-tensor-v1",
    }
    if start is not None:
        boundary["transformer_start_layer"] = start
        boundary["transformer_end_layer"] = end
    return {
        "model_id": f"qwen3-1.7b-s{stage_index}-s25",
        "artifact_id": f"qwen3-1.7b-w4a16-demo-v1-s{stage_index}-s25",
        "model_version": "qwen3-1.7b-demo-v1-unpinned-main",
        "runtime": "qnn",
        "artifact_path": filename,
        "checksum": f"sha256:{checksum}",
        "tokenizer_id": "Qwen/Qwen3-1.7B",
        "precision": "w4a16-name-w8a16-compile-observed",
        "supported_accelerators": ["htp"],
        "min_memory_mb": MEMORY_MB[stage_index],
        "max_context_tokens": 512,
        "supports_steering": False,
        "supports_data_parallel": False,
        "supports_layer_pipeline": True,
        "model_family": "qwen3-1.7b",
        "role": "pipeline_segment",
        "task_classes": ["reasoning_analysis"],
        "quality_score": 0.8,
        "runtime_version": "QAIRT-2.45",
        "target_compatibility_class": "android-arm64-sm8750-v79-qairt-2.45",
        "runtime_options": {
            "prompt_graph": f"prompt_ar128_cl512_{stage_index + 1}_of_4",
            "decode_graph": f"token_ar1_cl512_{stage_index + 1}_of_4",
            "tokenizer_owner": "stage-0",
            "sampling_owner": "stage-3-top1",
        },
        "split_boundary": boundary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", default="")
    parser.add_argument("--package", default="com.dragonnest.agent")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not PACKAGE_RE.fullmatch(args.package):
        raise SystemExit("invalid Android package name")
    inventory = {
        item["logical_artifact_id"]: item
        for item in json.loads(INVENTORY.read_text(encoding="utf-8"))["artifacts"]
    }
    # Verify all 1.67 GB before the first ADB mutation. A late corrupt stage
    # must not leave a partially installed catalog on the phone.
    verified: list[tuple[int, str, Path, str]] = []
    for index in range(4):
        logical_id = f"qwen3-1.7b-s{index}-s25"
        expected = inventory[logical_id]
        if expected.get("status") != "READY" or not expected.get("sha256"):
            raise RuntimeError(f"{logical_id}: inventory is not checksummed READY")
        source = locate_context(args.cache_root, index)
        if source.stat().st_size != expected["size_bytes"]:
            raise RuntimeError(f"{logical_id}: byte-size mismatch")
        checksum = sha256_file(source)
        if checksum != expected["sha256"]:
            raise RuntimeError(f"{logical_id}: SHA-256 mismatch")
        verified.append((index, logical_id, source, checksum))

    adb = [args.adb, *(["-s", args.serial] if args.serial else [])]
    run(*adb, "get-state", capture=True)
    run(*adb, "shell", "run-as", args.package, "mkdir", "-p", "files/dragonnest-models")
    remote_root = f"/data/local/tmp/dragonnest-stage-{uuid.uuid4().hex[:12]}"
    run(*adb, "shell", "mkdir", "-p", remote_root)

    records = []
    try:
        with tempfile.TemporaryDirectory(prefix="dragonnest-android-stage-") as temp:
            temp_dir = Path(temp)
            for index, logical_id, source, checksum in verified:
                filename = f"{logical_id}.bin"
                transit = f"{remote_root}/{filename}"
                destination = f"files/dragonnest-models/{filename}"
                run(*adb, "push", str(source), transit)
                run(*adb, "shell", "chmod", "0644", transit)
                run(
                    *adb,
                    "shell",
                    "run-as",
                    args.package,
                    "cp",
                    transit,
                    destination,
                )
                installed = run(
                    *adb,
                    "shell",
                    "run-as",
                    args.package,
                    "sha256sum",
                    destination,
                    capture=True,
                ).split()[0]
                if installed != checksum:
                    raise RuntimeError(f"{logical_id}: installed SHA-256 mismatch")
                records.append(android_record(index, filename, checksum))

            manifest = temp_dir / "manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": 1, "models": records}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            marker = temp_dir / ".installed"
            marker.write_text("1", encoding="ascii")
            for local, name in (
                (manifest, "manifest.json"),
                (marker, ".installed"),
            ):
                transit = f"{remote_root}/dragonnest-{name.lstrip('.')}"
                run(*adb, "push", str(local), transit)
                run(
                    *adb,
                    "shell",
                    "run-as",
                    args.package,
                    "cp",
                    transit,
                    f"files/dragonnest-models/{name}",
                )
        run(*adb, "shell", "am", "force-stop", args.package)
    finally:
        subprocess.run(
            [*adb, "shell", "rm", "-rf", remote_root],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print("ARTIFACTS INSTALLED")
    for _, logical_id, _, _ in verified:
        print(f"  {logical_id}")
    print("RUNTIME NOT YET EXECUTABLE")
    print("The Agent will not advertise these stages until the direct QNN binding passes physical validation.")
    print("Relaunch PersonaCare to reload the app-private artifact manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

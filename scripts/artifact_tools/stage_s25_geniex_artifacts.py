#!/usr/bin/env python3
"""Provision checksummed Qwen3-0.6B GenieX bundles into PersonaCare's private store."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any


PACKAGE = "com.dragonnest.agent"
MODEL_ROOT = f"/sdcard/Android/data/{PACKAGE}/files/dragonnest-models"
REQUIRED_FILES = {
    "genie_config.json",
    "metadata.json",
    "part1_of_2.bin",
    "part2_of_2.bin",
    "tokenizer.json",
}
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]+$")
EXPECTED_GRAPHS = {
    "prompt_ar128_cl512_1_of_2",
    "token_ar1_cl512_1_of_2",
    "prompt_ar128_cl512_2_of_2",
    "token_ar1_cl512_2_of_2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial")
    parser.add_argument("--package", default=PACKAGE)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the external cache and inventory without connecting to a phone",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help=(
            "restrict verification/provisioning to this model_id; may be "
            "repeated. Default is every record in the inventory (Base, "
            "Concise, and Detailed). A device only ever advertises the "
            "models it was actually given -- provisioning a subset is not "
            "a weaker check, it is a smaller, still-exact one."
        ),
    )
    return parser.parse_args()


def run(command: list[str], *, capture: bool = True) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_tree(root: Path) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    # Release archives may sit beside their extracted contents. They are
    # provenance, not runtime inputs, and must not consume app-private storage.
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item.suffix.lower() != ".zip"
    ):
        relative = path.relative_to(root).as_posix()
        file_hashes[relative] = digest_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), file_hashes


def discover_bundle(root: Path) -> Path:
    candidates = [root] + sorted(item for item in root.rglob("*") if item.is_dir())
    matches: list[Path] = []
    for candidate in candidates:
        names = {item.name for item in candidate.iterdir() if item.is_file()}
        bins = sorted(name for name in names if name.endswith(".bin"))
        if REQUIRED_FILES.issubset(names) and bins == ["part1_of_2.bin", "part2_of_2.bin"]:
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one complete GenieX bundle under {root}, found {matches}")
    return matches[0]


def adb(args: argparse.Namespace, *arguments: str, capture: bool = True) -> str:
    return run([args.adb, "-s", args.serial, *arguments], capture=capture)


def shell(args: argparse.Namespace, command: str) -> str:
    return adb(args, "shell", command)


def run_as(args: argparse.Namespace, command: str) -> str:
    remote = f"run-as {shlex.quote(args.package)} sh -c {shlex.quote(command)}"
    return adb(args, "shell", remote)


def select_records(
    records: list[dict[str, Any]], model_ids: list[str] | None
) -> list[dict[str, Any]]:
    """Select inventory records to verify/provision.

    Default (model_ids is None) requires the full Base+Concise+Detailed
    catalog -- that is the readiness gate for a complete S25 Local rollout.
    An explicit model_ids subset (e.g. just Base) is a smaller, still-exact
    check: a device should advertise what it was actually given, not what
    the full catalog wishes were installed.
    """
    if model_ids is None:
        if len(records) != 3:
            raise RuntimeError(
                "S25 GenieX inventory must contain exactly Base, Concise, and Detailed"
            )
        return records
    requested = list(dict.fromkeys(model_ids))
    by_id = {record["model_id"]: record for record in records}
    missing = [model_id for model_id in requested if model_id not in by_id]
    if missing:
        raise RuntimeError(f"--model-id requested {missing} not present in inventory")
    return [by_id[model_id] for model_id in requested]


def assert_safe(record: dict[str, Any]) -> None:
    for field in ("model_id", "artifact_id", "source_directory"):
        value = str(record[field])
        if not SAFE_ID.fullmatch(value):
            raise RuntimeError(f"unsafe {field}: {value!r}")
    if record.get("geniex_autoregressive_ready") is not True:
        raise RuntimeError(
            f"{record['model_id']} is not certified for GenieX autoregressive execution"
        )
    if set(record.get("graph_names", [])) != EXPECTED_GRAPHS:
        raise RuntimeError(
            f"{record['model_id']} does not declare the complete prompt/decode graph set"
        )


def manifest_entry(record: dict[str, Any], tree_hash: str) -> dict[str, Any]:
    profile = str(record.get("behavior_profile_id", ""))
    mode = "baked_profile" if profile else "none"
    return {
        "model_id": record["model_id"],
        "artifact_id": record["artifact_id"],
        "model_version": record["model_version"],
        "runtime": "genie",
        "artifact_path": record["model_id"],
        "checksum": f"sha256-tree:{tree_hash}",
        "tokenizer_id": record["tokenizer_id"],
        "precision": "w4a16",
        "supported_accelerators": ["htp"],
        "min_memory_mb": 2048,
        "max_context_tokens": 512,
        "supports_steering": False,
        "supports_data_parallel": True,
        "supports_layer_pipeline": False,
        "model_family": "qwen3",
        "role": "small_chat",
        "task_classes": ["chat_qa", "summarization", "translation_rewrite"],
        "quality_score": 0.70,
        "steering_vector_ids": [],
        "supported_steering_layers": [],
        "runtime_version": "GenieX-0.3.5 / QAIRT-2.45",
        "steering_mode": mode,
        "behavior_profile_id": profile,
        "target_compatibility_class": "android-arm64-v8a-sm8750-v79-qairt-2.45-geniex-0.3.5",
        "runtime_options": {"backend": "htp", "max_new_tokens": 96},
    }


def phone_sha256(args: argparse.Namespace, path: str, *, app_private: bool) -> str:
    output = run_as(args, f"sha256sum '{path}'") if app_private else shell(
        args, f"sha256sum '{path}'"
    )
    return output.split()[0].lower()


def main() -> None:
    args = parse_args()
    if args.package != PACKAGE:
        raise RuntimeError(f"this physical provisioning flow is restricted to {PACKAGE}")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    records = select_records(inventory.get("artifacts", []), args.model_id)

    prepared: list[tuple[dict[str, Any], Path, str, dict[str, str]]] = []
    for record in records:
        assert_safe(record)
        bundle = discover_bundle(args.cache_root / record["source_directory"])
        tree_hash, file_hashes = digest_tree(bundle)
        expected = str(record["sha256_tree"]).lower()
        if tree_hash != expected:
            raise RuntimeError(
                f"{record['model_id']} tree checksum mismatch: expected {expected}, got {tree_hash}"
            )
        expected_files = {key: value.lower() for key, value in record["files"].items()}
        if file_hashes != expected_files:
            raise RuntimeError(f"{record['model_id']} per-file inventory does not match source bundle")
        prepared.append((record, bundle, tree_hash, file_hashes))

    if args.verify_only:
        print("ARTIFACT CACHE CHECKSUM VERIFIED")
        for record, _, tree_hash, _ in prepared:
            print(f"  {record['model_id']} sha256-tree:{tree_hash}")
        return
    if not args.serial:
        raise RuntimeError("--serial is required unless --verify-only is used")

    remote_root = f"/data/local/tmp/dragonnest-geniex-{uuid.uuid4().hex}"
    shell(args, f"mkdir -p '{remote_root}'")
    try:
        # Upload and verify the shell-owned temporary copy before changing app-private state.
        for record, bundle, _, file_hashes in prepared:
            remote = f"{remote_root}/{record['model_id']}"
            shell(args, f"mkdir -p '{remote}'")
            for relative in file_hashes:
                source = bundle / Path(relative)
                remote_file = f"{remote}/{relative}"
                remote_parent = remote_file.rsplit("/", 1)[0]
                shell(args, f"mkdir -p '{remote_parent}'")
                adb(args, "push", str(source), remote_file, capture=False)
            for relative, expected in file_hashes.items():
                actual = phone_sha256(args, f"{remote}/{relative}", app_private=False)
                if actual != expected:
                    raise RuntimeError(f"temporary phone checksum mismatch: {record['model_id']}/{relative}")

        entries = [manifest_entry(record, tree_hash) for record, _, tree_hash, _ in prepared]
        manifest = {"format": "dragonnest-android-artifacts-v1", "models": entries}
        with tempfile.TemporaryDirectory(prefix="dragonnest-manifest-") as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            remote_manifest = f"{remote_root}/manifest.json"
            adb(args, "push", str(manifest_path), remote_manifest, capture=False)

        run_as(args, f"mkdir -p '{MODEL_ROOT}'")
        for record, _, _, file_hashes in prepared:
            model_id = record["model_id"]
            incoming = f"{MODEL_ROOT}/.{model_id}.incoming"
            final = f"{MODEL_ROOT}/{model_id}"
            run_as(
                args,
                f"rm -rf '{incoming}' && mkdir -p '{incoming}' && "
                f"cp -R '{remote_root}/{model_id}/.' '{incoming}/'",
            )
            for relative, expected in file_hashes.items():
                actual = phone_sha256(args, f"{incoming}/{relative}", app_private=True)
                if actual != expected:
                    raise RuntimeError(f"app-private checksum mismatch: {model_id}/{relative}")
            run_as(args, f"rm -rf '{final}' && mv '{incoming}' '{final}'")

        run_as(
            args,
            f"cp '{remote_root}/manifest.json' '{MODEL_ROOT}/manifest.json.incoming' && "
            f"mv '{MODEL_ROOT}/manifest.json.incoming' '{MODEL_ROOT}/manifest.json'",
        )
        adb(args, "shell", "am", "force-stop", args.package)
        adb(args, "shell", "monkey", "-p", args.package, "1")
    finally:
        shell(args, f"rm -rf '{remote_root}'")

    print("ARTIFACTS INSTALLED AND CHECKSUM VERIFIED")
    for record, _, tree_hash, _ in prepared:
        print(f"  {record['model_id']} sha256-tree:{tree_hash}")
    print("PersonaCare restarted; the Agent advertises only artifacts that pass its GenieX load probe.")


if __name__ == "__main__":
    main()

"""Verify a local artifact cache against docs/results/demo_artifact_inventory.json.

Usage:
    python scripts/artifact_tools/verify_demo_cache.py <cache_root>

For every pipeline entry with a recorded sha256/size, resolves its canonical
target/stage directory and reports OK / MISSING /
CHECKSUM_MISMATCH. Entries with no recorded checksum (bytes not staged on
this host) are reported as NO_CHECKSUM_RECORDED, not silently skipped.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "results" / "demo_artifact_inventory.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_candidate(cache_root: Path, artifact_id: str) -> Path | None:
    match = re.fullmatch(r"qwen3-1\.7b-s([0-3])-(s25|xelite)", artifact_id)
    if match is None:
        return None
    stage, target = match.groups()
    directory = cache_root / f"qwen3-1.7b/{target}/stage-{stage}"
    files = sorted(directory.rglob("*.bin")) if directory.is_dir() else []
    if len(files) > 1:
        raise RuntimeError(f"{artifact_id}: multiple .bin candidates under {directory}")
    return files[0] if files else None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_demo_cache.py <cache_root>", file=sys.stderr)
        return 2
    cache_root = Path(sys.argv[1])
    data = json.loads(INVENTORY.read_text())
    ok = missing = mismatch = unchecked = 0
    for r in data["artifacts"]:
        aid = r["logical_artifact_id"]
        expected = r.get("sha256") or r.get("sha256_tree")
        if not expected:
            print(f"{aid:<28} NO_CHECKSUM_RECORDED")
            unchecked += 1
            continue
        if not aid.startswith("qwen3-1.7b-"):
            print(f"{aid:<28} NOT_PIPELINE_CACHE_ENTRY")
            unchecked += 1
            continue
        found = find_candidate(cache_root, aid)
        if found is None:
            print(f"{aid:<28} MISSING")
            missing += 1
            continue
        if found.stat().st_size != r.get("size_bytes"):
            print(
                f"{aid:<28} SIZE_MISMATCH expected={r.get('size_bytes')} "
                f"actual={found.stat().st_size}"
            )
            mismatch += 1
            continue
        actual = sha256_of(found)
        if actual == expected:
            print(f"{aid:<28} OK ({found})")
            ok += 1
        else:
            print(f"{aid:<28} CHECKSUM_MISMATCH expected={expected} actual={actual}")
            mismatch += 1
    print(f"\nok={ok} missing={missing} mismatch={mismatch} unchecked={unchecked}")
    return 1 if (missing or mismatch) else 0


if __name__ == "__main__":
    raise SystemExit(main())

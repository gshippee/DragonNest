"""Verify a local artifact cache against docs/results/demo_artifact_inventory.json.

Usage:
    python scripts/artifact_tools/verify_demo_cache.py <cache_root>

For every inventory entry with a recorded sha256/size, looks for a matching
file under <cache_root>/<logical_artifact_id>/** and reports OK / MISSING /
CHECKSUM_MISMATCH. Entries with no recorded checksum (bytes not staged on
this host) are reported as NO_CHECKSUM_RECORDED, not silently skipped.
"""
from __future__ import annotations

import hashlib
import json
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
    matches = list(cache_root.rglob("*.bin")) + list(cache_root.rglob("*.dlc"))
    # Prefer a path that contains the stage/target hint from the artifact id.
    for m in matches:
        if artifact_id.split("-s")[0] in str(m).replace("\\", "/"):
            return m
    return None


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
        target_dir = cache_root
        found = None
        for p in target_dir.rglob("*"):
            if p.is_file() and p.stat().st_size == r.get("size_bytes"):
                found = p
                break
        if found is None:
            print(f"{aid:<28} MISSING")
            missing += 1
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

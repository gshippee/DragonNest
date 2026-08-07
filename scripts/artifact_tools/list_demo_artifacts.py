"""Print the demo artifact inventory as a status matrix.

Reads docs/results/demo_artifact_inventory.json (committed, sanitized, no
local paths) and prints one line per artifact. Does not touch the cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "results" / "demo_artifact_inventory.json"


def main() -> int:
    data = json.loads(INVENTORY.read_text())
    rows = data["artifacts"]
    width = max(len(r["logical_artifact_id"]) for r in rows)
    for r in rows:
        size = r.get("size_bytes")
        size_str = f"{size / 1e6:.1f} MB" if size else "n/a"
        print(f"{r['logical_artifact_id']:<{width}}  {r['status']:<14}  {size_str:>10}  {r.get('target_compatibility_class', '')}")
    check = data.get("cross_target_boundary_check", {})
    print()
    print(f"cross_target_boundary_check: {check.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dragon_nest.artifacts import ArtifactError, calculate_checksum


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate a checksum compatible with DragonNest artifact manifests"
    )
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    try:
        print(calculate_checksum(args.artifact))
    except ArtifactError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

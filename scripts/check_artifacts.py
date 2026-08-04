from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dragon_nest.artifacts import ArtifactError, ArtifactRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DragonNest model artifacts")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs" / "model-artifacts.yaml",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Only verify paths; never use this when advertising Agent capabilities.",
    )
    args = parser.parse_args()

    registry = ArtifactRegistry.from_yaml(args.manifest)
    failed = False
    for artifact in registry.all():
        try:
            path = registry.validate(
                artifact.model_id, verify_checksum=not args.skip_checksum
            )
        except ArtifactError as exc:
            failed = True
            print(f"UNAVAILABLE {artifact.model_id}: {exc}")
        else:
            print(f"READY       {artifact.model_id}: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

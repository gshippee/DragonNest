from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dragon_nest.artifacts import calculate_checksum  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download one existing AI Hub model and print its DragonNest checksum."
    )
    parser.add_argument("model_id", help="existing AI Hub model ID; no compile is submitted")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        import qai_hub
    except ImportError as exc:
        raise SystemExit(
            "qai-hub is unavailable; use the isolated requirements-ai-hub.txt environment"
        ) from exc
    client = qai_hub.Client()
    model = client.get_model(args.model_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(model.download(str(args.output)))
    print(f"Downloaded: {downloaded}")
    print(calculate_checksum(downloaded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

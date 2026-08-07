"""Build phone-demo-bundle/manifest.json + directory tree from a local cache.

Prepares pre-provisioned bytes for the Android artifact installer. Does not
touch AI Hub and does not fabricate bytes: any artifact missing from the
cache is listed in manifest.json with status MISSING rather than silently
dropped, so the installer/build step fails loudly instead of shipping a
partial fleet unnoticed.

Usage: python scripts/artifact_tools/stage_android_artifacts.py <cache_root> <out_dir>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ARTIFACT_DIRS = {
    "qwen3-0.6b-base": "qwen3-0.6b/s25/base",
    "qwen3-0.6b-concise": "qwen3-0.6b/s25/concise-l7-alpha-m4",
    "qwen3-1.7b-s0": "qwen3-1.7b/s25/stage-0",
    "qwen3-1.7b-s1": "qwen3-1.7b/s25/stage-1",
    "qwen3-1.7b-s2": "qwen3-1.7b/s25/stage-2",
    "qwen3-1.7b-s3": "qwen3-1.7b/s25/stage-3",
}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: stage_android_artifacts.py <cache_root> <out_dir>", file=sys.stderr)
        return 2
    cache_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"schema_version": 1, "artifacts": {}}
    for name, rel in ARTIFACT_DIRS.items():
        src = cache_root / rel
        dst = out_dir / name
        if not src.exists() or not any(src.iterdir()):
            manifest["artifacts"][name] = {"status": "MISSING", "source": str(src)}
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        manifest["artifacts"][name] = {"status": "STAGED"}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    missing = [k for k, v in manifest["artifacts"].items() if v["status"] == "MISSING"]
    if missing:
        print(f"WARNING: {len(missing)} artifact(s) missing from cache: {', '.join(missing)}")
    print(f"Wrote {out_dir / 'manifest.json'}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

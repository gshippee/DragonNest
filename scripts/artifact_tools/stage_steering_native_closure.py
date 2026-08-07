#!/usr/bin/env python3
"""Stage the physically proven forked-GenieX native closure beside the stock runtime.

The PersonaCare hardware APK packages Qualcomm's stock
``com.qualcomm.qti:geniex-android:0.3.5`` AAR, whose native closure is what the
accepted Balanced/Base Local path runs on. SteerLab's forked GenieX ships
*the same sonames* with different bytes, so dropping it into ``vendor/jniLibs``
would silently replace the stock runtime for every model -- including Base.
That is exactly the regression this tool exists to prevent.

Instead of rebuilding the fork (its toolchain is not on this machine, and a
rebuild would no longer be the physically proven binary), the colliding
sonames are renamed *in place* to equal-length names. ``DT_SONAME`` and
``DT_NEEDED`` are offsets into ``.dynstr``; replacing a string with another of
identical length leaves every offset, section size, and program header
untouched, so no ELF rewriting or ``patchelf``-class tooling is required. Each
colliding soname occurs exactly once per file, which is asserted before any
byte is written.

The result is two independent GenieX closures in one APK:

    stock AAR   libgeniex.so   libgeniex_core.so   ...  -> "genie" runtime
    fork        libgnxfrk.so   libgnxfrk_core.so   ...  -> "genie_aux" runtime

``libgeniex_plugin.so`` and ``libsteeringlab_jni.so`` keep their filenames:
neither name exists in the stock AAR, and the proven JNI shim ``dlopen``s the
plugin by that exact filename. Their *dependencies* are still patched.

QNN libraries are deliberately not staged. Four of the fork's five are
byte-identical to the AAR's, and ``libQnnHtpV79Skel.so`` -- whose filename the
DSP resolves and therefore cannot be renamed -- is left to the stock AAR, whose
``libQnnHtp.so`` and ``libQnnHtpV79Stub.so`` are byte-identical to the fork's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

# Colliding sonames -> equal-length replacements. Every pattern ends in ".so",
# so no key is a substring of another and replacement order is irrelevant.
RENAMES = {
    "libgeniex.so": "libgnxfrk.so",
    "libgeniex_core.so": "libgnxfrk_core.so",
    "libgeniex-proc.so": "libgnxfrk-proc.so",
    "libgeniex_vlm.so": "libgnxfrk_vlm.so",
    "libgeniex-proc-vision.so": "libgnxfrk-proc-vision.so",
}

# Fork-only filenames: no stock collision, and both are referenced by name at
# runtime (System.loadLibrary / dlopen), so the files keep their names.
KEEP_FILENAMES = ("libgeniex_plugin.so", "libsteeringlab_jni.so")

STAGED = tuple(RENAMES) + KEEP_FILENAMES

APK_LIB_PREFIX = "lib/arm64-v8a/"
AAR_LIB_PREFIX = "jni/arm64-v8a/"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_sonames(data: bytes, source_name: str) -> tuple[bytes, dict[str, int]]:
    """Replace colliding soname strings with equal-length fork-private names."""
    applied: dict[str, int] = {}
    for old, new in RENAMES.items():
        if len(old) != len(new):  # pragma: no cover - guarded by test
            raise SystemExit(
                f"rename {old} -> {new} changes length; .dynstr offsets would shift"
            )
        count = data.count(old.encode())
        if count == 0:
            continue
        if count != 1:
            raise SystemExit(
                f"{source_name}: expected exactly one occurrence of {old!r}, found "
                f"{count}. Equal-length patching is only safe for the single "
                f".dynstr entry; refusing to guess."
            )
        data = data.replace(old.encode(), new.encode())
        applied[old] = count
    return data, applied


def stock_aar_lib_names(aar: Path | None) -> set[str]:
    if aar is None or not aar.is_file():
        return set()
    with zipfile.ZipFile(aar) as archive:
        return {
            name[len(AAR_LIB_PREFIX):]
            for name in archive.namelist()
            if name.startswith(AAR_LIB_PREFIX) and name.endswith(".so")
        }


def stage(apk: Path, out_dir: Path, stock_names: set[str]) -> dict:
    with zipfile.ZipFile(apk) as archive:
        available = {
            name[len(APK_LIB_PREFIX):]: name
            for name in archive.namelist()
            if name.startswith(APK_LIB_PREFIX)
        }
        missing = [name for name in STAGED if name not in available]
        if missing:
            raise SystemExit(f"{apk} is missing forked libraries: {missing}")

        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for source_name in STAGED:
            raw = archive.read(available[source_name])
            patched, applied = patch_sonames(raw, source_name)
            target_name = RENAMES.get(source_name, source_name)
            if target_name in stock_names:
                raise SystemExit(
                    f"refusing to stage {target_name}: it collides with the stock "
                    f"GenieX AAR and would replace the accepted Base runtime"
                )
            (out_dir / target_name).write_bytes(patched)
            records.append(
                {
                    "source_name": source_name,
                    "staged_name": target_name,
                    "size_bytes": len(patched),
                    "source_sha256": sha256(raw),
                    "staged_sha256": sha256(patched),
                    "patched_references": applied,
                }
            )

    # Any stale file from an earlier layout would be packaged too.
    staged_names = {record["staged_name"] for record in records}
    for existing in sorted(out_dir.glob("*.so")):
        if existing.name not in staged_names:
            raise SystemExit(
                f"unexpected pre-existing library {existing}; remove it before staging"
            )

    return {
        "source_apk": str(apk),
        "source_apk_sha256": sha256(apk.read_bytes()),
        "staged_dir": str(out_dir),
        "renames": RENAMES,
        "kept_filenames": list(KEEP_FILENAMES),
        "libraries": records,
        "stock_aar_collisions": sorted(staged_names & stock_names),
    }


def verify(out_dir: Path, manifest: dict, stock_names: set[str]) -> None:
    for record in manifest["libraries"]:
        path = out_dir / record["staged_name"]
        if not path.is_file():
            raise SystemExit(f"missing staged library: {path}")
        actual = sha256(path.read_bytes())
        if actual != record["staged_sha256"]:
            raise SystemExit(
                f"{path} sha256 {actual} != recorded {record['staged_sha256']}"
            )
        if record["staged_name"] in stock_names:
            raise SystemExit(f"{path} collides with the stock GenieX AAR")
    print(f"verified {len(manifest['libraries'])} staged libraries in {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk", type=Path, help="SteerLab proof APK holding the fork")
    parser.add_argument("--out", type=Path, required=True, help="jniLibs/<abi> dir")
    parser.add_argument(
        "--stock-aar",
        type=Path,
        help="stock geniex-android AAR; staged names are checked against it",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    stock_names = stock_aar_lib_names(args.stock_aar)
    if args.stock_aar and not stock_names:
        raise SystemExit(f"no arm64-v8a libraries found in {args.stock_aar}")

    if args.verify_only:
        if not args.manifest.is_file():
            raise SystemExit(f"missing manifest: {args.manifest}")
        verify(args.out, json.loads(args.manifest.read_text()), stock_names)
        return 0

    if args.apk is None:
        raise SystemExit("--apk is required unless --verify-only is passed")
    if not args.apk.is_file():
        raise SystemExit(f"missing SteerLab APK: {args.apk}")
    if args.out.exists():
        shutil.rmtree(args.out)

    manifest = stage(args.apk, args.out, stock_names)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    for record in manifest["libraries"]:
        renamed = record["source_name"] != record["staged_name"]
        print(
            f"  {'renamed ' if renamed else 'kept    '}"
            f"{record['source_name']:26} -> {record['staged_name']:26} "
            f"{record['size_bytes']:>11,} bytes"
        )
    print(f"\nstaged {len(manifest['libraries'])} libraries into {args.out}")
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

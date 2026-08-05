#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_root="${DRAGONNEST_QAIRT_SDK_ROOT:-}"
bundle=""
vendor_root="$repo_root/android-agent/vendor"
model_id="qwen3-1.7b-s25-genie"
model_version="qairt-2.48.0.260626-s25"
min_memory_mb=4096

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_android_genie_runtime.sh --bundle <genie-bundle> \
      [--qairt-sdk <sdk-root>] [--model-id <id>] [--model-version <version>] \
      [--min-memory-mb <mb>] [--vendor-root <directory>]

Stages an AI Hub geniex_qairt bundle and the matching QAIRT Android libraries
into the Git-ignored Android vendor directory. The next hardware build must set
DRAGONNEST_QAIRT_SDK_ROOT to the same SDK root.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) bundle="$2"; shift 2 ;;
    --qairt-sdk) sdk_root="$2"; shift 2 ;;
    --model-id) model_id="$2"; shift 2 ;;
    --model-version) model_version="$2"; shift 2 ;;
    --min-memory-mb) min_memory_mb="$2"; shift 2 ;;
    --vendor-root) vendor_root="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$bundle" || -z "$sdk_root" ]]; then
  usage >&2
  exit 2
fi

bundle="$(realpath "$bundle")"
sdk_root="$(realpath "$sdk_root")"
if [[ ! -f "$bundle/genie_config.json" || ! -f "$bundle/tokenizer.json" ]]; then
  echo "Expected a Genie bundle containing genie_config.json and tokenizer.json: $bundle" >&2
  exit 2
fi

sdk_lib="$sdk_root/lib/aarch64-android"
for library in \
  libGenie.so \
  libQnnSystem.so \
  libQnnHtp.so \
  libQnnHtpPrepare.so \
  libQnnHtpV79Stub.so; do
  if [[ ! -f "$sdk_lib/$library" ]]; then
    echo "Required QAIRT Android library is missing: $sdk_lib/$library" >&2
    exit 2
  fi
done

if [[ ! "$min_memory_mb" =~ ^[0-9]+$ ]] || [[ "$min_memory_mb" -eq 0 ]]; then
  echo "--min-memory-mb must be a positive integer" >&2
  exit 2
fi

jni_dir="$vendor_root/jniLibs/arm64-v8a"
model_root="$vendor_root/model-assets/models"
target_bundle="$model_root/$model_id"
mkdir -p "$jni_dir" "$model_root"

if [[ -e "$target_bundle" ]]; then
  echo "Refusing to overwrite existing staged model: $target_bundle" >&2
  echo "Remove it explicitly after confirming it is no longer needed." >&2
  exit 2
fi

for library in \
  libGenie.so \
  libQnnSystem.so \
  libQnnHtp.so \
  libQnnHtpPrepare.so \
  libQnnHtpV79Stub.so; do
  install -m 0644 "$sdk_lib/$library" "$jni_dir/$library"
done
cp -a "$bundle" "$target_bundle"

bundle_checksum="$({
  cd "$target_bundle"
  while IFS= read -r -d '' path; do
    printf '%s\0' "${path#./}"
    cat "$path"
    printf '\0'
  done < <(find . -type f -print0 | sort -z)
} | sha256sum | awk '{print $1}')"

export BUNDLE_CHECKSUM="$bundle_checksum"
export MODEL_ID="$model_id"
export MODEL_VERSION="$model_version"
export MIN_MEMORY_MB="$min_memory_mb"
export MANIFEST_PATH="$model_root/manifest.json"
python3 - <<'PY'
import json
import os
from pathlib import Path

manifest = {
    "models": [{
        "model_id": os.environ["MODEL_ID"],
        "model_version": os.environ["MODEL_VERSION"],
        "runtime": "genie",
        "artifact_path": os.environ["MODEL_ID"],
        "checksum": "sha256-tree:" + os.environ["BUNDLE_CHECKSUM"],
        "tokenizer_id": "Qwen/Qwen3-1.7B",
        "precision": "w4a16",
        "supported_accelerators": ["htp"],
        "min_memory_mb": int(os.environ["MIN_MEMORY_MB"]),
        "max_context_tokens": 512,
        "supports_steering": False,
        "supports_data_parallel": True,
        "supports_layer_pipeline": False,
        "model_family": "qwen3",
        "role": "small_chat",
        "task_classes": [
            "chat_qa",
            "summarization",
            "translation_rewrite",
            "reasoning_analysis",
            "code_assistance",
        ],
        "quality_score": 0.84,
        "runtime_version": "QAIRT-2.48.0.260626",
        "runtime_options": {"backend": "htp", "config": "genie_config.json"},
    }]
}
Path(os.environ["MANIFEST_PATH"]).write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

echo "Staged Genie model: $target_bundle"
echo "Staged manifest: $model_root/manifest.json"
echo "Build with: DRAGONNEST_QAIRT_SDK_ROOT=$sdk_root DRAGONNEST_ANDROID_INCLUDE_MODEL_ARTIFACTS=true scripts/build_android.sh"

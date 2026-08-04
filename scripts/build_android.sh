#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tool_root="${DRAGONNEST_ANDROID_TOOL_ROOT:-/tmp/dragonnest-toolchain}"

export GRADLE_USER_HOME="${GRADLE_USER_HOME:-$tool_root/gradle-home}"
export ANDROID_USER_HOME="${ANDROID_USER_HOME:-$tool_root/android-home}"
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$tool_root/android-sdk}"
export ANDROID_HOME="${ANDROID_HOME:-$ANDROID_SDK_ROOT}"

if [[ -z "${JAVA_HOME:-}" && -x "$tool_root/jdk/bin/java" ]]; then
  export JAVA_HOME="$tool_root/jdk"
fi

mkdir -p "$GRADLE_USER_HOME" "$ANDROID_USER_HOME"

if [[ -z "${JAVA_HOME:-}" || ! -x "$JAVA_HOME/bin/java" ]]; then
  echo "JAVA_HOME must point to JDK 17 or newer." >&2
  exit 2
fi
if [[ ! -d "$ANDROID_SDK_ROOT/platforms/android-35" ]]; then
  echo "Android SDK 35 is required at ANDROID_SDK_ROOT." >&2
  exit 2
fi

gradle_args=()
if [[ "${DRAGONNEST_ANDROID_INCLUDE_MODEL_ARTIFACTS:-true}" == "false" ]]; then
  gradle_args+=("-PincludeModelArtifacts=false")
fi

cd "$repo_root/android-agent"
./gradlew --no-daemon "${gradle_args[@]}" :app:testDebugUnitTest :app:assembleDebug

apk="$repo_root/android-agent/app/build/outputs/apk/debug/app-debug.apk"
echo "APK: $apk"

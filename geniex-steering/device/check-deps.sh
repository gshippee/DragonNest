#!/bin/bash
R=/root/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf
for so in /root/gxq-build/sdk-android/src/libgeniex.so \
          /root/gxq-build/sdk-android/plugins/qairt/libgeniex_plugin.so \
          /root/gxq-build/sdk-android/bin/libgeniex_vlm.so \
          /root/gxq-build/sdk-android/lib/libgeniex-proc-vision.so; do
  echo "== $(basename $so)"
  "$R" -d "$so" | grep NEEDED
done

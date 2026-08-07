#!/bin/bash
# Builds the SteerLab JNI shim and stages all runtime .so files into jniLibs.
set -e
rsync -a /mnt/c/Users/shubh/Downloads/qcom_hackathon2/geniex-steering/apk/ /root/steerlab-apk/
NDK=/root/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin
J=/root/steerlab-apk/app/src/main/jniLibs/arm64-v8a
mkdir -p "$J"
"$NDK/aarch64-linux-android31-clang++" -shared -fPIC -O2 -std=c++17 -static-libstdc++ \
  -I/root/geniex-fork/sdk/include \
  /root/steerlab-apk/app/src/main/cpp/steeringlab_jni.cpp \
  -L/root/gxq-build/sdk-android/src -lgeniex -llog \
  -o "$J/libsteeringlab_jni.so"
echo JNI_OK
cp /root/gxq-build/sdk-android/src/libgeniex.so \
   /root/gxq-build/sdk-android/plugins/qairt/libgeniex_plugin.so \
   /root/gxq-build/sdk-android/bin/libgeniex_core.so \
   /root/gxq-build/sdk-android/bin/libgeniex_vlm.so \
   /root/gxq-build/sdk-android/lib/libgeniex-proc.so \
   /root/gxq-build/sdk-android/lib/libgeniex-proc-vision.so \
   /root/gxq-build/android/bin/htp-files/libQnnHtp.so \
   /root/gxq-build/android/bin/htp-files/libQnnSystem.so \
   /root/gxq-build/android/bin/htp-files/libQnnHtpNetRunExtensions.so \
   /root/gxq-build/android/bin/htp-files/libQnnHtpV79Stub.so \
   /root/gxq-build/android/bin/htp-files/libQnnHtpV79Skel.so \
   "$J/"
ls "$J"

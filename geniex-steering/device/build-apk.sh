#!/bin/bash
# Builds SteerLab debug APK in WSL. Assumes build-jni.sh already staged jniLibs.
set -e
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
if [ ! -d /root/gradle-8.7 ]; then
  cd /root && curl -sLO https://services.gradle.org/distributions/gradle-8.7-bin.zip \
    && unzip -q gradle-8.7-bin.zip && rm gradle-8.7-bin.zip
fi
cd /root/steerlab-apk
echo "sdk.dir=/root/android-sdk" > local.properties
/root/gradle-8.7/bin/gradle --no-daemon :app:assembleDebug
ls -la app/build/outputs/apk/debug/

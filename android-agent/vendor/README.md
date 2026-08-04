# Android Vendor Runtime Inputs

This directory is intentionally excluded from Git. A real Snapdragon build puts
licensed Qualcomm SDK libraries and validated model assets here before building
the APK:

```text
vendor/
  jniLibs/arm64-v8a/*.so
  model-assets/models/manifest.json
  model-assets/models/<artifact paths declared in the manifest>
```

The app includes a direct QAIRT 2.48 Genie JNI bridge. Build it with
`DRAGONNEST_QAIRT_SDK_ROOT` set to the extracted SDK root, then use
`scripts/prepare_android_genie_runtime.sh` to stage the required Android shared
objects and a compiled bundle here. The optional AAR extension route remains
available for a raw QNN bridge.

An optional AAR may provide this public no-argument class:

- `com.dragonnest.agent.vendor.QnnRuntimeBridge`

The class implements `com.dragonnest.agent.AndroidRuntimeBridge`. `isAvailable`
must load/probe the declared artifact and return `false` if the runtime cannot
execute it on the target. `execute` must run the supplied artifact, return text
or a `BoundaryTensor`, and report the actual accelerator. The base APK will not
advertise a model to the Brain until its SHA-256 checksum and bridge probe pass.

`model-assets/models/manifest.json` uses the Android manifest schema described
in `android-agent/README.md`. Paths are relative to `models/` and are copied to
the app-private `files/dragonnest-models/` directory at first launch.

# S25 physical handoff: Qwen3-0.6B base and baked-concise

This is the no-reasoning handoff for the next Samsung Galaxy S25 session. It
does **not** create a steering vector or use runtime steering.

Evidence labels used below:

- **[repo]** checked into DragonNest;
- **[local]** inspected in the adjacent PersonaCare workspace/release assets;
- **[physical pending]** requires the S25;
- **[desktop prerequisite]** must be completed and tested before handing an APK
  to the phone operator.

## What already exists

**[local]** Both external bundles are present and their ZIP SHA-256 values
match the `static-steering-v1` release manifest and DragonNest provenance:

| DragonNest model ID | release ZIP | ZIP bytes | ZIP SHA-256 |
| --- | --- | ---: | --- |
| `qwen3-0.6b-s25-base` | `qwen3-0.6b-base-sm8750.zip` | 629468676 | `2854ef411208b1315855584b223e32627922c8bb5b192ce88d7ee13010d5c8fe` |
| `qwen3-0.6b-s25-concise` | `qwen3-0.6b-steered-l7-alpha-m4-sm8750.zip` | 629447381 | `e932381129d5d93514d6cefadff7e4bfe55dd6cf23532c183e17cfc13937a183` |

Each ZIP contains `genie_config.json`, `tokenizer.json`,
`htp_backend_ext_config.json`, `metadata.json`, and the linked
`part1_of_2.bin`/`part2_of_2.bin` contexts. The metadata targets Samsung Galaxy
S25 Family / `sm8750-ac` / Hexagon v79 and records QAIRT compiler
`2.45.0.260326154327`. The matching local runtime SDK is
`2.45.41.260507`; the original Android demo used GenieX Android `0.3.5`.

**[repo]** The concise bundle is an ordinary statically modified graph:

- realization: `baked_profile`;
- behavior profile: `concise`;
- provenance vector: `concise-vs-verbose-layer-7`;
- fixed layer/alpha provenance: layer 7, alpha -4;
- `supports_steering: false` (there is no runtime vector input).

Cloud compile/link and bundle/APK validation exist. Final integrated
DragonNest execution on the physical S25 is still pending.

## Hard runtime-version gate

Do **not** stage these QAIRT 2.45 contexts with the current QAIRT 2.48 bridge
and call the combination validated.

**[repo]** DragonNest currently builds a direct Genie C-API JNI bridge against
the selected `DRAGONNEST_QAIRT_SDK_ROOT`. Its existing staging script requires
these exact SDK libraries under `lib/aarch64-android/`:

```text
libGenie.so
libQnnSystem.so
libQnnHtp.so
libQnnHtpPrepare.so
libQnnHtpV79Stub.so
```

The APK also builds `libdragonnest_genie_jni.so`. The direct bridge is
documented and tested as the QAIRT 2.48 path.

**[local]** The available QAIRT `2.45.41.260507` SDK contains the QNN v79
runtime pieces but does not contain `libGenie.so` or `libQnnHtpPrepare.so` at
the locations required by `scripts/prepare_android_genie_runtime.sh`.
Therefore the current staging/build command cannot produce a matching 2.45
DragonNest Genie APK. There is also no local proof that a QAIRT 2.48 Genie/QNN
runtime can load these 2.45 compiled contexts.

**[desktop prerequisite]** The shortest safe bridge is to package the same
`com.qualcomm.qti:geniex-android:0.3.5` runtime used by the original verified
APK behind DragonNest's `AndroidRuntimeBridge` contract. Treat that AAR and
its transitive native libraries as the authoritative native closure; do not
manually cherry-pick `.so` files from an old APK. Its inspected APK includes,
among its runtime closure, `libgeniex.so`, `libgeniex_core.so`,
`libgeniex_plugin_qairt.so`, `libnpu_jni.so`, `libQnnSystem.so`,
`libQnnHtp.so`, `libQnnHtpPrepare.so`, and the v79 stub/skel libraries.

The adapter must make `isAvailable()` create/probe the declared bundle and
must report false on any ABI, context-load, or HTP failure. `execute()` must
return the generated text with accelerator `htp`. This is a desktop coding and
emulator/unit-test task; it must be complete before the physical session.

## Artifact staging and Android manifest

**[repo]** Packaged model assets are copied at first launch from:

```text
android-agent/vendor/model-assets/models/
```

to app-private `files/dragonnest-models/`. The registry accepts a JSON
`models` array, confines paths to that directory, and verifies every artifact
before advertisement. For these directory bundles, `checksum` must be the
post-extraction `sha256-tree:` digest produced with DragonNest's canonical
relative-path/content algorithm. The ZIP SHA-256 values above are release
download checks; they are **not** valid substitutes for the tree hashes.

The two manifest entries must use these routing identities:

| field | base | concise |
| --- | --- | --- |
| `model_id` | `qwen3-0.6b-s25-base` | `qwen3-0.6b-s25-concise` |
| `artifact_id` | `qwen3-0.6b-w4a16-sm8750-base-qairt245` | `qwen3-0.6b-w4a16-sm8750-concise-l7-am4-qairt245` |
| `runtime` | `genie` | `genie` |
| `runtime_version` | `QAIRT-2.45 / GenieX-0.3.5` | `QAIRT-2.45 / GenieX-0.3.5` |
| `target_compatibility_class` | `android-arm64-sm8750-v79-qairt-2.45-geniex-0.3.5` | same |
| `steering_mode` | `none` | `baked_profile` |
| `behavior_profile_id` | empty/omitted | `concise-l7-alpha-m4` |
| `supports_steering` | `false` | `false` |

Both entries also retain model version
`c1899de289a04d12100db370d81485cdf75e47ca`, tokenizer
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, W4A16,
HTP, 512 maximum context tokens, and 2048 MB minimum memory. Do not add a
runtime steering vector ID to the Android capability. The Brain catalog holds
the concise artifact's vector provenance.

### Coexistence decision

**[repo]** The registry and runtime catalog can hold both model directories in
one APK and advertise both manifest entries. The runtime creates a model
context per request, so both real models must be advertised as installed/cold,
never warm. This is the preferred acceptance build despite its size (about
1.45 GB of uncompressed model content before APK/runtime overhead).

There is currently no post-enrollment artifact downloader/installer in the
integrated DragonNest app. `AndroidModelAssetInstaller` installs signed APK
assets once and then honors its `.installed` marker. Therefore post-enrollment
installation is **not** the shortest supported path. Also, the current staging
script emits one manifest entry per invocation; it must be extended on the
desktop to stage both bundles atomically and emit one two-entry manifest.

## Desktop release gate

Before involving the phone, the desktop owner must produce one hardware APK
and a secret-free build record proving all of the following:

1. both release ZIP hashes equal the table above;
2. both extracted bundle tree hashes equal their manifest values;
3. the matching GenieX 0.3.5 bridge and its complete AAR native closure are in
   the APK (no QAIRT 2.48 substitution claim);
4. `manifest.json` has exactly the two real entries above;
5. Android unit tests and `assembleDebug` pass with
   `DRAGONNEST_ANDROID_INCLUDE_MODEL_ARTIFACTS=true`;
6. APK inspection shows both model directories, the two-entry manifest, and
   arm64 runtime libraries;
7. the resulting APK SHA-256 is recorded.

Verify the release downloads on the desktop with:

```powershell
Get-FileHash -Algorithm SHA256 .\qwen3-0.6b-base-sm8750.zip
Get-FileHash -Algorithm SHA256 .\qwen3-0.6b-steered-l7-alpha-m4-sm8750.zip
```

Do not give the phone operator an APK until the runtime-version gate above is
resolved. The repository's current one-bundle QAIRT 2.48 staging command is
not an operator command for these artifacts.

## No-reasoning physical session

Once the desktop release gate passes, Shubham only performs the following.
Use a fresh install; an `adb install -r` over an older APK preserves the
`.installed` marker and can preserve stale assets.

```powershell
adb devices
adb install .\android-agent\app\build\outputs\apk\debug\app-debug.apk
adb shell am start -n com.dragonnest.agent/.AgentSettingsActivity
```

On the desktop, start the LAN-visible Brain with a non-default token:

```powershell
$env:DRAGONNEST_ENROLLMENT_TOKEN = "<random-shared-token>"
.\scripts\run_demo_brain.ps1
```

Open `http://<desktop-lan-ip>:8080/admin`, create the phone enrollment QR, and
scan it in PersonaCare. Do not put the token in logs or the proof record.

### Physical acceptance test

1. In the Brain device record, require the enrolled phone to be connected and
   routable, with `npu_status=available`, positive live available memory, and
   both exact model/artifact IDs from the table above. Neither real artifact
   may appear in the heartbeat warm-model list.
2. With no behavior profile, submit `Reply with exactly: S25_BASE_OK` through
   the dashboard's normal **Route & execute** behavior/API path for family
   `qwen3`. Require the chosen route to be the phone and
   `qwen3-0.6b-s25-base`.
3. With behavior profile `concise`, submit
   `Reply with exactly: S25_CONCISE_OK` through the same public path. Require
   the chosen route to be the phone and `qwen3-0.6b-s25-concise`, with route
   realization `baked_profile`. Here `concise` is the Brain behavior-profile
   ID; the phone capability retains the bundle-specific provenance ID
   `concise-l7-alpha-m4`, as declared by `configs/model-artifacts.yaml` and
   `configs/hardware-fabric.yaml`.
4. For both results require success, nonempty output, runtime `genie`, actual
   accelerator `htp`, and the expected artifact identity where exposed.
5. Require `supports_steering=false` and no runtime steering input/claim for
   both. The concise route must be labeled `baked_profile`, never
   `runtime_vector` or prompt fallback.
6. Save a secret-free proof containing APK SHA-256, task/attempt IDs, chosen
   routes, scheduler explanations, runtime/accelerator metrics, output hashes,
   and sanitized phone telemetry. Do not save enrollment material, model
   paths, bundle contents, or credentials.

**[physical pending]** Passing those two scheduled round trips is the first
evidence that the external QAIRT 2.45 bundles are integrated into the combined
PersonaCare/DragonNest APK. Until then, retain the existing AI Hub verification
label and do not claim physical S25 execution.

# S25 baked-profile recovery and Detailed build

This record separates three evidence classes: bytes recovered from the working
phone, release archives recovered from the desktop, and the not-yet-physical
Detailed candidate. Model/APK/runtime bytes remain outside Git.

## Preserved working reference

The working reference is `com.personacare.steeringdemo` 1.0 on a Samsung
SM-S938U1 (Android 15, SM8750-ac / Hexagon v79). It is debuggable, so `run-as`
worked. The installed base APK and app-private Base/Concise model trees were
copied without modifying the package. The installed APK contains GenieX/QNN
2.45 native libraries but no model binaries; the executable context binaries
live under app-private `files/models/`.

The installed APK SHA-256 is
`9038cff6548fe7fa1aef3bb007648451d088c04c02f2a18a19245fab1bf7b419`
(99,966,284 bytes). It differs from the earlier archived reference APK
`d8247adf01d2e6c2c612a1471d5257fdef01a2073f4f9cf92120885a71007738`
(99,648,611 bytes), although both report the same package/version/launcher.

The known release ZIPs match their recorded provenance exactly:

- Base: `2854ef411208b1315855584b223e32627922c8bb5b192ce88d7ee13010d5c8fe`
- Concise: `e932381129d5d93514d6cefadff7e4bfe55dd6cf23532c183e17cfc13937a183`

The phone's `.bin` files are larger and do not hash-identically to files in
those release ZIPs. Config, tokenizer, Genie config, and metadata files do
match. The phone binaries are preserved as physical evidence; the release ZIPs
remain the authoritative release provenance. Do not collapse these into a
false byte-for-byte claim.

## Exact static bake

The source is `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`. The recipe normalizes
`concise-vs-verbose-layer-7` to unit L2 norm, multiplies it by alpha, inserts an
ordinary constant ONNX `Add` after
`/model/model/layers.7/Add_1_output_0`, rewires the three consumers, and adds
the matching encoding records. There is no runtime steering input.

`scripts/artifact_tools/bake_qwen06_profile.py` pins the base ONNX and vector
SHA-256 values. Rebuilding Concise with alpha `-4` reproduced the known baked
ONNX exactly:
`03f6a9fa01997098e00720a1abb0dd587591e279e22ac183cd17e2074c4b0438`.
Changing only alpha to `+4` produced the Detailed source ONNX:
`8d5a612b415333067910a41c368cfc4a034c443f02d705fbdbb3e41e2b046714`.

The recovered export command is:

```powershell
python -m qai_hub_models.models.qwen3_0_6b.export `
  --runtime geniex_qairt `
  --checkpoint <baked-checkpoint> `
  --sequence-lengths 128 `
  --context-lengths 512 `
  --device "Samsung Galaxy S25 (Family)" `
  --skip-profiling `
  --zip-assets
```

Historical and Detailed compile jobs explicitly report
`--quantize_full_type w8a16`. Artifact/model names remain `w4a16`; this
unresolved naming/provenance mismatch is intentionally preserved.

Detailed compile jobs: `jprw0m4k5` (part 1) and `jp2ewq76p` (part 2).
Target link/download and physical HTP validation status are recorded in
`docs/results/s25_steering_recovery.json`.

## Product policy

PersonaCare sends only `persona_id = balanced | concise | detailed`. Brain
maps Balanced to the base executable and Concise/Detailed to exact baked
artifacts. All three advertise `supports_steering = false`. An unavailable
exact bake fails as `PROFILE_UNAVAILABLE`; it is never mislabeled as runtime
steering or silently replaced with prompt conditioning.


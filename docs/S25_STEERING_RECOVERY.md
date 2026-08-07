# S25 baked-profile recovery and Detailed build

This record separates three evidence classes: bytes recovered from the working
phone, release archives recovered from the desktop, and the newly built and
physically verified Detailed artifact. Model/APK/runtime bytes remain outside
Git.

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

For DragonNest execution, the active external cache now uses the recovered
phone-class dual-phase bundles rather than the older prompt-only release
copies. `scripts/artifact_tools/stage_s25_geniex_artifacts.py --verify-only`
validates the committed per-file inventory and these tree hashes:

- Base: `efc5728ba3ac7ee4a5bc2ee7fc8aaad8e875d66625234d71180dcec74a695827`
- Concise: `eaa310354020e460d0b7862de4871a031fad8d48c0c2551241a2c78a82f4eb0e`
- Detailed: `460e2c1cbff39607210ef0a6c9ac0fd603729d39046327dce7dc9405f1cf93eb`

The former Base/Concise prompt-only cache trees were moved aside with explicit
`prompt-only-invalid` names; they were not deleted and cannot be provisioned
by the active inventory.

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

The recovered command above compiles the prompt graphs. Packaging only those
graphs produced link jobs `jgnkndxvg` and `jp2ewqjxp`; the resulting bundle was
correctly rejected by GenieX during physical model creation because it lacked
the token/decode graphs. That prompt-only bundle is retained as invalid
evidence and is not schedulable.

The complete Detailed build uses prompt jobs `jprw0m4k5` / `jp2ewq76p` and
token jobs `jp06j8jep` / `jp81xdx85`. Combined prompt+token links are
`jp16jkjl5` and `jgd23y3l5`. The final bundle is 631,616,571 bytes with
SHA-256
`739f2c1a2339b41195591cc89ebeae43af49dbd290bf70982af204ddfdd5f8f9`.
Its context binaries are 311,226,368 and 407,252,992 bytes, matching the size
class of the physically working two-graph phone contexts.

## Physical S25 result

The complete Detailed bundle was copied into the preserved debuggable
`com.personacare.steeringdemo` app-private model store with SHA-256 verification
before and after transfer. On the physical SM-S938U1, GenieX created both Base
and Detailed QAIRT pipelines successfully with `device_id: NPU`; the logs show
the v79 HTP skeleton and no mock executor.

Eight deterministic Base/Concise/Detailed comparisons used the same prompts,
top-k 1, and maximum of 192 generated tokens. Concise (-4) had a median word
delta of -5 versus Base, was shorter in 5/8 runs, and totaled -4 words across
the set. Detailed (+4) had a median delta of +8.5, was longer in 5/8 runs, and
totaled +123 words. Detailed TTFT was 17-19 ms and decode throughput was
120.1-122.8 tokens/s. These are directional behavior observations, not a hard
output-length guarantee.

The preserved APK hardcodes its second-card caption as alpha -4. During the
Detailed run that caption was stale; the app-private Detailed shard-2 SHA-256
was `b7eb9b0db2a2711fe69bcb2438114b756dbc9d28b6872c69958c2ed35fdeac6b`.
Hashes and runtime logs, not that label, identify the executed artifact.

## Product policy

PersonaCare sends only `persona_id = balanced | concise | detailed`. Brain
maps Balanced to the base executable and Concise/Detailed to exact baked
artifacts. All three advertise `supports_steering = false`. An unavailable
exact bake fails as `PROFILE_UNAVAILABLE`; it is never mislabeled as runtime
steering or silently replaced with prompt conditioning.


# X Elite physical worker handoff

The X Elite implementation/proof task is complete. No laptop-side LLM or code
changes are required for the remaining cross-host validation.

From a clean checkout of `origin/main`, Shubham only needs to run:

```powershell
git pull
$env:DRAGONNEST_ENROLLMENT_TOKEN = "<same token printed/used by the desktop Brain>"
.\scripts\run_xelite_worker.ps1 -Brain <desktop-lan-ip>:50051
```

The launcher pins the physically verified Qwen3-4B bundle checksum, validates
the manifest, advertises `pc-01` as installed/cold with runtime steering off,
and starts the normal gRPC Device Agent. It must refuse ambiguous or checksum-
mismatched bundles rather than guessing.

The physical proof is recorded in
`docs/results/xelite_worker_status.md`. The only remaining X Elite claim is a
Brain on a genuinely separate desktop host; the desktop-side validation
harness owns that experiment and its secret-free proof output.

Never commit enrollment tokens, proprietary model/runtime files, absolute
bundle paths, credentials, or verbose runtime logs.

# X Elite physical worker handoff

The X Elite implementation/proof task is complete. The hackathon stage runs
the DragonNest Brain and the `pc-01` worker on this same laptop; no laptop-side
LLM or code changes are required.

From a clean checkout of `origin/main`, Shubham only needs to run:

```powershell
git pull
.\scripts\run_xelite_demo.ps1
```

The demo launcher starts a LAN-visible Brain and dashboard plus a loopback
`pc-01` worker in separate windows with one generated token. The worker pins
the physically verified Qwen3-4B bundle checksum, validates the manifest,
advertises installed/cold with runtime steering off, and refuses ambiguous or
checksum-mismatched bundles rather than guessing.

The physical proof is recorded in
`docs/results/xelite_worker_status.md`. The remaining stage proof starts in
PersonaCare on the S25: normal RAM routes to the phone mock runtime, then a
64 MB simulated heartbeat reroutes the next request to real Genie/HTP here.

Never commit enrollment tokens, proprietary model/runtime files, absolute
bundle paths, credentials, or verbose runtime logs.

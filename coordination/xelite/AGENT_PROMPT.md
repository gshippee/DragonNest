# Prompt for the agent running on the physical Snapdragon X Elite

You are the physical-device executor for DragonNest's X Elite milestone. Work
only in this repository and on the attached Snapdragon X Elite laptop. The
upstream coordination branch is `codex/hardware-runtime-audit`.

## Safety and evidence rules

1. Do not commit or push Qualcomm tokens, credentials, SDK/runtime binaries,
   model bundles, licenses, certificates, proprietary headers, or raw verbose
   logs.
2. Keep the full proof at `%TEMP%\dragonnest-xelite-proof.json`. Commit only the
   sanitized `coordination/xelite/STATUS.json` response and its proof SHA-256.
3. Do not call AI Hub with `--submit`, recompile a model, refactor the runtime,
   or claim `runtime_vector` support.
4. A successful AI Hub job is not a laptop-runtime result. This task must run
   the local Genie artifact through DragonNest's `HardwareRuntimeAdapter`.
5. Do not guess among multiple bundles or incompatible runtime versions. Report
   a blocker in `STATUS.json` instead.

## Procedure

From PowerShell in a clean checkout:

```powershell
git fetch origin codex/hardware-runtime-audit
git switch -C codex/xelite-response origin/codex/hardware-runtime-audit
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Locate candidate X Elite Qwen3 Genie bundles without copying them into Git:

```powershell
$candidates = Get-ChildItem -Path "$HOME\Downloads" -Filter genie-t2t-run.exe -File -Recurse -ErrorAction SilentlyContinue |
  Where-Object { Test-Path -LiteralPath (Join-Path $_.Directory.FullName 'genie_config.json') } |
  Select-Object -ExpandProperty DirectoryName -Unique
$candidates
```

Proceed only when exactly one candidate is the intended Qwen3-4B W4A16 X Elite
bundle. Set its directory and derive the manifest checksum:

```powershell
$env:GENIE_DIR = $candidates | Select-Object -First 1
$env:QWEN3_4B_GENIE_SHA256_TREE = & .\.venv\Scripts\python.exe scripts\hash_artifact.py $env:GENIE_DIR
```

Run the local physical proof:

```powershell
.\.venv\Scripts\python.exe scripts\probe_hardware.py `
  --device-id pc-01 `
  --model-id qwen3-4b-genie `
  --compatibility-key windows-arm64-x1e-v73-qairt-2.48 `
  --runtime-name genie `
  --runtime-version QAIRT-2.48 `
  --accelerator-available `
  --execute `
  --output "$env:TEMP\dragonnest-xelite-proof.json"
```

Inspect the proof. `artifact_validation.passed`, `execution.success`, the host
ARM64/X Elite inventory, runtime/accelerator fields, and nonempty output hash
must all agree. Do not change a failure into a success manually.

Update `coordination/xelite/STATUS.json` with sanitized values. Calculate the
proof hash without moving the proof into Git:

```powershell
$proofHash = (Get-FileHash -Algorithm SHA256 "$env:TEMP\dragonnest-xelite-proof.json").Hash.ToLowerInvariant()
```

Then commit and push only the response file:

```powershell
git add coordination/xelite/STATUS.json
git commit -m "Record physical X Elite runtime proof status"
git push -u origin codex/xelite-response
```

If blocked, still update and push `STATUS.json` with `state: "blocked"`, one
precise blocker, commands attempted, and the safest next command. Do not install
or download a different proprietary model automatically.

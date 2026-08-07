# Copies cached X Elite-target artifacts into a staging directory the
# X Elite worker launcher can point at. Does not compile or download from
# AI Hub; the cache must already be populated (see docs/results/
# demo_artifact_inventory.json for what should be present and its checksums).
#
# Usage: .\scripts\artifact_tools\stage_xelite_artifacts.ps1 -CacheRoot C:\DragonNestArtifacts -StageDir .\xelite-demo-bundle

param(
    [Parameter(Mandatory = $true)][string]$CacheRoot,
    [Parameter(Mandatory = $true)][string]$StageDir
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

$items = @(
    @{ Src = Join-Path $CacheRoot "qwen3-4b\xelite\genie"; Dst = Join-Path $StageDir "qwen3-4b-genie" },
    @{ Src = Join-Path $CacheRoot "qwen3-1.7b\xelite\stage-0"; Dst = Join-Path $StageDir "qwen3-1.7b-s0" },
    @{ Src = Join-Path $CacheRoot "qwen3-1.7b\xelite\stage-1"; Dst = Join-Path $StageDir "qwen3-1.7b-s1" },
    @{ Src = Join-Path $CacheRoot "qwen3-1.7b\xelite\stage-2"; Dst = Join-Path $StageDir "qwen3-1.7b-s2" },
    @{ Src = Join-Path $CacheRoot "qwen3-1.7b\xelite\stage-3"; Dst = Join-Path $StageDir "qwen3-1.7b-s3" }
)

foreach ($item in $items) {
    if (-not (Test-Path $item.Src)) {
        Write-Warning "Missing from cache, skipping: $($item.Src)"
        continue
    }
    Copy-Item -Recurse -Force -Path $item.Src -Destination $item.Dst
    Write-Host "Staged $($item.Src) -> $($item.Dst)"
}

Write-Host "Done. Run scripts\artifact_tools\verify_demo_cache.py against $StageDir before demo day."

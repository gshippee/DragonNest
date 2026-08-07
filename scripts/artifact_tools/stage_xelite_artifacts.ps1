# Verify and stage the four external X Elite QNN context binaries.
# This does not download, compile, or claim physical QAIRT 2.48 compatibility.
param(
    [Parameter(Mandatory = $true)][string]$CacheRoot,
    [Parameter(Mandatory = $true)][string]$StageDir
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$inventoryPath = Join-Path $repoRoot "docs\results\demo_artifact_inventory.json"
$inventory = (Get-Content -Raw -LiteralPath $inventoryPath | ConvertFrom-Json).artifacts
$cache = (Resolve-Path -LiteralPath $CacheRoot).Path
$stage = [System.IO.Path]::GetFullPath($StageDir)
New-Item -ItemType Directory -Force -Path $stage | Out-Null

for ($index = 0; $index -lt 4; $index++) {
    $logicalId = "qwen3-1.7b-s$index-xelite"
    $record = $inventory | Where-Object { $_.logical_artifact_id -eq $logicalId }
    if ($null -eq $record) {
        throw "Inventory does not contain $logicalId"
    }
    $sourceDir = Join-Path $cache "qwen3-1.7b\xelite\stage-$index"
    $candidates = @(Get-ChildItem -LiteralPath $sourceDir -Recurse -File -Filter *.bin)
    if ($candidates.Count -ne 1) {
        throw "Expected exactly one .bin under $sourceDir; found $($candidates.Count)"
    }
    $source = $candidates[0]
    if ($source.Length -ne [int64]$record.size_bytes) {
        throw "$logicalId byte-size mismatch"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $source.FullName).Hash.ToLowerInvariant()
    if ($actual -ne $record.sha256) {
        throw "$logicalId SHA-256 mismatch"
    }
    $destination = Join-Path $stage "$logicalId.bin"
    if (Test-Path -LiteralPath $destination) {
        throw "Refusing to overwrite existing staged artifact: $destination"
    }
    Copy-Item -LiteralPath $source.FullName -Destination $destination
    $staged = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
    if ($staged -ne $record.sha256) {
        throw "$logicalId staged-copy SHA-256 mismatch"
    }
    Write-Host "Verified and staged $logicalId -> $destination"
    Write-Host "  `$env:QWEN3_1_7B_S${index}_XELITE_QNN='$destination'"
}

Write-Host "The contexts are QAIRT 2.45 artifacts. Validate loading with the laptop's"
Write-Host "QAIRT 2.48 runtime before advertising them as physically executable."

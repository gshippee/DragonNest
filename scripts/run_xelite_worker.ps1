<#
.SYNOPSIS
    One-command launcher for a DragonNest Snapdragon X Elite gRPC Device Agent
    running the real Qwen3-4B Genie/HTP runtime.

.DESCRIPTION
    Discovers (or accepts) a Qwen3-4B Genie bundle directory, validates it
    against DragonNest's existing artifact contract (path + tree checksum),
    and starts scripts/run_agent.py with the real HardwareRuntimeAdapter so
    this machine becomes a physical compute worker for a DragonNest Brain.

    This script never talks to the OpenAI-compatible HTTP endpoint adapter —
    it only drives the gRPC Device Agent path.

.PARAMETER Brain
    Brain gRPC address, e.g. 192.168.137.1:50051. Required.

.PARAMETER GenieDir
    Path to a Qwen3-4B Genie bundle directory (must contain genie-t2t-run.exe
    and genie_config.json). If omitted, this script searches under the current
    user's home directory for exactly one candidate and refuses to proceed if
    zero or more than one is found.

.PARAMETER DeviceId
    Device id to advertise to the Brain. Default: pc-01.

.PARAMETER EnrollmentToken
    Brain enrollment token. Defaults to $env:DRAGONNEST_ENROLLMENT_TOKEN, and
    falls back to the DragonNest dev default with a warning if unset.

.EXAMPLE
    .\scripts\run_xelite_worker.ps1 -Brain 192.168.137.1:50051
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Brain,

    [string]$GenieDir,

    [string]$DeviceId = "pc-01",

    [string]$CompatibilityKey = "windows-arm64-x1e-v73-qairt-2.48",

    [string]$RuntimeVersion = "QAIRT-2.48",

    [string]$EnrollmentToken
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Fail($message) {
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 1
}

# --- Discover the bundle if not given -------------------------------------

if (-not $GenieDir) {
    $candidates = @(Get-ChildItem -Path $HOME -Filter "genie-t2t-run.exe" -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.Directory.FullName "genie_config.json") } |
        Select-Object -ExpandProperty DirectoryName -Unique)

    if (-not $candidates -or $candidates.Count -eq 0) {
        Fail "No Qwen3-4B Genie bundle found under $HOME (looked for a directory containing both genie-t2t-run.exe and genie_config.json). Pass -GenieDir explicitly."
    }
    if ($candidates.Count -gt 1) {
        Write-Host "Multiple candidate bundles found; refusing to guess:" -ForegroundColor Yellow
        $candidates | ForEach-Object { Write-Host "  $_" }
        Fail "Ambiguous bundle candidates. Re-run with -GenieDir pointing at the correct one."
    }
    $GenieDir = $candidates[0]
}

if (-not (Test-Path (Join-Path $GenieDir "genie-t2t-run.exe"))) {
    Fail "genie-t2t-run.exe not found in $GenieDir"
}
if (-not (Test-Path (Join-Path $GenieDir "genie_config.json"))) {
    Fail "genie_config.json not found in $GenieDir"
}

# --- Resolve the venv python -------------------------------------------

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Fail ".venv not found. Run: py -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e `".[dev]`""
}

# --- Checksum + artifact validation ----------------------------------------

Write-Host "Hashing artifact tree..."
$checksum = & $Python "scripts\hash_artifact.py" $GenieDir
if ($LASTEXITCODE -ne 0 -or -not $checksum) {
    Fail "Failed to compute artifact checksum for $GenieDir"
}

$env:GENIE_DIR = $GenieDir
$env:QWEN3_4B_GENIE_SHA256_TREE = $checksum

$validation = & $Python "scripts\check_artifacts.py" 2>&1
$genieLine = $validation | Where-Object { $_ -match "qwen3-4b-genie" }
if (-not ($genieLine -match "^READY")) {
    Write-Host ($genieLine -join "`n") -ForegroundColor Red
    Fail "Artifact validation failed for qwen3-4b-genie. Not starting the worker."
}

# --- Enrollment token --------------------------------------------------

if (-not $EnrollmentToken) {
    if ($env:DRAGONNEST_ENROLLMENT_TOKEN) {
        $EnrollmentToken = $env:DRAGONNEST_ENROLLMENT_TOKEN
    } else {
        Write-Host "WARNING: no -EnrollmentToken or `$env:DRAGONNEST_ENROLLMENT_TOKEN set; using the DragonNest dev default. Set one for anything beyond a trusted local demo network." -ForegroundColor Yellow
        $EnrollmentToken = "dev-token"
    }
}

# --- Report state before connecting -----------------------------------

$system = Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty Model
$soc = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
$bundleName = Split-Path -Leaf $GenieDir

Write-Host ""
Write-Host "DragonNest X Elite Worker" -ForegroundColor Cyan
Write-Host "Host:            $system"
Write-Host "SoC:             $soc"
Write-Host "Artifact:        qwen3-4b-genie ($bundleName)"
Write-Host "Runtime:         Genie / $RuntimeVersion"
Write-Host "Execution:       HTP"
Write-Host "Steering:        none"
Write-Host "Artifact state:  installed / cold"
Write-Host "Brain:           $Brain"
Write-Host "Connecting..."
Write-Host ""

& $Python "scripts\run_agent.py" `
    --device-id $DeviceId `
    --brain $Brain `
    --enrollment-token $EnrollmentToken `
    --fabric "configs\hardware-fabric.yaml" `
    --artifact-manifest "configs\model-artifacts.yaml" `
    --compatibility-key $CompatibilityKey `
    --runtime-name genie `
    --runtime-version $RuntimeVersion `
    --accelerator-available

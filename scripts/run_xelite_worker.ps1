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

.PARAMETER ExpectedChecksum
    Expected sha256-tree checksum for the physically verified Qwen3-4B bundle.
    The default pins the exact bundle recorded in
    docs/results/xelite_worker_status.md so auto-discovery cannot silently
    select a different Genie model.

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

    [string]$EnrollmentToken,

    [string]$ExpectedChecksum = "sha256-tree:940ab2c9958a4f0a53b6964fa96fc427f1f4d33dd1046584e040ec6f2298c929",

    [switch]$EnableQwen17Pipeline,

    [string]$Qairt245Root = $env:QAIRT_ROOT,

    [string]$Qwen17Tokenizer = $env:DRAGONNEST_QWEN17_TOKENIZER
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
if ($checksum -ne $ExpectedChecksum) {
    Fail "Bundle checksum does not match the physically verified qwen3-4b-genie artifact. Expected $ExpectedChecksum but found $checksum. Refusing to advertise the wrong model."
}

$env:GENIE_DIR = $GenieDir
$env:QWEN3_4B_GENIE_SHA256_TREE = $checksum

if ($EnableQwen17Pipeline) {
    if (-not $Qairt245Root -or -not (Test-Path -LiteralPath $Qairt245Root -PathType Container)) {
        Fail "Qwen3-1.7B requires -Qairt245Root (or `$env:QAIRT_ROOT) pointing at the physically verified QAIRT 2.45 install."
    }
    $env:QAIRT_ROOT = (Resolve-Path -LiteralPath $Qairt245Root).Path
    $contextUtility = Join-Path $env:QAIRT_ROOT "bin\aarch64-windows-msvc\qnn-context-binary-utility.exe"
    $netRun = Join-Path $env:QAIRT_ROOT "bin\aarch64-windows-msvc\qnn-net-run.exe"
    if (-not (Test-Path -LiteralPath $contextUtility) -or -not (Test-Path -LiteralPath $netRun)) {
        Fail "QAIRT 2.45 root is missing qnn-context-binary-utility.exe or qnn-net-run.exe: $env:QAIRT_ROOT"
    }
    foreach ($index in 0..3) {
        $name = "QWEN3_1_7B_S${index}_XELITE_QNN"
        $path = [Environment]::GetEnvironmentVariable($name)
        if (-not $path -or -not (Test-Path -LiteralPath $path -PathType Leaf)) {
            Fail "$name must point at the checksummed S$index context binary."
        }
    }
    & $Python -c "import torch, transformers, sentencepiece" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Fail "Qwen3-1.7B Python dependencies are missing. Run: .\.venv\Scripts\python.exe -m pip install -e `".[dev,xelite]`""
    }
    if (-not $Qwen17Tokenizer) {
        $Qwen17Tokenizer = "Qwen/Qwen3-1.7B"
    }
    $env:DRAGONNEST_QWEN17_TOKENIZER = $Qwen17Tokenizer
    & $Python -c "from transformers import AutoConfig, AutoTokenizer; import sys; AutoConfig.from_pretrained(sys.argv[1]); AutoTokenizer.from_pretrained(sys.argv[1], is_fast=False)" $Qwen17Tokenizer
    if ($LASTEXITCODE -ne 0) {
        Fail "Qwen3-1.7B tokenizer/config is unavailable. Connect once to fetch Qwen/Qwen3-1.7B or pass -Qwen17Tokenizer <local-directory>."
    }
}

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
if ($EnableQwen17Pipeline) {
    Write-Host "Elastic stages:  qwen3-1.7b S0-S3 / QNN / QAIRT-2.45 / HTP"
    Write-Host "Elastic state:   production provider enabled; physical Agent acceptance pending"
}
Write-Host "Brain:           $Brain"
Write-Host "Connecting..."
Write-Host ""

$agentArguments = @(
    "scripts\run_agent.py",
    "--device-id", $DeviceId,
    "--brain", $Brain,
    "--enrollment-token", $EnrollmentToken,
    "--fabric", "configs\hardware-fabric.yaml",
    "--artifact-manifest", "configs\model-artifacts.yaml",
    "--compatibility-key", $CompatibilityKey,
    "--runtime-name", "genie",
    "--runtime-version", $RuntimeVersion,
    "--accelerator-available"
)
if ($EnableQwen17Pipeline) {
    $agentArguments += @(
        "--compatible-target-class", "windows-arm64-x1e-v73-qairt-2.45",
        "--enable-qwen17-pipeline"
    )
}
& $Python @agentArguments

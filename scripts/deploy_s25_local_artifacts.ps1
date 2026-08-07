[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CacheRoot,
    [string]$AdbPath = "adb",
    [string]$Serial = "",
    [string]$PythonPath = "",
    [string]$Package = "com.dragonnest.agent",
    # Default is every profile (Base, Concise, Detailed). Pass a subset (e.g.
    # -Profiles Base) to verify/provision only the profiles whose bundles
    # are actually present -- a device should advertise what it was
    # actually given, not what the full catalog wishes were installed.
    [ValidateSet("Base", "Concise", "Detailed")]
    [string[]]$Profiles = @("Base", "Concise", "Detailed")
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$inventory = Join-Path $repoRoot "docs\results\s25_geniex_artifacts.json"
$stager = Join-Path $PSScriptRoot "artifact_tools\stage_s25_geniex_artifacts.py"
$profileToModelId = @{
    Base     = "qwen3-0.6b-s25-base"
    Concise  = "qwen3-0.6b-s25-concise"
    Detailed = "qwen3-0.6b-s25-detailed"
}

$deviceLines = @(& $AdbPath devices 2>&1)
if ($LASTEXITCODE -ne 0) { throw "adb devices failed" }
$connected = @($deviceLines | Where-Object { $_ -match "^([^\s]+)\s+device$" } | ForEach-Object { $Matches[1] })
if ($Serial) {
    if ($connected -notcontains $Serial) { throw "Requested serial '$Serial' is not connected." }
    $selected = $Serial
} elseif ($connected.Count -eq 1) {
    $selected = $connected[0]
} else {
    throw "Expected exactly one authorized Android device, found $($connected.Count)."
}

$model = (& $AdbPath -s $selected shell getprop ro.product.model).Trim()
$soc = (& $AdbPath -s $selected shell getprop ro.soc.model).Trim()
if ($model -notmatch "SM-S938" -or $soc -notmatch "SM8750") {
    throw "Expected the physical Galaxy S25 Ultra / SM8750, found '$model' / '$soc'."
}
& $AdbPath -s $selected shell run-as $Package id | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "$Package is absent or not debuggable. Install the DragonNest hardware debug APK first."
}

if (-not $PythonPath) {
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $PythonPath = if (Test-Path -LiteralPath $venvPython) {
        $venvPython
    } else {
        (Get-Command python -ErrorAction Stop).Source
    }
}
$resolvedCache = (Resolve-Path -LiteralPath $CacheRoot).Path

Write-Host "Provisioning only $Package on $model ($selected)." -ForegroundColor Cyan
Write-Host "The historical steering-demo package is not read or modified by this script." -ForegroundColor DarkGray

$stagerArgs = @(
    "--cache-root", $resolvedCache,
    "--inventory", $inventory,
    "--adb", $AdbPath,
    "--serial", $selected,
    "--package", $Package
)
$allProfiles = @("Base", "Concise", "Detailed")
$isDefaultProfileSet = ($Profiles.Count -eq $allProfiles.Count) -and (@(Compare-Object $Profiles $allProfiles).Count -eq 0)
if (-not $isDefaultProfileSet) {
    Write-Host "Restricting to profile(s): $($Profiles -join ', ')" -ForegroundColor Yellow
    foreach ($profileName in $Profiles) {
        $stagerArgs += @("--model-id", $profileToModelId[$profileName])
    }
}

& $PythonPath $stager @stagerArgs
if ($LASTEXITCODE -ne 0) { throw "S25 GenieX provisioning failed with exit code $LASTEXITCODE." }

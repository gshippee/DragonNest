[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CacheRoot,
    [string]$AdbPath = "adb",
    [string]$Serial = "",
    [string]$PythonPath = "",
    [string]$Package = "com.dragonnest.agent",
    [string]$ExpectedModelPattern = "Galaxy S25|SM-S938"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$inventoryPath = Join-Path $repoRoot "docs\results\demo_artifact_inventory.json"
$stagerPath = Join-Path $PSScriptRoot "artifact_tools\stage_android_artifacts.py"
$resolvedCache = (Resolve-Path -LiteralPath $CacheRoot).Path

if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) {
    throw "Artifact inventory is missing: $inventoryPath"
}
if (-not (Test-Path -LiteralPath $stagerPath -PathType Leaf)) {
    throw "Android staging helper is missing: $stagerPath"
}

$deviceLines = @(& $AdbPath devices 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "adb devices failed. Confirm Android platform-tools is installed and on PATH."
}
$connected = @($deviceLines | Where-Object { $_ -match "^([^\s]+)\s+device$" } | ForEach-Object { $Matches[1] })
$unready = @($deviceLines | Where-Object { $_ -match "^([^\s]+)\s+(unauthorized|offline)$" })
if ($unready.Count -gt 0) {
    throw "ADB reports an unauthorized/offline device: $($unready -join '; '). Unlock the phone and accept USB debugging."
}
if ($Serial) {
    if ($connected -notcontains $Serial) {
        throw "Requested serial '$Serial' is not the one connected ADB device. Connected: $($connected -join ', ')"
    }
    $selectedSerial = $Serial
} elseif ($connected.Count -eq 1) {
    $selectedSerial = $connected[0]
} else {
    throw "Expected exactly one authorized Android device, found $($connected.Count): $($connected -join ', ')"
}

function Invoke-SelectedAdb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $output = @(& $AdbPath -s $selectedSerial @Arguments 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "adb $($Arguments -join ' ') failed: $($output -join [Environment]::NewLine)"
    }
    return $output
}

$model = ((Invoke-SelectedAdb shell getprop ro.product.model) -join " ").Trim()
$deviceCode = ((Invoke-SelectedAdb shell getprop ro.product.device) -join " ").Trim()
if ("$model $deviceCode" -notmatch $ExpectedModelPattern) {
    throw "Connected Android device '$model' ($deviceCode) does not match expected S25 pattern '$ExpectedModelPattern'."
}

$runAs = @(Invoke-SelectedAdb shell run-as $Package id)
if (($runAs -join " ") -notmatch "uid=") {
    throw "$Package is not installed as a debuggable application; run-as did not return an app uid."
}
$packageDump = @(Invoke-SelectedAdb shell dumpsys package $Package)
if (($packageDump -join "`n") -notmatch "DEBUGGABLE") {
    throw "$Package is installed, but Android did not report the DEBUGGABLE flag. Install the thin debug APK."
}

if (-not $PythonPath) {
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $PythonPath = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { (Get-Command python -ErrorAction Stop).Source }
}

$inventory = Get-Content -Raw -LiteralPath $inventoryPath | ConvertFrom-Json
$requiredIds = 0..3 | ForEach-Object { "qwen3-1.7b-s$_-s25" }
foreach ($artifactId in $requiredIds) {
    $record = $inventory.artifacts | Where-Object logical_artifact_id -eq $artifactId
    if (-not $record -or $record.status -ne "READY" -or -not $record.sha256) {
        throw "$artifactId does not have a checksummed READY inventory record."
    }
}

Write-Host "S25 preflight passed: $model ($selectedSerial)" -ForegroundColor Green
Write-Host "The Python helper will verify all four source hashes before the first phone mutation."
& $PythonPath $stagerPath $resolvedCache --adb $AdbPath --serial $selectedSerial --package $Package
if ($LASTEXITCODE -ne 0) {
    throw "S25 artifact staging failed with exit code $LASTEXITCODE."
}

foreach ($optionalId in @("qwen3-0.6b-s25-base", "qwen3-0.6b-s25-concise")) {
    $record = $inventory.artifacts | Where-Object logical_artifact_id -eq $optionalId
    $matches = @(Get-ChildItem -LiteralPath $resolvedCache -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match [regex]::Escape($optionalId) })
    if ($matches.Count -eq 0) {
        Write-Host "OPTIONAL SKIP: $optionalId is not present in this cache."
    } elseif (-not $record.sha256) {
        Write-Warning "$optionalId bytes were found, but the committed inventory has no authoritative checksum; refusing to stage them."
    } else {
        Write-Host "OPTIONAL DETECTED: $optionalId has checksummed bytes, but optional 0.6B installation remains separate from the Act 3 stage manifest."
    }
}

Write-Host "ARTIFACTS INSTALLED" -ForegroundColor Green
Write-Host "RUNTIME NOT YET EXECUTABLE" -ForegroundColor Yellow
Write-Host "PersonaCare was stopped. Relaunch it to reload the app-private artifact catalog."

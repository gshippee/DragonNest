[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CacheRoot,
    [string]$AdbPath = "adb",
    [string]$Serial = "",
    [string]$PythonPath = "",
    [string]$Package = "com.dragonnest.agent"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$inventory = Join-Path $repoRoot "docs\results\s25_geniex_artifacts.json"
$stager = Join-Path $PSScriptRoot "artifact_tools\stage_s25_geniex_artifacts.py"

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
& $PythonPath $stager `
    --cache-root $resolvedCache `
    --inventory $inventory `
    --adb $AdbPath `
    --serial $selected `
    --package $Package
if ($LASTEXITCODE -ne 0) { throw "S25 GenieX provisioning failed with exit code $LASTEXITCODE." }

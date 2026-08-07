[CmdletBinding()]
param(
    [string]$AdbPath = "adb",
    [string]$Package = "",
    [string]$OutputRoot = "C:\DragonNestRecovered\S25SteeringDemo",
    [string]$ExpectedModelPattern = "Galaxy S25|SM-S938"
)

$ErrorActionPreference = "Stop"

$deviceLines = @(& $AdbPath devices -l 2>&1)
if ($LASTEXITCODE -ne 0) { throw "adb devices failed." }
$connected = @($deviceLines | Where-Object { $_ -match "^([^\s]+)\s+device\b" } | ForEach-Object { $Matches[1] })
if ($connected.Count -ne 1) {
    throw "Expected exactly one authorized Android device, found $($connected.Count)."
}
$serial = $connected[0]

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $output = @(& $AdbPath -s $serial @Arguments 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "adb $($Arguments -join ' ') failed: $($output -join [Environment]::NewLine)"
    }
    return $output
}

$model = ((Invoke-Adb shell getprop ro.product.model) -join " ").Trim()
$deviceCode = ((Invoke-Adb shell getprop ro.product.device) -join " ").Trim()
if ("$model $deviceCode" -notmatch $ExpectedModelPattern) {
    throw "Connected device '$model' ($deviceCode) is not the expected S25."
}

if (-not $Package) {
    $activityDump = (Invoke-Adb shell dumpsys activity activities) -join "`n"
    $match = [regex]::Match($activityDump, "topResumedActivity=.*?\s([A-Za-z0-9_.]+)/")
    if (-not $match.Success) {
        throw "Could not identify the resumed package. Open the working steering demo and retry."
    }
    $Package = $match.Groups[1].Value
}
$thirdParty = (Invoke-Adb shell pm list packages -3) -join "`n"
if ($thirdParty -notmatch "(?m)^package:$([regex]::Escape($Package))$") {
    throw "Foreground package '$Package' is not an installed third-party package."
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$destination = Join-Path $OutputRoot "$Package-$timestamp"
$apkDirectory = Join-Path $destination "installed-apk"
$dataDirectory = Join-Path $destination "private-data"
New-Item -ItemType Directory -Force -Path $apkDirectory,$dataDirectory | Out-Null

$metadata = @(
    "adb=$($deviceLines -join ' ')"
    "manufacturer=$(((Invoke-Adb shell getprop ro.product.manufacturer) -join ' ').Trim())"
    "model=$model"
    "device=$deviceCode"
    "fingerprint=$(((Invoke-Adb shell getprop ro.build.fingerprint) -join ' ').Trim())"
    "security_patch=$(((Invoke-Adb shell getprop ro.build.version.security_patch) -join ' ').Trim())"
    "package=$Package"
    ((Invoke-Adb shell dumpsys package $Package) | Select-String -Pattern "versionCode=|versionName=|firstInstallTime=|lastUpdateTime=|DEBUGGABLE" | ForEach-Object Line)
)
$metadata | Set-Content -LiteralPath (Join-Path $destination "device-metadata.txt") -Encoding UTF8

$apkPaths = @(Invoke-Adb shell pm path $Package) | ForEach-Object { $_ -replace "^package:", "" }
if ($apkPaths.Count -eq 0) { throw "Package manager returned no APK paths for $Package." }
for ($index = 0; $index -lt $apkPaths.Count; $index++) {
    $name = if ($index -eq 0) { "base.apk" } else { "split-$index.apk" }
    Invoke-Adb pull $apkPaths[$index] (Join-Path $apkDirectory $name) | Out-Null
}

$runAs = @(& $AdbPath -s $serial shell run-as $Package id 2>&1)
if ($LASTEXITCODE -eq 0 -and ($runAs -join " ") -match "uid=") {
    foreach ($directoryName in @("files", "no_backup", "shared_prefs", "databases")) {
        & $AdbPath -s $serial shell run-as $Package sh -c "test -d $directoryName" 2>$null
        if ($LASTEXITCODE -ne 0) { continue }
        $archive = Join-Path $dataDirectory "$directoryName.tar"
        $command = '"' + $AdbPath + '" -s ' + $serial + ' exec-out run-as ' + $Package +
            ' tar -cf - ' + $directoryName + ' > "' + $archive + '"'
        cmd.exe /d /c $command
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $archive)) {
            throw "Failed to preserve app-private $directoryName."
        }
        tar -tf $archive | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Recovered archive is invalid: $archive" }
    }
} else {
    "run-as unavailable; Android sandbox was not bypassed." |
        Set-Content -LiteralPath (Join-Path $dataDirectory "RUN_AS_UNAVAILABLE.txt")
}

$hashes = Get-ChildItem -LiteralPath $destination -Recurse -File | ForEach-Object {
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    [pscustomobject]@{ sha256 = $hash.Hash.ToLowerInvariant(); size_bytes = $_.Length; path = $_.FullName.Substring($destination.Length + 1) }
}
$hashes | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $destination "sha256.json") -Encoding UTF8
Write-Host "READ-ONLY RECOVERY COMPLETE: $destination" -ForegroundColor Green
Write-Host "No package or app data was installed, cleared, or modified."


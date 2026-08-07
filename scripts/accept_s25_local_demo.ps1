[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CacheRoot,
    [Parameter(Mandatory = $true)]
    [string]$BrainHost,
    [int]$BrainPort = 50051,
    [string]$EnrollmentToken = "",
    [string]$ApkPath = "",
    [string]$AdbPath = "adb",
    [string]$Serial = "",
    [string]$Package = "com.dragonnest.agent",
    [string]$PythonPath = "",
    [int]$AdmissionTimeoutSeconds = 90
)

# One-command physical acceptance for "Compute = Local" on PersonaCare against
# a connected Galaxy S25 Ultra. Fails loudly and sequentially through:
#   A. ADB device identity
#   B. hardware APK install
#   C. Base/Concise/Detailed artifact provisioning (checksum-verified)
#   D. runtime/admission verification (real GenieX/HTP, no mock)
#   E. Brain enrollment status (stops for the one step that requires the
#      Compose UI: scanning/entering the enrollment code)
#
# This script does not drive the Compose UI and does not invent a second
# enrollment protocol. Enrollment credentials are Android-Keystore-encrypted
# on device (EnrollmentStore/UserProfileStore) and can only be written by the
# app itself via the Connect screen or a scanned QR code.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$androidDir = Join-Path $repoRoot "android-agent"

function Fail {
    param([string]$Category, [string]$Message)
    Write-Host ""
    Write-Host "[$Category] $Message" -ForegroundColor Red
    throw "$Category`: $Message"
}

function Step {
    param([string]$Id, [string]$Title)
    Write-Host ""
    Write-Host "== $Id. $Title ==" -ForegroundColor Cyan
}

function Format-NativeArg {
    param([string]$Value)
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '""') + '"' }
    return $Value
}

function Invoke-Native {
    # Routes through cmd.exe so PowerShell 5.1 never wraps merged native
    # stderr lines as terminating NativeCommandErrors under
    # $ErrorActionPreference = "Stop" (stderr is otherwise already captured
    # for you and must not be redirected directly in-process).
    param([string]$Exe, [string[]]$Arguments)
    $commandLine = (Format-NativeArg $Exe) + " " + (($Arguments | ForEach-Object { Format-NativeArg $_ }) -join " ")
    $output = @(cmd.exe /d /c "$commandLine 2>&1")
    return [pscustomobject]@{ Output = $output; ExitCode = $LASTEXITCODE }
}

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $result = Invoke-Native -Exe $AdbPath -Arguments (@("-s", $script:selected) + $Arguments)
    if ($result.ExitCode -ne 0) {
        throw "adb $($Arguments -join ' ') failed: $($result.Output -join [Environment]::NewLine)"
    }
    return $result.Output
}

function Invoke-RunAs {
    param([string]$Command)
    $remote = "run-as $Package sh -c " + "'" + ($Command -replace "'", "'\\''") + "'"
    return Invoke-Adb shell $remote
}

# ---------------------------------------------------------------------------
Step "A" "ADB device identity"
$devicesResult = Invoke-Native -Exe $AdbPath -Arguments @("devices")
$deviceLines = $devicesResult.Output
if ($devicesResult.ExitCode -ne 0) { Fail "BRAIN_UNREACHABLE" "adb devices failed to run. Is adb on PATH / is a device authorized?" }
$connected = @($deviceLines | Where-Object { $_ -match "^([^\s]+)\s+device$" } | ForEach-Object { $Matches[1] })
if ($Serial) {
    if ($connected -notcontains $Serial) { Fail "BRAIN_UNREACHABLE" "Requested serial '$Serial' is not an authorized connected device." }
    $selected = $Serial
} elseif ($connected.Count -eq 1) {
    $selected = $connected[0]
} elseif ($connected.Count -eq 0) {
    Fail "BRAIN_UNREACHABLE" "No authorized Android device is connected. Plug in the S25 and accept the USB debugging prompt."
} else {
    Fail "BRAIN_UNREACHABLE" "Expected exactly one authorized Android device, found $($connected.Count): $($connected -join ', '). Pass -Serial."
}

$model = (Invoke-Adb shell getprop ro.product.model) -join " " | ForEach-Object { $_.Trim() }
$soc = (Invoke-Adb shell getprop ro.soc.model) -join " " | ForEach-Object { $_.Trim() }
if ($model -notmatch "SM-S938" -or $soc -notmatch "SM8750") {
    Fail "BRAIN_UNREACHABLE" "Connected device '$model' / '$soc' is not the physical Galaxy S25 Ultra / SM8750."
}
Write-Host "  Device: $model / $soc ($selected)"

# ---------------------------------------------------------------------------
Step "B" "Hardware APK install"
if (-not $ApkPath) {
    $ApkPath = Join-Path $androidDir "app\build\outputs\apk\debug\app-debug.apk"
}
if (-not (Test-Path -LiteralPath $ApkPath)) {
    Fail "APK_NOT_HARDWARE_BUILD" "No APK at '$ApkPath'. Run scripts\build_s25_local_demo.ps1 first."
}
$resolvedApkPath = (Resolve-Path -LiteralPath $ApkPath).Path
Write-Host "  Installing $resolvedApkPath (adb install -r; app-private storage is preserved)"
$installResult = Invoke-Native -Exe $AdbPath -Arguments @("-s", $selected, "install", "-r", $resolvedApkPath)
if ($installResult.ExitCode -ne 0 -or ($installResult.Output -join "`n") -notmatch "Success") {
    Fail "APK_NOT_HARDWARE_BUILD" "adb install -r failed: $($installResult.Output -join ' ')"
}
$runAsProbe = Invoke-Native -Exe $AdbPath -Arguments @("-s", $selected, "shell", "run-as", $Package, "id")
if ($runAsProbe.ExitCode -ne 0) {
    Fail "APK_NOT_HARDWARE_BUILD" "$Package is not debuggable/run-as-able after install. Confirm the APK came from build_s25_local_demo.ps1 (debug hardware build)."
}
Write-Host "  $Package installed and debuggable."

# ---------------------------------------------------------------------------
Step "C" "Provisioning Base/Concise/Detailed artifacts"
$deployScript = Join-Path $PSScriptRoot "deploy_s25_local_artifacts.ps1"
try {
    $deployArgs = @{
        CacheRoot = $CacheRoot
        AdbPath   = $AdbPath
        Serial    = $selected
        Package   = $Package
    }
    if ($PythonPath) { $deployArgs["PythonPath"] = $PythonPath }
    & $deployScript @deployArgs
    if ($LASTEXITCODE -ne 0) { throw "deploy_s25_local_artifacts.ps1 exited with code $LASTEXITCODE" }
} catch {
    $message = $_.Exception.Message
    if ($message -match "checksum mismatch") {
        Fail "ARTIFACT_CHECKSUM_FAILED" $message
    } else {
        Fail "ARTIFACT_MISSING" $message
    }
}
Write-Host "  Base/Concise/Detailed artifacts installed and checksum-verified; app restarted."

# ---------------------------------------------------------------------------
Step "D" "Runtime/admission verification (GenieX/HTP, no mock)"
Write-Host "  Clearing logcat and restarting $Package for a clean admission trace..."
& $AdbPath -s $selected logcat -c
Invoke-Adb shell am force-stop $Package | Out-Null
Invoke-Adb shell monkey -p $Package 1 | Out-Null

$expectedModelIds = @("qwen3-0.6b-s25-base", "qwen3-0.6b-s25-concise", "qwen3-0.6b-s25-detailed")
$deadline = (Get-Date).AddSeconds($AdmissionTimeoutSeconds)
$registrationSeen = $false
$rejectionMessage = $null
$runtimeLog = ""
do {
    Start-Sleep -Seconds 3
    $runtimeLog = (Invoke-Adb logcat -d -s "DragonNestRuntime:*" "DragonNestGenieX:*") -join "`n"
    $debugLogRaw = (Invoke-RunAs "cat shared_prefs/client-debug.xml 2>/dev/null") -join "`n"
    if ($debugLogRaw -match "Brain returned RegistrationAccepted") { $registrationSeen = $true; break }
    if ($debugLogRaw -match "Brain rejected registration:\s*([^<]*)") { $rejectionMessage = $Matches[1]; break }
} while ((Get-Date) -lt $deadline)

if ($runtimeLog -match "GenieX initialization failed") {
    Fail "GENIEX_INIT_FAILED" "logcat shows a GenieX initialization failure:`n$runtimeLog"
}
$missingModels = @()
foreach ($modelId in $expectedModelIds) {
    if ($runtimeLog -match [regex]::Escape($modelId)) {
        # Any DragonNestRuntime line naming this model is a rejection
        # (Ignoring model with missing/invalid checksum, target-incompatible,
        # or no available bridge) -- admitted models are never logged.
        $missingModels += $modelId
    }
}
if ($missingModels.Count -gt 0) {
    Fail "ARTIFACT_NOT_ADVERTISED" "DragonNestRuntime rejected: $($missingModels -join ', ').`n$runtimeLog"
}
if ($rejectionMessage) {
    Fail "ENROLLMENT_FAILED" "Brain rejected registration: $rejectionMessage"
}
if (-not $registrationSeen) {
    Write-Host "  No DragonNestRuntime rejections were logged for Base/Concise/Detailed, but registration was not confirmed within ${AdmissionTimeoutSeconds}s." -ForegroundColor Yellow
    Write-Host "  This is expected if the device is not yet enrolled with a Brain (see step E)." -ForegroundColor Yellow
} else {
    Write-Host "  Real GenieX/HTP runtime admitted Base/Concise/Detailed; Brain accepted registration." -ForegroundColor Green
}

Write-Host ""
Write-Host "  Recent on-device diagnostic log (shared_prefs/client-debug.xml):"
$debugLogRaw = (Invoke-RunAs "cat shared_prefs/client-debug.xml 2>/dev/null") -join "`n"
$eventsMatch = [regex]::Match($debugLogRaw, '<string name="events">(.*?)</string>', [System.Text.RegularExpressions.RegexOptions]::Singleline)
if ($eventsMatch.Success) {
    ($eventsMatch.Groups[1].Value -split "\\n" | Select-Object -First 10) | ForEach-Object { Write-Host "    $_" }
}

# ---------------------------------------------------------------------------
Step "E" "Brain enrollment"
$configRaw = (Invoke-RunAs "cat shared_prefs/agent-configuration.xml 2>/dev/null") -join "`n"
$currentHost = [regex]::Match($configRaw, '<string name="brain_host">([^<]*)</string>')
$hasCredential = (Invoke-RunAs "cat shared_prefs/enrollment.xml 2>/dev/null") -join "`n"
$hasProfile = (Invoke-RunAs "cat shared_prefs/user-profile.xml 2>/dev/null") -join "`n"
$enrolled = $hasCredential -match "ciphertext"
$profiled = $hasProfile -match "ciphertext"
$hostMatches = $currentHost.Success -and ($currentHost.Groups[1].Value -eq $BrainHost)

if ($registrationSeen -and $enrolled -and $profiled -and $hostMatches) {
    Write-Host ""
    Write-Host "READY FOR HUMAN LOCAL REQUEST" -ForegroundColor Green
    Write-Host "The device is enrolled with Brain $BrainHost`:$BrainPort and advertising Base/Concise/Detailed."
    Write-Host "Open PersonaCare, set Compute = Local, pick a profile, and submit a prompt."
    Write-Host ""
    Write-Host "Non-UI check (requires --persona-id support added to submit_task.py):"
    Write-Host "  python scripts\submit_task.py `"What is the capital of Japan?`" --brain ${BrainHost}:${BrainPort} --preferred-mode local --origin-device-id <device-id-from-app> --persona-id balanced"
    exit 0
}

Write-Host ""
Write-Host "READY FOR HUMAN ENROLLMENT STEP" -ForegroundColor Yellow
Write-Host "Automation stops here: enrollment credentials and the persona profile are"
Write-Host "encrypted with an on-device Android Keystore key (EnrollmentStore /"
Write-Host "UserProfileStore) and can only be written by the app itself. Complete this"
Write-Host "on the phone, then re-run this script to verify:"
Write-Host ""
if (-not $hostMatches) {
    Write-Host "  - Brain host mismatch: device has '$($currentHost.Groups[1].Value)', requested '$BrainHost'."
}
if (-not $enrolled) {
    Write-Host "  - No enrollment credential stored on device."
}
if (-not $profiled) {
    Write-Host "  - No persona profile stored on device."
}
Write-Host ""
Write-Host "  In PersonaCare, open Connect and either:"
Write-Host "    (a) scan the QR code from the Brain dashboard's enrollment session"
Write-Host "        (POST /api/enrollment-sessions on ${BrainHost}, dev_mode only), or"
Write-Host "    (b) manually enter Host=$BrainHost Port=$BrainPort and enrollment code:"
if ($EnrollmentToken) {
    Write-Host "        $EnrollmentToken"
} else {
    Write-Host "        <the code your Brain operator issued -- pass -EnrollmentToken to print it here>"
}
Write-Host "  Then set a persona (Balanced/Concise/Detailed) on first launch if prompted."
exit 2

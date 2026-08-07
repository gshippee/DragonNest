[CmdletBinding()]
param(
    [string]$JavaHome = $env:JAVA_HOME,
    [string]$AndroidSdkRoot = $env:ANDROID_SDK_ROOT,
    [string]$GradleUserHome = $env:GRADLE_USER_HOME,
    [switch]$SkipUnitTests
)

# Builds the explicit hardware DragonNest Agent debug APK for the Galaxy S25:
# GenieX 0.3.5 runtime closure packaged, mock runtime executor compiled out.
# Never bundles model bundles/weights into the APK (those are adb-provisioned
# separately by scripts/accept_s25_local_demo.ps1).

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$androidDir = Join-Path $repoRoot "android-agent"

function Fail {
    param([string]$Category, [string]$Message)
    Write-Host "[$Category] $Message" -ForegroundColor Red
    throw "$Category`: $Message"
}

Write-Host "== 1. Verifying JDK ==" -ForegroundColor Cyan
if (-not $JavaHome) {
    Fail "JDK_NOT_FOUND" "JAVA_HOME is not set and -JavaHome was not passed. Install JDK 17+ and set JAVA_HOME."
}
$javaExe = Join-Path $JavaHome "bin\java.exe"
if (-not (Test-Path -LiteralPath $javaExe)) {
    Fail "JDK_NOT_FOUND" "No java.exe under '$JavaHome\bin'. JAVA_HOME must point at a JDK install root."
}
# java -version writes to stderr; route through cmd.exe so PowerShell sees
# plain merged text instead of wrapping each line as a terminating
# NativeCommandError under $ErrorActionPreference = "Stop".
$javaVersionOutput = (cmd.exe /d /c "`"$javaExe`" -version 2>&1") -join "`n"
$versionMatch = [regex]::Match($javaVersionOutput, 'version "(\d+)')
if (-not $versionMatch.Success -or [int]$versionMatch.Groups[1].Value -lt 17) {
    Fail "JDK_NOT_FOUND" "JDK 17 or newer is required. Found: $javaVersionOutput"
}
Write-Host "  JAVA_HOME=$JavaHome ($($versionMatch.Groups[0].Value))"

Write-Host "== 2. Verifying Android SDK ==" -ForegroundColor Cyan
if (-not $AndroidSdkRoot) {
    $localProps = Join-Path $androidDir "local.properties"
    if (Test-Path -LiteralPath $localProps) {
        $sdkLine = Get-Content -LiteralPath $localProps | Where-Object { $_ -match "^sdk\.dir=" }
        if ($sdkLine) {
            # local.properties is a Java .properties file: ':' and '\' are
            # backslash-escaped (e.g. "C\:\\Users\\...").
            $AndroidSdkRoot = (($sdkLine -replace "^sdk\.dir=", "") -replace "\\:", ":") -replace "\\\\", "\"
        }
    }
}
if (-not $AndroidSdkRoot -or -not (Test-Path -LiteralPath (Join-Path $AndroidSdkRoot "platforms\android-35"))) {
    Fail "ANDROID_SDK_MISSING" "Android SDK platform android-35 was not found under '$AndroidSdkRoot'. Set ANDROID_SDK_ROOT or android-agent/local.properties sdk.dir."
}
Write-Host "  ANDROID_SDK_ROOT=$AndroidSdkRoot"

$env:JAVA_HOME = $JavaHome
$env:ANDROID_SDK_ROOT = $AndroidSdkRoot
$env:ANDROID_HOME = $AndroidSdkRoot
if ($GradleUserHome) { $env:GRADLE_USER_HOME = $GradleUserHome }

Push-Location $androidDir
try {
    $gradlew = Join-Path $androidDir "gradlew.bat"
    if (-not (Test-Path -LiteralPath $gradlew)) {
        Fail "APK_BUILD_FAILED" "gradlew.bat not found under android-agent."
    }
    $gradleArgs = @("--no-daemon", "-PincludeS25GenieXRuntime=true")

    Write-Host "== 3. Verifying the licensed GenieX Maven/runtime dependency resolves ==" -ForegroundColor Cyan
    # Do not merge gradlew's stderr into the success stream here: under
    # $ErrorActionPreference = "Stop", PowerShell 5.1 wraps merged native
    # stderr lines as terminating NativeCommandErrors. Stdout (captured
    # below) is where Gradle prints the dependency tree; stderr still
    # reaches the console directly.
    $depsOutput = & $gradlew @gradleArgs ":app:dependencies" "--configuration" "debugRuntimeClasspath"
    $depsText = ($depsOutput | Out-String)
    Write-Host $depsText
    if ($LASTEXITCODE -ne 0) {
        Fail "GENIEX_INIT_FAILED" "Gradle dependency resolution failed before any GenieX check ran. See output above."
    }
    if ($depsText -notmatch "com\.qualcomm\.qti:geniex-android:0\.3\.5") {
        Fail "GENIEX_INIT_FAILED" "com.qualcomm.qti:geniex-android:0.3.5 did not appear in the resolved debugRuntimeClasspath. Configure access to Qualcomm's licensed GenieX Maven repository (repo URL/credentials) in GRADLE_USER_HOME/gradle.properties or a local Maven mirror, then retry."
    }
    Write-Host "  GenieX 0.3.5 runtime dependency resolved."

    if (-not $SkipUnitTests) {
        Write-Host "== 4. Running Android unit tests ==" -ForegroundColor Cyan
        & $gradlew @gradleArgs ":app:testDebugUnitTest"
        if ($LASTEXITCODE -ne 0) {
            Fail "ANDROID_UNIT_TESTS_FAILED" "gradlew :app:testDebugUnitTest failed with exit code $LASTEXITCODE."
        }
    } else {
        Write-Host "== 4. Skipping Android unit tests (-SkipUnitTests) ==" -ForegroundColor Yellow
    }

    Write-Host "== 5. Building the explicit hardware debug APK ==" -ForegroundColor Cyan
    & $gradlew @gradleArgs ":app:assembleDebug"
    if ($LASTEXITCODE -ne 0) {
        Fail "APK_BUILD_FAILED" "gradlew :app:assembleDebug failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host "== 6. Verifying the resulting APK ==" -ForegroundColor Cyan
$apkPath = Join-Path $androidDir "app\build\outputs\apk\debug\app-debug.apk"
if (-not (Test-Path -LiteralPath $apkPath)) {
    Fail "APK_BUILD_FAILED" "Expected APK not found at '$apkPath' after a successful build."
}
$resolvedApkPath = (Resolve-Path -LiteralPath $apkPath).Path
$hash = (Get-FileHash -LiteralPath $resolvedApkPath -Algorithm SHA256).Hash
$sizeBytes = (Get-Item -LiteralPath $resolvedApkPath).Length

$applicationId = "com.dragonnest.agent"
$metadataPath = Join-Path $androidDir "app\build\outputs\apk\debug\output-metadata.json"
if (Test-Path -LiteralPath $metadataPath) {
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($metadata.applicationId) { $applicationId = $metadata.applicationId }
}

# DRAGONNEST_ENABLE_MOCK_RUNTIME is the negation of includeS25GenieXRuntime
# (android-agent/app/build.gradle.kts). Confirm it directly from the
# generated BuildConfig where possible; fall back to the minSdk side-effect
# of the same Gradle conditional (minSdk=27 iff includeS25GenieXRuntime=true).
$mockDisabled = $null
$buildConfigCandidates = @(
    (Join-Path $androidDir "app\build\generated\source\buildConfig\debug\com\dragonnest\agent\BuildConfig.java"),
    (Join-Path $androidDir "app\build\generated\buildConfig\debug\com\dragonnest\agent\BuildConfig.java")
)
$buildConfigPath = $buildConfigCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($buildConfigPath) {
    $buildConfigText = Get-Content -LiteralPath $buildConfigPath -Raw
    $mockMatch = [regex]::Match($buildConfigText, "DRAGONNEST_ENABLE_MOCK_RUNTIME\s*=\s*(true|false)")
    if ($mockMatch.Success) { $mockDisabled = ($mockMatch.Groups[1].Value -eq "false") }
}
if ($null -eq $mockDisabled -and (Test-Path -LiteralPath $metadataPath)) {
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($metadata.minSdkVersionForDexing) {
        $mockDisabled = ($metadata.minSdkVersionForDexing -ge 27)
    }
}

Write-Host ""
Write-Host "APK path       : $resolvedApkPath"
Write-Host "APK SHA-256    : $hash"
Write-Host "APK size       : $sizeBytes bytes"
Write-Host "applicationId  : $applicationId"
if ($null -eq $mockDisabled) {
    Write-Host "Mock disabled  : could not be cheaply verified" -ForegroundColor Yellow
} elseif ($mockDisabled) {
    Write-Host "Mock disabled  : true (hardware build)" -ForegroundColor Green
} else {
    Fail "APK_NOT_HARDWARE_BUILD" "The produced APK still has DRAGONNEST_ENABLE_MOCK_RUNTIME=true; this is not a hardware build."
}
Write-Host ""
Write-Host "HARDWARE APK BUILD COMPLETE" -ForegroundColor Green

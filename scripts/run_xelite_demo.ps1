<#
.SYNOPSIS
    Starts the DragonNest Brain and real X Elite worker on one laptop.

.DESCRIPTION
    Opens two visible PowerShell windows. The Brain listens on the LAN while
    pc-01 connects over loopback and runs the physically verified
    qwen3-4b-genie Genie/HTP bundle through run_xelite_worker.ps1.

.PARAMETER EnrollmentToken
    Shared Brain/worker token. Defaults to DRAGONNEST_ENROLLMENT_TOKEN. If
    neither is supplied, a cryptographically random token is generated.

.PARAMETER GenieDir
    Optional path to the verified Genie bundle. If omitted, the existing
    worker launcher performs its strict discovery and checksum validation.
#>
param(
    [string]$EnrollmentToken = $env:DRAGONNEST_ENROLLMENT_TOKEN,
    [string]$GenieDir,
    [string]$StageDir,
    [string]$Qairt245Root,
    [string]$Qwen17Tokenizer
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$elasticRequested = [bool]$StageDir -or [bool]$Qairt245Root
$WorkerScript = Join-Path $PSScriptRoot $(if ($elasticRequested) {
    "run_xelite_elastic_worker.ps1"
} else {
    "run_xelite_worker.ps1"
})

function Fail([string]$Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function New-EnrollmentToken {
    $bytes = New-Object byte[] 32
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    } finally {
        $generator.Dispose()
    }
    return [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
}

function ConvertTo-EncodedCommand([string]$Command) {
    return [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Command))
}

function ConvertTo-SingleQuotedLiteral([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

if (-not (Test-Path -LiteralPath $Python)) {
    Fail ".venv not found. Create it and install DragonNest before running the demo."
}
if (-not (Test-Path -LiteralPath $WorkerScript)) {
    Fail "Missing scripts/run_xelite_worker.ps1."
}

$generatedToken = -not $EnrollmentToken
if ($generatedToken) {
    $EnrollmentToken = New-EnrollmentToken
}
if ($EnrollmentToken -eq "dev-token" -or $EnrollmentToken.Length -lt 24) {
    Fail "Use a nontrivial enrollment token of at least 24 characters; dev-token is refused."
}
$env:DRAGONNEST_ENROLLMENT_TOKEN = $EnrollmentToken

if ($GenieDir) {
    if (-not (Test-Path -LiteralPath $GenieDir -PathType Container)) {
        Fail "GenieDir does not exist: $GenieDir"
    }
    $env:DRAGONNEST_XELITE_DEMO_GENIE_DIR = (Resolve-Path -LiteralPath $GenieDir).Path
} else {
    Remove-Item Env:DRAGONNEST_XELITE_DEMO_GENIE_DIR -ErrorAction SilentlyContinue
}

$lanAddress = Get-NetIPConfiguration |
    Where-Object {
        $_.NetAdapter.Status -eq "Up" -and
        $_.IPv4DefaultGateway -and
        $_.IPv4Address
    } |
    ForEach-Object { $_.IPv4Address.IPAddress } |
    Where-Object { $_ -and $_ -notlike "169.254.*" -and $_ -ne "127.0.0.1" } |
    Select-Object -First 1
if (-not $lanAddress) {
    $lanAddress = Get-NetIPAddress -AddressFamily IPv4 -Type Unicast |
        Where-Object { $_.IPAddress -notlike "169.254.*" -and $_.IPAddress -ne "127.0.0.1" } |
        Sort-Object InterfaceMetric |
        Select-Object -First 1 -ExpandProperty IPAddress
}
if (-not $lanAddress) {
    Fail "No non-loopback IPv4 address was found for PersonaCare enrollment."
}

$PowerShellPath = (Get-Process -Id $PID).Path
$repoLiteral = ConvertTo-SingleQuotedLiteral $RepoRoot
$pythonLiteral = ConvertTo-SingleQuotedLiteral $Python
$brainScriptLiteral = ConvertTo-SingleQuotedLiteral (Join-Path $PSScriptRoot "run_brain.py")
$workerScriptLiteral = ConvertTo-SingleQuotedLiteral $WorkerScript

$brainCommand = @"
Set-Location -LiteralPath $repoLiteral
`$Host.UI.RawUI.WindowTitle = 'DragonNest Brain'
& $pythonLiteral $brainScriptLiteral --address 0.0.0.0:50051 --http-host 0.0.0.0 --http-port 8080 --enrollment-token `$env:DRAGONNEST_ENROLLMENT_TOKEN --default-task-timeout-ms 75000 --disable-http-endpoints
"@

$workerInvocation = "& $workerScriptLiteral -Brain '127.0.0.1:50051'"
if ($elasticRequested -and (-not $StageDir -or -not $Qairt245Root)) {
    Fail "-StageDir and -Qairt245Root must be supplied together for Elastic."
}
if ($elasticRequested) {
    $stageLiteral = ConvertTo-SingleQuotedLiteral ((Resolve-Path -LiteralPath $StageDir).Path)
    $qairtLiteral = ConvertTo-SingleQuotedLiteral ((Resolve-Path -LiteralPath $Qairt245Root).Path)
    $workerInvocation += " -StageDir $stageLiteral -Qairt245Root $qairtLiteral"
}
if ($GenieDir) {
    $workerInvocation += " -GenieDir `$env:DRAGONNEST_XELITE_DEMO_GENIE_DIR"
}
if ($Qwen17Tokenizer) {
    $tokenizerLiteral = ConvertTo-SingleQuotedLiteral $Qwen17Tokenizer
    $workerInvocation += " -Qwen17Tokenizer $tokenizerLiteral"
}
$workerCommand = @"
Set-Location -LiteralPath $repoLiteral
`$Host.UI.RawUI.WindowTitle = 'DragonNest X Elite Worker (pc-01)'
$workerInvocation
"@

$brainProcess = Start-Process `
    -FilePath $PowerShellPath `
    -ArgumentList @("-NoExit", "-EncodedCommand", (ConvertTo-EncodedCommand $brainCommand)) `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Normal `
    -PassThru

Start-Sleep -Seconds 1

$workerProcess = Start-Process `
    -FilePath $PowerShellPath `
    -ArgumentList @("-NoExit", "-EncodedCommand", (ConvertTo-EncodedCommand $workerCommand)) `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Normal `
    -PassThru

Write-Host ""
Write-Host "DragonNest same-host X Elite demo started" -ForegroundColor Cyan
Write-Host "Brain process:       $($brainProcess.Id)"
Write-Host "Worker process:      $($workerProcess.Id)"
Write-Host "Laptop worker:       pc-01 -> 127.0.0.1:50051"
Write-Host "Elastic provider:    $(if ($elasticRequested) { 'Qwen3-1.7B S0-S3 enabled (acceptance pending)' } else { 'disabled' })"
Write-Host "PersonaCare Brain:   ${lanAddress}:50051" -ForegroundColor Green
Write-Host "Dashboard:           http://${lanAddress}:8080/admin" -ForegroundColor Green
Write-Host "Task timeout:        75000 ms"
Write-Host "Enrollment token:    $(if ($generatedToken) { 'generated and shared automatically' } else { 'supplied and shared automatically' })"
Write-Host ""
Write-Host "Windows Firewall was not changed. If the phone cannot connect, allow inbound TCP 50051 and 8080 manually for this trusted network." -ForegroundColor Yellow

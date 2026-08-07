<#
.SYNOPSIS
    Start a LAN-visible DragonNest Brain for the physical X Elite worker demo.

.DESCRIPTION
    Uses DRAGONNEST_ENROLLMENT_TOKEN when set, otherwise generates and prints a
    random shared token. Binds gRPC and the dashboard to all interfaces, prints
    candidate LAN addresses, and disables the unrelated OpenAI-compatible HTTP
    endpoint registration subsystem for this physical-agent experiment.

    This script never modifies Windows Firewall.

.EXAMPLE
    $env:DRAGONNEST_ENROLLMENT_TOKEN = "<random-shared-token>"
    .\scripts\run_demo_brain.ps1
#>
param(
    [string]$EnrollmentToken = $env:DRAGONNEST_ENROLLMENT_TOKEN,

    [ValidateRange(1, 65535)]
    [int]$GrpcPort = 50051,

    [ValidateRange(1, 65535)]
    [int]$HttpPort = 8080
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

function Fail($message) {
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path -LiteralPath $Python)) {
    Fail ".venv not found. Run: py -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e `".[dev]`""
}

$generatedToken = $false
if (-not $EnrollmentToken) {
    $bytes = New-Object byte[] 32
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    $EnrollmentToken = [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
    $generatedToken = $true
}
if ($EnrollmentToken -eq "dev-token") {
    Fail "Refusing the literal dev-token on a LAN-visible Brain. Set DRAGONNEST_ENROLLMENT_TOKEN to a random shared value or let this script generate one."
}
$env:DRAGONNEST_ENROLLMENT_TOKEN = $EnrollmentToken

$addresses = @()
try {
    $addresses = @(Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred -ErrorAction Stop |
        Where-Object {
            $_.IPAddress -ne "127.0.0.1" -and
            -not $_.IPAddress.StartsWith("169.254.")
        } |
        Sort-Object InterfaceMetric, IPAddress |
        Select-Object -ExpandProperty IPAddress -Unique)
}
catch {
    Write-Host "WARNING: could not enumerate LAN IPv4 addresses: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "DragonNest Desktop Brain" -ForegroundColor Cyan
Write-Host "gRPC bind:       0.0.0.0:$GrpcPort"
Write-Host "Dashboard bind:  0.0.0.0:$HttpPort"
Write-Host "Endpoint pool:   disabled (physical gRPC agents only for this experiment)"
if ($generatedToken) {
    Write-Host "Generated enrollment token (copy this to the X Elite):" -ForegroundColor Yellow
} else {
    Write-Host "Using DRAGONNEST_ENROLLMENT_TOKEN / -EnrollmentToken:" -ForegroundColor Green
}
Write-Host "  $EnrollmentToken"
Write-Host "X Elite token command:"
Write-Host "  `$env:DRAGONNEST_ENROLLMENT_TOKEN = `"$EnrollmentToken`""
Write-Host ""

if ($addresses.Count -eq 0) {
    Write-Host "No non-loopback IPv4 address was detected. Run ipconfig and choose the desktop address reachable from the X Elite." -ForegroundColor Yellow
} else {
    Write-Host "Candidate X Elite commands (choose the address on the shared LAN):"
    foreach ($address in $addresses) {
        Write-Host "  .\scripts\run_xelite_worker.ps1 -Brain ${address}:$GrpcPort"
        Write-Host "  Dashboard: http://${address}:$HttpPort/admin"
    }
}
Write-Host ""
Write-Host "Windows Firewall is not changed. If the X Elite cannot connect, allow inbound TCP $GrpcPort and $HttpPort manually for the intended network profile." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the Brain."
Write-Host ""

Set-Location $RepoRoot
& $Python "scripts\run_brain.py" `
    --address "0.0.0.0:$GrpcPort" `
    --http-host "0.0.0.0" `
    --http-port $HttpPort `
    --enrollment-token $EnrollmentToken `
    --disable-http-endpoints

exit $LASTEXITCODE

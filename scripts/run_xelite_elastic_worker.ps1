<#
.SYNOPSIS
    Starts pc-01 with both verified 4B Genie and production 1.7B QNN stages.

.DESCRIPTION
    This is the physical-acceptance handoff for DragonNest's normal Agent
    protocol. It pins the four externally staged context binaries, then reuses
    run_xelite_worker.ps1 for 4B validation, enrollment, and Agent startup.
    It never invokes the standalone smoke-test bypass.
#>
param(
    [Parameter(Mandatory = $true)][string]$Brain,
    [Parameter(Mandatory = $true)][string]$StageDir,
    [Parameter(Mandatory = $true)][string]$Qairt245Root,
    [string]$GenieDir,
    [string]$Qwen17Tokenizer,
    [string]$EnrollmentToken = $env:DRAGONNEST_ENROLLMENT_TOKEN
)

$ErrorActionPreference = "Stop"
$stage = (Resolve-Path -LiteralPath $StageDir).Path
for ($index = 0; $index -lt 4; $index++) {
    $path = Join-Path $stage "qwen3-1.7b-s$index-xelite.bin"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing staged Qwen3-1.7B context: $path"
    }
    [Environment]::SetEnvironmentVariable(
        "QWEN3_1_7B_S${index}_XELITE_QNN", $path, "Process"
    )
}

$arguments = @{
    Brain = $Brain
    EnableQwen17Pipeline = $true
    Qairt245Root = $Qairt245Root
}
if ($GenieDir) { $arguments.GenieDir = $GenieDir }
if ($Qwen17Tokenizer) { $arguments.Qwen17Tokenizer = $Qwen17Tokenizer }
if ($EnrollmentToken) { $arguments.EnrollmentToken = $EnrollmentToken }

& (Join-Path $PSScriptRoot "run_xelite_worker.ps1") @arguments

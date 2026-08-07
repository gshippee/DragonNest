<#
.SYNOPSIS
    Optional high-frequency available-memory sampler for X Elite calibration.

.DESCRIPTION
    Run this in a second X Elite terminal while the normal DragonNest Agent is
    already connected and the desktop validation harness submits requests.
    It does not start/call the Agent, Brain, Genie, or HTP and writes only
    timestamped system available-memory values to a secret-free JSON file.

.EXAMPLE
    .\scripts\sample_xelite_memory.ps1 `
      -DurationSeconds 180 `
      -Output "$env:TEMP\dragonnest-xelite-memory-samples.json"
#>
param(
    [ValidateRange(1, 3600)]
    [int]$DurationSeconds = 180,

    [ValidateRange(50, 5000)]
    [int]$IntervalMilliseconds = 100,

    [string]$Output = (Join-Path $env:TEMP "dragonnest-xelite-memory-samples.json")
)

$ErrorActionPreference = "Stop"

if (-not ("DragonNestNativeMemory" -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class DragonNestNativeMemory {
    [StructLayout(LayoutKind.Sequential)]
    public class MemoryStatusEx {
        public uint Length = (uint)Marshal.SizeOf(typeof(MemoryStatusEx));
        public uint MemoryLoad;
        public ulong TotalPhysical;
        public ulong AvailablePhysical;
        public ulong TotalPageFile;
        public ulong AvailablePageFile;
        public ulong TotalVirtual;
        public ulong AvailableVirtual;
        public ulong AvailableExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GlobalMemoryStatusEx([In, Out] MemoryStatusEx status);
}
"@
}

$samples = [System.Collections.Generic.List[object]]::new()
$started = [DateTimeOffset]::UtcNow
$deadline = [DateTimeOffset]::UtcNow.AddSeconds($DurationSeconds)
Write-Host "Sampling X Elite available memory every $IntervalMilliseconds ms for up to $DurationSeconds seconds."
Write-Host "Leave the normal DragonNest worker running; press Ctrl+C after the desktop runs finish."

try {
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $status = [DragonNestNativeMemory+MemoryStatusEx]::new()
        if (-not [DragonNestNativeMemory]::GlobalMemoryStatusEx($status)) {
            throw "GlobalMemoryStatusEx failed with Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
        }
        $samples.Add([ordered]@{
            timestamp_utc = [DateTimeOffset]::UtcNow.ToString("o")
            available_memory_mb = [math]::Round($status.AvailablePhysical / 1MB)
        })
        Start-Sleep -Milliseconds $IntervalMilliseconds
    }
}
finally {
    if ($samples.Count -gt 0) {
        $values = @($samples | ForEach-Object { $_.available_memory_mb })
        $proof = [ordered]@{
            schema_version = 1
            started_at_utc = $started.ToString("o")
            finished_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
            interval_milliseconds = $IntervalMilliseconds
            sample_count = $samples.Count
            minimum_available_memory_mb = ($values | Measure-Object -Minimum).Minimum
            maximum_available_memory_mb = ($values | Measure-Object -Maximum).Maximum
            samples = $samples
        }
        $destination = [IO.Path]::GetFullPath($Output)
        $parent = Split-Path -Parent $destination
        if (-not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        $proof | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $destination -Encoding UTF8
        Write-Host "Wrote $($samples.Count) secret-free samples to $destination"
    }
}

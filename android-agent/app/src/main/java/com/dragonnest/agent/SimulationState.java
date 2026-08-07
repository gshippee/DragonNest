package com.dragonnest.agent;

public record SimulationState(
        Float batteryPercentage,
        Float thermalLevel,
        Float cpuUtilization,
        Float acceleratorUtilization,
        Float gpuUtilization,
        Float npuUtilization,
        Float networkRttMs,
        Long availableMemoryMb,
        boolean offline) {
    public static SimulationState none() {
        return new SimulationState(null, null, null, null, null, null, null, null, false);
    }

    public TelemetrySnapshot apply(TelemetrySnapshot source) {
        boolean enabled = batteryPercentage != null || thermalLevel != null
                || cpuUtilization != null || acceleratorUtilization != null
                || gpuUtilization != null || npuUtilization != null
                || networkRttMs != null || availableMemoryMb != null || offline;
        if (!enabled) {
            return source;
        }
        return new TelemetrySnapshot(
                batteryPercentage != null ? batteryPercentage : source.batteryPercentage(),
                source.charging(),
                thermalLevel != null ? thermalLevel : source.thermalLevel(),
                availableMemoryMb != null ? availableMemoryMb : source.availableMemoryMb(),
                cpuUtilization != null ? cpuUtilization : source.cpuUtilization(),
                acceleratorUtilization != null
                        ? acceleratorUtilization : source.acceleratorUtilization(),
                gpuUtilization != null ? gpuUtilization : source.gpuUtilization(),
                npuUtilization != null ? npuUtilization : source.npuUtilization(),
                networkRttMs != null ? networkRttMs : source.networkRttMs(),
                source.activeTaskIds(),
                source.warmModelIds(),
                !offline,
                true);
    }
}

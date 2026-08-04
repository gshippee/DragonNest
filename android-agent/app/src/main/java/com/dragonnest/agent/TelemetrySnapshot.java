package com.dragonnest.agent;

import java.util.List;

public record TelemetrySnapshot(
        float batteryPercentage,
        boolean charging,
        float thermalLevel,
        long availableMemoryMb,
        float cpuUtilization,
        float acceleratorUtilization,
        float networkRttMs,
        List<String> activeTaskIds,
        List<String> warmModelIds,
        boolean reachable,
        boolean simulated) {
    public static final float UNKNOWN = -1.0f;
}

package com.dragonnest.agent;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Build;
import android.os.PowerManager;

/** Samples the platform values that can change during a runtime invocation. */
final class AndroidExecutionSampler {
    private final ActivityManager activityManager;
    private final PowerManager powerManager;

    AndroidExecutionSampler(Context context) {
        activityManager = context.getSystemService(ActivityManager.class);
        powerManager = context.getSystemService(PowerManager.class);
    }

    Snapshot sample() {
        ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(memory);
        Float thermal = null;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            thermal = normalizeThermal(powerManager.getCurrentThermalStatus());
        }
        return new Snapshot(memory.availMem / (1024L * 1024L), thermal);
    }

    static ExecutionDetails withObservedDeltas(
            ExecutionDetails details, Snapshot before, Snapshot after) {
        long memoryDelta = before.availableMemoryMb() - after.availableMemoryMb();
        Float thermalDelta = before.thermalLevel() == null || after.thermalLevel() == null
                ? null : after.thermalLevel() - before.thermalLevel();
        return new ExecutionDetails(
                details.modelId(),
                details.modelVersion(),
                details.runtimeName(),
                details.runtimeVersion(),
                details.accelerator(),
                memoryDelta,
                thermalDelta);
    }

    private static Float normalizeThermal(int status) {
        if (status < PowerManager.THERMAL_STATUS_NONE) {
            return null;
        }
        return Math.min(status / (float) PowerManager.THERMAL_STATUS_SHUTDOWN, 1.0f);
    }

    record Snapshot(long availableMemoryMb, Float thermalLevel) { }
}

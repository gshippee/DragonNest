package com.dragonnest.agent;

import android.app.ActivityManager;
import android.content.Context;
import android.os.BatteryManager;
import android.os.Build;
import android.os.PowerManager;
import android.util.Log;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

public final class AndroidTelemetry {
    private static final String TAG = "DragonNestTelemetry";

    private final ActivityManager activityManager;
    private final BatteryManager batteryManager;
    private final PowerManager powerManager;
    private volatile SimulationState simulation = SimulationState.none();

    // Delta state for /proc/stat-based CPU sampling; null until the first sample.
    private long[] previousCpuJiffies;
    // Delta state for /sys/class/kgsl gpubusy-based GPU sampling.
    private long previousGpuBusy = -1;
    private long previousGpuTotal = -1;

    public AndroidTelemetry(Context context) {
        activityManager = context.getSystemService(ActivityManager.class);
        batteryManager = context.getSystemService(BatteryManager.class);
        powerManager = context.getSystemService(PowerManager.class);
    }

    public void setSimulation(SimulationState simulation) {
        this.simulation = simulation;
    }

    public TelemetrySnapshot sample(List<String> activeTasks, List<String> warmModels) {
        ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(memory);
        float battery = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
        boolean charging = batteryManager.isCharging();
        float thermal = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q
                ? normalizeThermal(powerManager.getCurrentThermalStatus())
                : TelemetrySnapshot.UNKNOWN;
        float cpuUtilization = sampleCpuUtilization();
        float gpuUtilization = sampleGpuUtilization();
        // No public Android API and no busy/load sysfs node exists for the
        // Hexagon NPU/CDSP (only IOMMU/remoteproc plumbing paths) — unlike the
        // GPU's kgsl-3d0/gpubusy node, there is nothing to read here.
        float npuUtilization = TelemetrySnapshot.UNKNOWN;
        TelemetrySnapshot measured = new TelemetrySnapshot(
                battery < 0 ? TelemetrySnapshot.UNKNOWN : battery,
                charging,
                thermal,
                memory.availMem / (1024L * 1024L),
                cpuUtilization,
                Math.max(gpuUtilization, npuUtilization),
                gpuUtilization,
                npuUtilization,
                TelemetrySnapshot.UNKNOWN,
                List.copyOf(activeTasks),
                List.copyOf(warmModels),
                true,
                false);
        return simulation.apply(measured);
    }

    private static float normalizeThermal(int status) {
        if (status < PowerManager.THERMAL_STATUS_NONE) {
            return TelemetrySnapshot.UNKNOWN;
        }
        return Math.min(status / (float) PowerManager.THERMAL_STATUS_SHUTDOWN, 1.0f);
    }

    /**
     * System-wide CPU utilization from the aggregate {@code /proc/stat} line,
     * delta-based across samples (same technique {@code top}/{@code htop} use).
     *
     * <p>This file is world-readable DAC-wise on stock Android, but SELinux
     * policy — not file permission bits — is the real gate for a third-party
     * app process, and that varies by OEM/ROM. We degrade to UNKNOWN on any
     * read failure rather than assume access.
     */
    private float sampleCpuUtilization() {
        long[] jiffies = readCpuJiffies();
        if (jiffies == null) {
            return TelemetrySnapshot.UNKNOWN;
        }
        if (previousCpuJiffies == null) {
            previousCpuJiffies = jiffies;
            return TelemetrySnapshot.UNKNOWN; // no delta yet
        }
        long idleDelta = (jiffies[3] + jiffies[4]) - (previousCpuJiffies[3] + previousCpuJiffies[4]);
        long totalDelta = 0;
        for (int i = 0; i < jiffies.length; i++) {
            totalDelta += jiffies[i] - previousCpuJiffies[i];
        }
        previousCpuJiffies = jiffies;
        if (totalDelta <= 0) {
            return 0f;
        }
        float busy = (totalDelta - idleDelta) / (float) totalDelta;
        return Math.min(Math.max(busy, 0f), 1f);
    }

    /** Parses the {@code cpu } aggregate line: user nice system idle iowait irq softirq steal guest guest_nice. */
    private long[] readCpuJiffies() {
        try (BufferedReader reader = new BufferedReader(new FileReader("/proc/stat"))) {
            String line = reader.readLine();
            if (line == null || !line.startsWith("cpu ")) {
                return null;
            }
            String[] parts = line.trim().split("\\s+");
            long[] jiffies = new long[parts.length - 1];
            for (int i = 1; i < parts.length; i++) {
                jiffies[i - 1] = Long.parseLong(parts[i]);
            }
            return jiffies;
        } catch (IOException | NumberFormatException | SecurityException e) {
            Log.d(TAG, "/proc/stat unavailable: " + e.getMessage());
            return null;
        }
    }

    /**
     * Adreno GPU utilization from the kgsl driver's {@code gpubusy} node,
     * delta-based across samples: two cumulative counters, "busy cycles" and
     * "total wall cycles", whose ratio over a sampling window is the busy
     * fraction. Same best-effort/SELinux caveat as {@link #sampleCpuUtilization}.
     */
    private float sampleGpuUtilization() {
        long[] busyTotal = readGpuBusy();
        if (busyTotal == null) {
            return TelemetrySnapshot.UNKNOWN;
        }
        long busy = busyTotal[0];
        long total = busyTotal[1];
        if (previousGpuTotal < 0) {
            previousGpuBusy = busy;
            previousGpuTotal = total;
            return TelemetrySnapshot.UNKNOWN; // no delta yet
        }
        long busyDelta = busy - previousGpuBusy;
        long totalDelta = total - previousGpuTotal;
        previousGpuBusy = busy;
        previousGpuTotal = total;
        if (totalDelta <= 0) {
            return 0f;
        }
        return Math.min(Math.max(busyDelta / (float) totalDelta, 0f), 1f);
    }

    private long[] readGpuBusy() {
        try (BufferedReader reader = new BufferedReader(
                new FileReader("/sys/class/kgsl/kgsl-3d0/gpubusy"))) {
            String line = reader.readLine();
            if (line == null) {
                return null;
            }
            String[] parts = line.trim().split("\\s+");
            if (parts.length < 2) {
                return null;
            }
            return new long[] {Long.parseLong(parts[0]), Long.parseLong(parts[1])};
        } catch (IOException | NumberFormatException | SecurityException e) {
            Log.d(TAG, "gpubusy unavailable: " + e.getMessage());
            return null;
        }
    }
}

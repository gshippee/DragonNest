package com.dragonnest.agent;

import android.app.ActivityManager;
import android.content.Context;
import android.os.BatteryManager;
import android.os.Build;
import android.os.PowerManager;

import java.util.List;

public final class AndroidTelemetry {
    private final ActivityManager activityManager;
    private final BatteryManager batteryManager;
    private final PowerManager powerManager;
    private volatile SimulationState simulation = SimulationState.none();

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
        TelemetrySnapshot measured = new TelemetrySnapshot(
                battery < 0 ? TelemetrySnapshot.UNKNOWN : battery,
                charging,
                thermal,
                memory.availMem / (1024L * 1024L),
                TelemetrySnapshot.UNKNOWN,
                TelemetrySnapshot.UNKNOWN,
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
}

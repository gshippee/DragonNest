package com.dragonnest.agent;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;

import java.util.UUID;

public final class AgentConfiguration {
    private static final String PREFERENCES = "agent-configuration";
    private final SharedPreferences preferences;

    public AgentConfiguration(Context context) {
        preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE);
    }

    public String brainHost() {
        return preferences.getString("brain_host", "10.0.2.2");
    }

    public int brainPort() {
        return preferences.getInt("brain_port", 50051);
    }

    public boolean useTls() {
        return preferences.getBoolean("use_tls", false);
    }

    public String deviceId() {
        String current = preferences.getString("device_id", "");
        if (!current.trim().isEmpty()) {
            return current;
        }
        String generated = "android-" + UUID.randomUUID();
        preferences.edit().putString("device_id", generated).apply();
        return generated;
    }

    public String displayName() {
        return preferences.getString("display_name", Build.MANUFACTURER + " " + Build.MODEL);
    }

    public void save(String host, int port, boolean tls, String displayName) {
        preferences.edit()
                .putString("brain_host", host.trim())
                .putInt("brain_port", port)
                .putBoolean("use_tls", tls)
                .putString("display_name", displayName.trim())
                .apply();
    }

    public void saveEnrollmentEndpoint(String host, int port, boolean tls) {
        save(host, port, tls, displayName());
    }

    public void clearEnrollmentEndpoint() {
        preferences.edit()
                .remove("brain_host")
                .remove("brain_port")
                .remove("use_tls")
                .apply();
    }

    public SimulationState simulation() {
        return new SimulationState(
                optionalFloat("simulation_battery"),
                optionalFloat("simulation_thermal"),
                optionalFloat("simulation_cpu"),
                optionalFloat("simulation_accelerator"),
                optionalFloat("simulation_rtt"),
                preferences.getBoolean("simulation_offline", false));
    }

    public void saveSimulation(
            Float battery,
            Float thermal,
            Float cpu,
            Float accelerator,
            Float rtt,
            boolean offline) {
        SharedPreferences.Editor editor = preferences.edit();
        putOptional(editor, "simulation_battery", battery);
        putOptional(editor, "simulation_thermal", thermal);
        putOptional(editor, "simulation_cpu", cpu);
        putOptional(editor, "simulation_accelerator", accelerator);
        putOptional(editor, "simulation_rtt", rtt);
        editor.putBoolean("simulation_offline", offline).apply();
    }

    private Float optionalFloat(String key) {
        return preferences.contains(key) ? preferences.getFloat(key, 0) : null;
    }

    private static void putOptional(
            SharedPreferences.Editor editor, String key, Float value) {
        if (value == null) {
            editor.remove(key);
        } else {
            editor.putFloat(key, value);
        }
    }
}

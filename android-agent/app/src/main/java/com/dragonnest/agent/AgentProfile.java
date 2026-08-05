package com.dragonnest.agent;

import android.app.ActivityManager;
import android.content.Context;

import com.dragonnest.proto.RegisterDevice;

import java.util.List;

public final class AgentProfile {
    private final Context context;
    private final AgentConfiguration configuration;
    private final AndroidHardwareInventory hardwareInventory;
    private final AndroidRuntimeCatalog runtimeCatalog;
    private final UserProfileStore userProfileStore;

    public AgentProfile(
            Context context,
            AgentConfiguration configuration,
            AndroidRuntimeCatalog runtimeCatalog) {
        this.context = context.getApplicationContext();
        this.configuration = configuration;
        this.runtimeCatalog = runtimeCatalog;
        this.hardwareInventory = new AndroidHardwareInventory(this.context, runtimeCatalog);
        this.userProfileStore = new UserProfileStore(this.context);
    }

    public RegisterDevice registration(String enrollmentCredential) {
        ActivityManager manager = context.getSystemService(ActivityManager.class);
        ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo();
        manager.getMemoryInfo(memory);
        RegisterDevice.Builder registration = RegisterDevice.newBuilder()
                .setDeviceId(configuration.deviceId())
                .setDisplayName(configuration.displayName())
                .setDeviceType("phone")
                .setPlatform("android")
                .setAgentVersion("0.2.0")
                .setEnrollmentToken(enrollmentCredential)
                .setTotalMemoryMb(memory.totalMem / (1024L * 1024L))
                .addAllModels(runtimeCatalog.capabilities())
                .setHardware(hardwareInventory.snapshot());
        UserProfile userProfile = userProfileStore.load();
        if (userProfile != null) {
            registration.setPersonalProfile(userProfile.registration());
        }
        return registration.build();
    }

    public List<String> warmModelIds() {
        return runtimeCatalog.warmModelIds();
    }
}

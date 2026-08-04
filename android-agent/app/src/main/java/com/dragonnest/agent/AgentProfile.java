package com.dragonnest.agent;

import android.app.ActivityManager;
import android.content.Context;

import com.dragonnest.proto.ModelCapability;
import com.dragonnest.proto.RegisterDevice;

import java.util.List;

public final class AgentProfile {
    private final Context context;
    private final AgentConfiguration configuration;

    public AgentProfile(Context context, AgentConfiguration configuration) {
        this.context = context.getApplicationContext();
        this.configuration = configuration;
    }

    public RegisterDevice registration(String enrollmentCredential) {
        ActivityManager manager = context.getSystemService(ActivityManager.class);
        ActivityManager.MemoryInfo memory = new ActivityManager.MemoryInfo();
        manager.getMemoryInfo(memory);
        ModelCapability mock = ModelCapability.newBuilder()
                .setModelId("android-mock-v1")
                .setModelFamily("mock")
                .setRole("small_chat")
                .addAllTaskClasses(List.of(
                        "chat_qa",
                        "summarization",
                        "translation_rewrite",
                        "reasoning_analysis",
                        "document_extraction",
                        "code_assistance"))
                .setMaxContextTokens(4096)
                .setWarm(true)
                .setQualityScore(0.60f)
                .setModelVersion("mock-v1")
                .setTokenizerId("mock-tokenizer")
                .setPrecision("text")
                .setRuntimeName("mock")
                .setRuntimeVersion("dragon-nest-android-0.1.0")
                .addSupportedAccelerators("cpu")
                .setMinMemoryMb(128)
                .addSteeringVectorIds("concise-vs-verbose-layer-7")
                .addSupportedSteeringLayers(7)
                .setSupportsSteering(true)
                .setSupportsDataParallel(true)
                .build();
        return RegisterDevice.newBuilder()
                .setDeviceId(configuration.deviceId())
                .setDisplayName(configuration.displayName())
                .setDeviceType("phone")
                .setPlatform("android")
                .setAgentVersion("0.1.0")
                .setEnrollmentToken(enrollmentCredential)
                .setTotalMemoryMb(memory.totalMem / (1024L * 1024L))
                .addModels(mock)
                .build();
    }

    public List<String> warmModelIds() {
        return List.of("android-mock-v1");
    }
}

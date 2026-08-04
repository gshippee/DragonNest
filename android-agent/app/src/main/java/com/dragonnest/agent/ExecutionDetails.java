package com.dragonnest.agent;

/** Immutable execution metadata sent with every task, shard, and pipeline result. */
public record ExecutionDetails(
        String modelId,
        String modelVersion,
        String runtimeName,
        String runtimeVersion,
        String accelerator,
        Long observedMemoryDeltaMb,
        Float observedThermalDelta) {
    public ExecutionDetails {
        modelId = modelId == null ? "" : modelId;
        modelVersion = modelVersion == null ? "" : modelVersion;
        runtimeName = runtimeName == null ? "" : runtimeName;
        runtimeVersion = runtimeVersion == null ? "" : runtimeVersion;
        accelerator = accelerator == null ? "" : accelerator;
    }

    public static ExecutionDetails mock() {
        return new ExecutionDetails(
                MockAndroidTaskExecutor.MODEL_ID,
                "mock-v1",
                "mock",
                "dragon-nest-android-0.1.0",
                "cpu",
                null,
                null);
    }
}

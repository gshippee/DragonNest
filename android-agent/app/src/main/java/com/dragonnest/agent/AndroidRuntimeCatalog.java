package com.dragonnest.agent;

import android.content.Context;
import android.util.Log;

import com.dragonnest.proto.ExecutePipelineStage;
import com.dragonnest.proto.ExecuteShard;
import com.dragonnest.proto.ExecuteTask;
import com.dragonnest.proto.ModelCapability;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The single source of Android model truth for registration and task execution.
 * Real runtimes are deliberately absent until both their artifacts and vendor
 * bridge are verified on the device.
 */
public final class AndroidRuntimeCatalog implements AndroidTaskExecutor {
    private static final String TAG = "DragonNestRuntime";
    private final Map<String, AndroidTaskExecutor> executors;
    private final List<ModelCapability> capabilities;
    private final List<String> warmModelIds;
    private final String npuStatus;
    private final String npuName;
    private final String qnnRuntimeVersion;

    private AndroidRuntimeCatalog(
            Map<String, AndroidTaskExecutor> executors,
            List<ModelCapability> capabilities,
            List<String> warmModelIds,
            String npuStatus,
            String npuName,
            String qnnRuntimeVersion) {
        this.executors = Map.copyOf(executors);
        this.capabilities = List.copyOf(capabilities);
        this.warmModelIds = List.copyOf(warmModelIds);
        this.npuStatus = npuStatus;
        this.npuName = npuName;
        this.qnnRuntimeVersion = qnnRuntimeVersion;
    }

    public static AndroidRuntimeCatalog create(Context context) {
        Map<String, AndroidTaskExecutor> executors = new LinkedHashMap<>();
        List<ModelCapability> capabilities = new ArrayList<>();
        List<String> warm = new ArrayList<>();
        MockAndroidTaskExecutor mock = new MockAndroidTaskExecutor();
        executors.put(MockAndroidTaskExecutor.MODEL_ID, mock);
        capabilities.add(mockCapability());
        warm.add(MockAndroidTaskExecutor.MODEL_ID);

        AndroidArtifactRegistry registry;
        try {
            AndroidModelAssetInstaller.installIfAbsent(context);
            registry = AndroidArtifactRegistry.loadInstalled(context);
        } catch (Exception failure) {
            Log.w(TAG, "Ignoring invalid Android model manifest", failure);
            registry = AndroidArtifactRegistry.fromJson("{\"models\":[]}",
                    context.getFilesDir().toPath().resolve(AndroidArtifactRegistry.MODEL_DIRECTORY));
        }
        Map<String, AndroidRuntimeBridge> bridges = new LinkedHashMap<>();
        boolean npuAvailable = false;
        String availableNpuName = "";
        String availableQnnVersion = "";
        for (AndroidModelArtifact artifact : registry.all()) {
            if (!registry.isVerified(artifact)) {
                Log.w(TAG, "Ignoring model with missing or invalid checksum: " + artifact.modelId());
                continue;
            }
            AndroidRuntimeBridge bridge = bridges.computeIfAbsent(
                    artifact.runtime(), RuntimeBridgeLoader::load);
            if (bridge == null || !bridge.isAvailable(context, artifact)) {
                Log.w(TAG, "No available " + artifact.runtime()
                        + " bridge for model " + artifact.modelId());
                continue;
            }
            AndroidTaskExecutor executor = artifact.runtime().equals("qnn")
                    ? new QnnAndroidTaskExecutor(context, artifact, bridge)
                    : new GenieAndroidTaskExecutor(context, artifact, bridge);
            executors.put(artifact.modelId(), executor);
            capabilities.add(artifact.capability());
            warm.add(artifact.modelId());
            if (usesNpu(artifact)) {
                npuAvailable = true;
                availableNpuName = "Qualcomm " + artifact.supportedAccelerators().get(0).toUpperCase();
                if (artifact.runtime().equals("qnn")) {
                    availableQnnVersion = bridge.runtimeVersion();
                }
            }
        }
        return new AndroidRuntimeCatalog(
                executors,
                capabilities,
                warm,
                npuAvailable ? "available" : "unavailable",
                availableNpuName,
                availableQnnVersion);
    }

    public List<ModelCapability> capabilities() {
        return capabilities;
    }

    public List<String> warmModelIds() {
        return warmModelIds;
    }

    public String npuStatus() {
        return npuStatus;
    }

    public String npuName() {
        return npuName;
    }

    public String qnnRuntimeVersion() {
        return qnnRuntimeVersion;
    }

    @Override
    public TaskExecutionResult execute(ExecuteTask command) throws Exception {
        return executorFor(command.getModelId()).execute(command);
    }

    @Override
    public TaskExecutionResult executeShard(ExecuteShard command) throws Exception {
        return executorFor(command.getModelId()).executeShard(command);
    }

    @Override
    public TaskExecutionResult executePipelineStage(ExecutePipelineStage command) throws Exception {
        return executorFor(command.getModelId()).executePipelineStage(command);
    }

    private AndroidTaskExecutor executorFor(String modelId) {
        AndroidTaskExecutor executor = executors.get(modelId);
        if (executor == null) {
            throw new IllegalArgumentException("Model is not available on this Android Agent: " + modelId);
        }
        return executor;
    }

    private static boolean usesNpu(AndroidModelArtifact artifact) {
        return artifact.supportedAccelerators().stream().anyMatch(
                accelerator -> accelerator.equalsIgnoreCase("htp")
                        || accelerator.equalsIgnoreCase("npu"));
    }

    private static ModelCapability mockCapability() {
        return ModelCapability.newBuilder()
                .setModelId(MockAndroidTaskExecutor.MODEL_ID)
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
    }
}

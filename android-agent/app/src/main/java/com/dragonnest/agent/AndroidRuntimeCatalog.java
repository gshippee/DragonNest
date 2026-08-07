package com.dragonnest.agent;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import com.dragonnest.proto.ExecutePipelineStage;
import com.dragonnest.proto.ExecuteShard;
import com.dragonnest.proto.ExecuteTask;
import com.dragonnest.proto.ModelCapability;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
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
        if (BuildConfig.DRAGONNEST_ENABLE_MOCK_RUNTIME) {
            MockAndroidTaskExecutor mock = new MockAndroidTaskExecutor();
            executors.put(MockAndroidTaskExecutor.MODEL_ID, mock);
            capabilities.add(mockCapability());
            warm.add(MockAndroidTaskExecutor.MODEL_ID);
        }

        AndroidArtifactRegistry registry;
        try {
            AndroidModelAssetInstaller.installIfAbsent(context);
            registry = AndroidArtifactRegistry.loadInstalled(context);
        } catch (Exception failure) {
            Log.w(TAG, "Ignoring invalid Android model manifest", failure);
            registry = AndroidArtifactRegistry.fromJson("{\"models\":[]}",
                    AndroidArtifactRegistry.modelRoot(context));
        }
        for (String skipped : registry.skippedEntries()) {
            Log.w(TAG, "Ignoring unusable Android manifest entry -- " + skipped);
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
            if (!isDeviceCompatible(artifact)) {
                Log.w(TAG, "Ignoring target-incompatible model " + artifact.modelId()
                        + " for " + Build.SOC_MODEL);
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
            // The current bridges create a runtime per request and do not retain
            // a model context.  Advertise installed, but never claim it is warm.
            if (usesNpu(artifact)) {
                npuAvailable = true;
                availableNpuName = "Qualcomm " + artifact.supportedAccelerators().get(0).toUpperCase();
                // HardwareInventory retains the historical qnn_runtime_version
                // field name, but GenieX is also backed by QAIRT/QNN. Report the
                // exact admitted bridge version for either real HTP runtime.
                availableQnnVersion = bridge.runtimeVersion();
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

    /**
     * A real artifact target becomes authoritative only after checksum,
     * runtime-load, and execution-ready probing have admitted its capability.
     * Thin/mock builds therefore retain their generic platform key.
     */
    public String compatibilityKey(String platformFallback) {
        return resolveCompatibilityKey(capabilities, platformFallback);
    }

    static String resolveCompatibilityKey(
            List<ModelCapability> availableCapabilities,
            String platformFallback) {
        LinkedHashSet<String> targets = new LinkedHashSet<>();
        for (ModelCapability capability : availableCapabilities) {
            if (!capability.getTargetCompatibilityClass().isBlank()) {
                targets.add(capability.getTargetCompatibilityClass());
            }
        }
        if (targets.isEmpty()) {
            return platformFallback;
        }
        String selected = targets.iterator().next();
        String family = selected.split("-qairt-", 2)[0];
        for (String target : targets) {
            if (!target.split("-qairt-", 2)[0].equals(family)) {
                throw new IllegalStateException(
                        "Loaded Android runtimes advertise incompatible target families: "
                                + targets);
            }
        }
        return selected;
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

    @Override
    public void cleanupTask(String taskId) {
        executors.values().forEach(executor -> executor.cleanupTask(taskId));
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

    private static boolean isDeviceCompatible(AndroidModelArtifact artifact) {
        String target = artifact.targetCompatibilityClass().toLowerCase();
        if (target.isBlank()) {
            return true;
        }
        String abi = Build.SUPPORTED_ABIS.length == 0
                ? "unknown" : Build.SUPPORTED_ABIS[0].toLowerCase();
        String soc = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S
                ? Build.SOC_MODEL.toLowerCase() : "unknown-soc";
        return target.startsWith("android-" + abi + "-" + soc);
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
                .setArtifactId(MockAndroidTaskExecutor.MODEL_ID)
                .addSteeringModes("runtime_vector")
                .build();
    }
}

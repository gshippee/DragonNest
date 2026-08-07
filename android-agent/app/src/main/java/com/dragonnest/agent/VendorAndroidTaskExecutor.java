package com.dragonnest.agent;

import android.content.Context;

import com.dragonnest.proto.ExecutePipelineStage;
import com.dragonnest.proto.ExecuteShard;
import com.dragonnest.proto.ExecuteTask;
import com.dragonnest.proto.PipelineOperation;

/** Shared transport adapter for a single checksum-validated QNN or Genie artifact. */
class VendorAndroidTaskExecutor implements AndroidTaskExecutor {
    private final Context context;
    private final AndroidModelArtifact artifact;
    private final AndroidRuntimeBridge bridge;
    private final AndroidExecutionSampler sampler;

    VendorAndroidTaskExecutor(
            Context context, AndroidModelArtifact artifact, AndroidRuntimeBridge bridge) {
        this.context = context.getApplicationContext();
        this.artifact = artifact;
        this.bridge = bridge;
        this.sampler = new AndroidExecutionSampler(this.context);
    }

    @Override
    public TaskExecutionResult execute(ExecuteTask command) throws Exception {
        return run(new RuntimeExecutionRequest(
                command.getTaskId(), "", 0, command.getRequestText(), command.getSteering(),
                null, true, PipelineOperation.PIPELINE_OPERATION_UNSPECIFIED, 0, 0));
    }

    @Override
    public TaskExecutionResult executeShard(ExecuteShard command) throws Exception {
        return run(new RuntimeExecutionRequest(
                command.getTaskId(), "", 0, command.getRequestText(), command.getSteering(),
                null, true, PipelineOperation.PIPELINE_OPERATION_UNSPECIFIED, 0, 0));
    }

    @Override
    public TaskExecutionResult executePipelineStage(ExecutePipelineStage command) throws Exception {
        return run(new RuntimeExecutionRequest(
                command.getTaskId(),
                command.getPipelineId(),
                command.getStageIndex(),
                command.getRequestText(),
                command.getSteering(),
                command.hasInputBoundary() ? command.getInputBoundary() : null,
                command.getFinalStage(),
                command.getOperation(),
                command.getTokenId(),
                command.getMaxNewTokens()));
    }

    private TaskExecutionResult run(RuntimeExecutionRequest request) throws Exception {
        long start = System.nanoTime();
        AndroidExecutionSampler.Snapshot before = sampler.sample();
        RuntimeExecutionResult result = bridge.execute(context, artifact, request);
        AndroidExecutionSampler.Snapshot after = sampler.sample();
        long latency = Math.max(1L, (System.nanoTime() - start) / 1_000_000L);
        String accelerator = result.accelerator().isBlank()
                ? artifact.supportedAccelerators().get(0) : result.accelerator();
        ExecutionDetails details = AndroidExecutionSampler.withObservedDeltas(
                new ExecutionDetails(
                        artifact.modelId(),
                        artifact.modelVersion(),
                        artifact.runtime(),
                        bridge.runtimeVersion().isBlank()
                                ? artifact.runtimeVersion() : bridge.runtimeVersion(),
                        accelerator,
                        null,
                        null),
                before,
                after);
        if (request.operation() != PipelineOperation.PIPELINE_OPERATION_UNSPECIFIED) {
            return TaskExecutionResult.generated(result, latency, details, request.operation());
        }
        if (result.boundary() != null) {
            return TaskExecutionResult.boundary(result.boundary(), latency, details);
        }
        return TaskExecutionResult.success(result.outputText(), latency, details);
    }

    @Override
    public void cleanupTask(String taskId) {
        bridge.releaseTask(taskId);
    }
}

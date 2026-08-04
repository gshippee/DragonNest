package com.dragonnest.agent;

import com.dragonnest.proto.BoundaryTensor;
import com.dragonnest.proto.ExecutePipelineStage;
import com.dragonnest.proto.ExecuteShard;
import com.dragonnest.proto.ExecuteTask;
import com.google.protobuf.ByteString;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

public final class MockAndroidTaskExecutor implements AndroidTaskExecutor {
    public static final String MODEL_ID = "android-mock-v1";

    @Override
    public TaskExecutionResult execute(ExecuteTask command) {
        long start = System.nanoTime();
        String output = "[Android mock result from " + command.getModelId() + "] "
                + command.getRequestText() + steeringSuffix(command.getSteering().getEnabled());
        return TaskExecutionResult.success(output, elapsedMs(start), ExecutionDetails.mock());
    }

    @Override
    public TaskExecutionResult executeShard(ExecuteShard command) {
        long start = System.nanoTime();
        String output = "[Android mock shard " + command.getShardId() + "] "
                + command.getRequestText() + steeringSuffix(command.getSteering().getEnabled());
        return TaskExecutionResult.success(output, elapsedMs(start), ExecutionDetails.mock());
    }

    @Override
    public TaskExecutionResult executePipelineStage(ExecutePipelineStage command) throws Exception {
        long start = System.nanoTime();
        if (command.getFinalStage()) {
            String checksum = command.hasInputBoundary()
                    ? command.getInputBoundary().getChecksum() : "none";
            return TaskExecutionResult.success(
                    "[Android mock pipeline result] boundary=" + checksum,
                    elapsedMs(start),
                    ExecutionDetails.mock());
        }
        byte[] payload = (command.getTaskId() + ":" + command.getStageId()
                + ":" + command.getModelId()).getBytes(StandardCharsets.UTF_8);
        String checksum = "sha256:" + hex(
                MessageDigest.getInstance("SHA-256").digest(payload));
        BoundaryTensor boundary = BoundaryTensor.newBuilder()
                .setTensorName("hidden")
                .setDtype("uint8")
                .addShape(payload.length)
                .setData(ByteString.copyFrom(payload))
                .setChecksum(checksum)
                .build();
        return TaskExecutionResult.boundary(
                boundary, elapsedMs(start), ExecutionDetails.mock());
    }

    private static String steeringSuffix(boolean enabled) {
        return enabled ? " [steering enabled]" : "";
    }

    private static long elapsedMs(long start) {
        return Math.max(1L, (System.nanoTime() - start) / 1_000_000L);
    }

    private static String hex(byte[] value) {
        StringBuilder output = new StringBuilder(value.length * 2);
        for (byte item : value) {
            output.append(String.format("%02x", item & 0xff));
        }
        return output.toString();
    }
}

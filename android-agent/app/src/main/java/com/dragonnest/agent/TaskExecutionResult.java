package com.dragonnest.agent;

import com.dragonnest.proto.BoundaryTensor;

public record TaskExecutionResult(
        boolean success,
        String outputText,
        String errorCode,
        String errorMessage,
        long latencyMs,
        BoundaryTensor boundary) {
    public static TaskExecutionResult success(String output, long latencyMs) {
        return new TaskExecutionResult(true, output, "", "", latencyMs, null);
    }

    public static TaskExecutionResult boundary(BoundaryTensor value, long latencyMs) {
        return new TaskExecutionResult(true, "", "", "", latencyMs, value);
    }

    public static TaskExecutionResult failure(String code, String message, long latencyMs) {
        return new TaskExecutionResult(false, "", code, message, latencyMs, null);
    }
}

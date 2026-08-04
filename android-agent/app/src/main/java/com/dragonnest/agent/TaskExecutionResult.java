package com.dragonnest.agent;

import com.dragonnest.proto.BoundaryTensor;

public record TaskExecutionResult(
        boolean success,
        String outputText,
        String errorCode,
        String errorMessage,
        long latencyMs,
        BoundaryTensor boundary,
        ExecutionDetails details) {
    public static TaskExecutionResult success(
            String output, long latencyMs, ExecutionDetails details) {
        return new TaskExecutionResult(true, output, "", "", latencyMs, null, details);
    }

    public static TaskExecutionResult boundary(
            BoundaryTensor value, long latencyMs, ExecutionDetails details) {
        return new TaskExecutionResult(true, "", "", "", latencyMs, value, details);
    }

    public static TaskExecutionResult failure(
            String code, String message, long latencyMs, ExecutionDetails details) {
        return new TaskExecutionResult(false, "", code, message, latencyMs, null, details);
    }
}

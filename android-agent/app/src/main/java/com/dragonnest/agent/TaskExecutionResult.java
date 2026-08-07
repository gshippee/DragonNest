package com.dragonnest.agent;

import com.dragonnest.proto.BoundaryTensor;
import com.dragonnest.proto.PipelineOperation;

public record TaskExecutionResult(
        boolean success,
        String outputText,
        String errorCode,
        String errorMessage,
        long latencyMs,
        BoundaryTensor boundary,
        ExecutionDetails details,
        Integer nextTokenId,
        boolean eos,
        String tokenText,
        PipelineOperation operation) {
    public static TaskExecutionResult success(
            String output, long latencyMs, ExecutionDetails details) {
        return new TaskExecutionResult(true, output, "", "", latencyMs, null, details,
                null, false, "", PipelineOperation.PIPELINE_OPERATION_UNSPECIFIED);
    }

    public static TaskExecutionResult boundary(
            BoundaryTensor value, long latencyMs, ExecutionDetails details) {
        return new TaskExecutionResult(true, "", "", "", latencyMs, value, details,
                null, false, "", PipelineOperation.PIPELINE_OPERATION_UNSPECIFIED);
    }

    public static TaskExecutionResult failure(
            String code, String message, long latencyMs, ExecutionDetails details) {
        return new TaskExecutionResult(false, "", code, message, latencyMs, null, details,
                null, false, "", PipelineOperation.PIPELINE_OPERATION_UNSPECIFIED);
    }

    public static TaskExecutionResult generated(
            RuntimeExecutionResult result,
            long latencyMs,
            ExecutionDetails details,
            PipelineOperation operation) {
        return new TaskExecutionResult(
                true, result.outputText(), "", "", latencyMs, result.boundary(), details,
                result.nextTokenId(), result.eos(), result.tokenText(), operation);
    }
}

package com.dragonnest.agent;

import com.dragonnest.proto.BoundaryTensor;
import com.dragonnest.proto.PipelineOperation;
import com.dragonnest.proto.SteeringSpec;

/** Runtime-neutral request passed from the Agent transport to a vendor runtime bridge. */
public record RuntimeExecutionRequest(
        String taskId,
        String pipelineId,
        int stageIndex,
        String requestText,
        SteeringSpec steering,
        BoundaryTensor inputBoundary,
        boolean finalStage,
        PipelineOperation operation,
        int tokenId,
        int maxNewTokens) { }

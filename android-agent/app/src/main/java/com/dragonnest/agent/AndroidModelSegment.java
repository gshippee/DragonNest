package com.dragonnest.agent;

/** Compatibility data required to route a QNN model segment in a layer pipeline. */
public record AndroidModelSegment(
        String pipelineId,
        int startLayer,
        int endLayer,
        int totalLayers,
        boolean includesEmbedding,
        boolean includesLmHead,
        String boundaryFormat) {
    public AndroidModelSegment {
        pipelineId = pipelineId == null ? "" : pipelineId;
        boundaryFormat = boundaryFormat == null ? "" : boundaryFormat;
    }
}

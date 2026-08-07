package com.dragonnest.agent;

/** Compatibility data required to route a QNN model segment in a layer pipeline. */
public record AndroidModelSegment(
        String pipelineId,
        int stageIndex,
        int stageCount,
        Integer transformerStartLayer,
        Integer transformerEndLayer,
        int totalLayers,
        boolean includesEmbedding,
        boolean includesLmHead,
        String inputTensor,
        String outputTensor,
        String boundaryFormat) {
    public AndroidModelSegment {
        pipelineId = pipelineId == null ? "" : pipelineId;
        inputTensor = inputTensor == null ? "" : inputTensor;
        outputTensor = outputTensor == null ? "" : outputTensor;
        boundaryFormat = boundaryFormat == null ? "" : boundaryFormat;
    }
}

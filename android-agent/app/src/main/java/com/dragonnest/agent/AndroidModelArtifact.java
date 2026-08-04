package com.dragonnest.agent;

import com.dragonnest.proto.ModelCapability;
import com.dragonnest.proto.ModelSegment;

import java.nio.file.Path;
import java.util.List;

/** A checksum-validated Android model artifact and its routing metadata. */
public record AndroidModelArtifact(
        String modelId,
        String modelVersion,
        String runtime,
        Path artifactPath,
        String checksum,
        String tokenizerId,
        String precision,
        List<String> supportedAccelerators,
        long minMemoryMb,
        int maxContextTokens,
        boolean supportsSteering,
        boolean supportsDataParallel,
        boolean supportsLayerPipeline,
        String modelFamily,
        String role,
        List<String> taskClasses,
        float qualityScore,
        List<String> steeringVectorIds,
        List<Integer> supportedSteeringLayers,
        AndroidModelSegment segment,
        String runtimeVersion,
        String runtimeOptionsJson) {
    public AndroidModelArtifact {
        supportedAccelerators = List.copyOf(supportedAccelerators);
        taskClasses = List.copyOf(taskClasses);
        steeringVectorIds = List.copyOf(steeringVectorIds);
        supportedSteeringLayers = List.copyOf(supportedSteeringLayers);
        runtimeOptionsJson = runtimeOptionsJson == null ? "{}" : runtimeOptionsJson;
    }

    public ModelCapability capability() {
        ModelCapability.Builder capability = ModelCapability.newBuilder()
                .setModelId(modelId)
                .setModelFamily(modelFamily)
                .setRole(role)
                .addAllTaskClasses(taskClasses)
                .setMaxContextTokens(maxContextTokens)
                .setWarm(true)
                .setQualityScore(qualityScore)
                .setModelVersion(modelVersion)
                .setTokenizerId(tokenizerId)
                .setPrecision(precision)
                .setBoundaryFormat(segment == null ? "" : segment.boundaryFormat())
                .setRuntimeName(runtime)
                .setRuntimeVersion(runtimeVersion)
                .addAllSupportedAccelerators(supportedAccelerators)
                .setMinMemoryMb(minMemoryMb)
                .addAllSteeringVectorIds(steeringVectorIds)
                .addAllSupportedSteeringLayers(supportedSteeringLayers)
                .setSupportsSteering(supportsSteering)
                .setSupportsDataParallel(supportsDataParallel)
                .setSupportsLayerPipeline(supportsLayerPipeline);
        if (segment != null) {
            capability.setSegment(ModelSegment.newBuilder()
                    .setPipelineId(segment.pipelineId())
                    .setStartLayer(segment.startLayer())
                    .setEndLayer(segment.endLayer())
                    .setTotalLayers(segment.totalLayers())
                    .setIncludesEmbedding(segment.includesEmbedding())
                    .setIncludesLmHead(segment.includesLmHead()));
        }
        return capability.build();
    }
}

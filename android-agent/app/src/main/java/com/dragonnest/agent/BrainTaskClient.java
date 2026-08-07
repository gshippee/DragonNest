package com.dragonnest.agent;

import com.dragonnest.proto.BrainControlGrpc;
import com.dragonnest.proto.SteeringSpec;
import com.dragonnest.proto.SubmitTaskRequest;
import com.dragonnest.proto.SubmitTaskResponse;

import java.util.concurrent.TimeUnit;

import io.grpc.ManagedChannel;
import io.grpc.okhttp.OkHttpChannelBuilder;

/** Small client-side facade for the user query screen. */
public final class BrainTaskClient {
    private final AgentConfiguration configuration;

    public BrainTaskClient(AgentConfiguration configuration) {
        this.configuration = configuration;
    }

    /** Vector the profile slider drives; the only one calibrated for this model. */
    private static final String PROFILE_VECTOR_ID = "concise-vs-verbose-layer-7";
    private static final int PROFILE_INJECTION_LAYER = 7;

    public SubmitTaskResponse submit(
            String prompt,
            String personaId,
            boolean useProfileContext,
            String computePreference) {
        return submit(prompt, personaId, useProfileContext, computePreference, 0.0f);
    }

    /**
     * @param steeringAlpha explicit activation-steering strength from the
     *     profile slider. Zero means "use whatever the requested persona's
     *     calibrated realization is", which is the behaviour every caller had
     *     before the slider existed.
     */
    public SubmitTaskResponse submit(
            String prompt,
            String personaId,
            boolean useProfileContext,
            String computePreference,
            float steeringAlpha) {
        OkHttpChannelBuilder builder = OkHttpChannelBuilder.forAddress(
                configuration.brainHost(), configuration.brainPort());
        if (configuration.useTls()) {
            builder.useTransportSecurity();
        } else {
            builder.usePlaintext();
        }
        ManagedChannel channel = builder.build();
        try {
            SubmitTaskRequest.Builder request = SubmitTaskRequest.newBuilder()
                    .setRequestText(prompt.trim())
                    .setPreferredMode(ComputePreference.Companion
                            .fromWireValue(computePreference).getWireValue())
                    .setExecutionMode("auto")
                    .setOriginDeviceId(configuration.deviceId())
                    .setUseProfileSteering(true)
                    .setPersonaId(personaId)
                    .setUseProfileContext(useProfileContext);
            if (Math.abs(steeringAlpha) > 0.01f) {
                // Brain still decides whether any enrolled deployment can honour
                // this; an unsupported vector/layer/alpha fails closed there
                // rather than being quietly dropped here.
                request.setSteering(SteeringSpec.newBuilder()
                        .setEnabled(true)
                        .setMode("runtime_vector")
                        .setVectorId(PROFILE_VECTOR_ID)
                        .setTargetLayer(PROFILE_INJECTION_LAYER)
                        .setAlpha(steeringAlpha)
                        .setPositions("all")
                        .setModelFamily("qwen3")
                        .setBehaviorProfileId(personaId));
            }
            return BrainControlGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(90, TimeUnit.SECONDS)
                    .submitTask(request.build());
        } finally {
            channel.shutdownNow();
        }
    }
}

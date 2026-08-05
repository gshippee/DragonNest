package com.dragonnest.agent;

import com.dragonnest.proto.BrainControlGrpc;
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

    public SubmitTaskResponse submit(
            String prompt,
            String personaId,
            boolean useProfileContext,
            boolean keepOnPhone) {
        OkHttpChannelBuilder builder = OkHttpChannelBuilder.forAddress(
                configuration.brainHost(), configuration.brainPort());
        if (configuration.useTls()) {
            builder.useTransportSecurity();
        } else {
            builder.usePlaintext();
        }
        ManagedChannel channel = builder.build();
        try {
            return BrainControlGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(90, TimeUnit.SECONDS)
                    .submitTask(SubmitTaskRequest.newBuilder()
                            .setRequestText(prompt.trim())
                            .setPreferredMode(keepOnPhone ? "private" : "auto")
                            .setExecutionMode("auto")
                            .setOriginDeviceId(configuration.deviceId())
                            .setUseProfileSteering(true)
                            .setPersonaId(personaId)
                            .setUseProfileContext(useProfileContext)
                            .build());
        } finally {
            channel.shutdownNow();
        }
    }
}

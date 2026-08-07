package com.dragonnest.agent;

import com.dragonnest.proto.BrainControlGrpc;
import com.dragonnest.proto.BrainToDevice;
import com.dragonnest.proto.DeviceToBrain;
import com.dragonnest.proto.ExecutePipelineStage;
import com.dragonnest.proto.ExecuteShard;
import com.dragonnest.proto.ExecuteTask;
import com.dragonnest.proto.ExecutionMetrics;
import com.dragonnest.proto.HealthUpdate;
import com.dragonnest.proto.PartialTaskResult;
import com.dragonnest.proto.PipelineStageResult;
import com.dragonnest.proto.ShutdownEvent;
import com.dragonnest.proto.TaskResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import io.grpc.ManagedChannel;
import io.grpc.okhttp.OkHttpChannelBuilder;
import io.grpc.stub.StreamObserver;

public final class GrpcAgentConnection implements AgentConnection {
    private final AgentConfiguration configuration;
    private final AgentProfile profile;
    private final AndroidTaskExecutor taskExecutor;
    private final ClientDebugLog debugLog;
    private final ExecutorService taskPool = Executors.newCachedThreadPool();
    private final ScheduledExecutorService timeoutPool =
            Executors.newSingleThreadScheduledExecutor();
    private final Map<String, Future<?>> attempts = new ConcurrentHashMap<>();
    private final Map<String, AtomicInteger> activeTasks = new ConcurrentHashMap<>();
    private final AtomicBoolean connected = new AtomicBoolean(false);
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Object outboundLock = new Object();
    private volatile ManagedChannel channel;
    private volatile StreamObserver<DeviceToBrain> outbound;
    private volatile long heartbeatIntervalMs = 2_000;
    private volatile long heartbeatStartedNanos;
    private volatile float networkRttMs = TelemetrySnapshot.UNKNOWN;

    public GrpcAgentConnection(
            AgentConfiguration configuration,
            AgentProfile profile,
            AndroidTaskExecutor taskExecutor,
            ClientDebugLog debugLog) {
        this.configuration = configuration;
        this.profile = profile;
        this.taskExecutor = taskExecutor;
        this.debugLog = debugLog;
    }

    @Override
    public String connect(String enrollmentCredential) throws Exception {
        debugLog.add("Opening gRPC stream to " + configuration.brainHost()
                + ":" + configuration.brainPort()
                + (configuration.useTls() ? " (TLS)" : ""));
        CountDownLatch registration = new CountDownLatch(1);
        AtomicReference<String> rejection = new AtomicReference<>("");
        AtomicReference<String> replacementCredential = new AtomicReference<>("");
        OkHttpChannelBuilder builder = OkHttpChannelBuilder.forAddress(
                configuration.brainHost(), configuration.brainPort());
        if (configuration.useTls()) {
            builder.useTransportSecurity();
        } else {
            builder.usePlaintext();
        }
        channel = builder
                .keepAliveTime(10, TimeUnit.SECONDS)
                .keepAliveTimeout(5, TimeUnit.SECONDS)
                .build();
        BrainControlGrpc.BrainControlStub stub = BrainControlGrpc.newStub(channel);
        outbound = stub.connect(new StreamObserver<>() {
            @Override
            public void onNext(BrainToDevice message) {
                switch (message.getPayloadCase()) {
                    case REGISTRATION_ACCEPTED -> {
                        replacementCredential.set(
                                message.getRegistrationAccepted().getDeviceCredential());
                        heartbeatIntervalMs = Math.max(
                                250,
                                message.getRegistrationAccepted().getHeartbeatIntervalMs());
                        connected.set(true);
                        debugLog.add("Brain returned RegistrationAccepted");
                        registration.countDown();
                    }
                    case REGISTRATION_REJECTED -> {
                        rejection.set(message.getRegistrationRejected().getReason());
                        debugLog.add("Brain rejected registration: "
                                + message.getRegistrationRejected().getReason());
                        registration.countDown();
                    }
                    case EXECUTE_TASK -> dispatch(message.getExecuteTask());
                    case EXECUTE_SHARD -> dispatch(message.getExecuteShard());
                    case EXECUTE_PIPELINE_STAGE ->
                            dispatch(message.getExecutePipelineStage());
                    case CANCEL_TASK -> cancel(message.getCancelTask().getAttemptId());
                    case HEARTBEAT_ACK -> {
                        if (heartbeatStartedNanos > 0) {
                            networkRttMs = Math.max(
                                    0,
                                    (System.nanoTime() - heartbeatStartedNanos) / 1_000_000f);
                        }
                    }
                    case PAYLOAD_NOT_SET -> { }
                }
            }

            @Override
            public void onError(Throwable failure) {
                rejection.compareAndSet("", failure.getMessage());
                debugLog.add("gRPC stream error: " + failureSummary(failure));
                connected.set(false);
                registration.countDown();
            }

            @Override
            public void onCompleted() {
                debugLog.add("gRPC stream completed by Brain");
                connected.set(false);
                registration.countDown();
            }
        });
        send(DeviceToBrain.newBuilder()
                .setRegisterDevice(profile.registration(enrollmentCredential))
                .build());
        if (!registration.await(10, TimeUnit.SECONDS)) {
            close();
            throw new TimeoutException("Brain registration timed out");
        }
        if (!connected.get()) {
            close();
            throw new IllegalStateException("Brain rejected registration: " + rejection.get());
        }
        return replacementCredential.get();
    }

    private static String failureSummary(Throwable failure) {
        String detail = failure.getMessage();
        return failure.getClass().getSimpleName()
                + (detail == null || detail.isBlank() ? "" : ": " + detail);
    }

    @Override
    public void sendHeartbeat(TelemetrySnapshot telemetry) {
        HealthUpdate.Builder health = HealthUpdate.newBuilder()
                .setDeviceId(configuration.deviceId())
                .setTimestampMs(System.currentTimeMillis())
                .setBatteryPct(telemetry.batteryPercentage())
                .setCharging(telemetry.charging())
                .setThermalLevel(telemetry.thermalLevel())
                .setCpuUtilization(telemetry.cpuUtilization())
                .setAcceleratorUtilization(telemetry.acceleratorUtilization())
                .setGpuUtilization(telemetry.gpuUtilization())
                .setNpuUtilization(telemetry.npuUtilization())
                .setAvailableMemoryMb(telemetry.availableMemoryMb())
                .setNetworkRttMs(networkRttMs >= 0 ? networkRttMs : telemetry.networkRttMs())
                .setReachable(telemetry.reachable())
                .setSimulatedConstraint(telemetry.simulated())
                .addAllActiveTaskIds(telemetry.activeTaskIds())
                .addAllWarmModelIds(telemetry.warmModelIds());
        heartbeatStartedNanos = System.nanoTime();
        send(DeviceToBrain.newBuilder().setHealthUpdate(health).build());
    }

    @Override
    public void sendShutdown(String reason) {
        if (outbound == null || closed.get()) {
            return;
        }
        send(DeviceToBrain.newBuilder()
                .setShutdown(ShutdownEvent.newBuilder()
                        .setDeviceId(configuration.deviceId())
                        .setReason(reason))
                .build());
    }

    @Override
    public boolean isConnected() {
        return connected.get() && !closed.get();
    }

    @Override
    public List<String> activeTaskIds() {
        return new ArrayList<>(activeTasks.keySet());
    }

    @Override
    public List<String> warmModelIds() {
        return profile.warmModelIds();
    }

    @Override
    public long heartbeatIntervalMs() {
        return heartbeatIntervalMs;
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        connected.set(false);
        for (Future<?> attempt : attempts.values()) {
            attempt.cancel(true);
        }
        attempts.clear();
        if (outbound != null) {
            synchronized (outboundLock) {
                outbound.onCompleted();
            }
        }
        taskPool.shutdownNow();
        timeoutPool.shutdownNow();
        if (channel != null) {
            channel.shutdown();
            try {
                channel.awaitTermination(1, TimeUnit.SECONDS);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void dispatch(ExecuteTask command) {
        submitAttempt(command.getTaskId(), command.getAttemptId(), command.getTimeoutMs(), () -> {
            TaskExecutionResult result;
            long start = System.nanoTime();
            try {
                result = taskExecutor.execute(command);
            } catch (Exception failure) {
                result = executionFailure(failure, start);
            }
            TaskResult response = TaskResult.newBuilder()
                    .setTaskId(command.getTaskId())
                    .setAttemptId(command.getAttemptId())
                    .setDeviceId(configuration.deviceId())
                    .setSuccess(result.success())
                    .setOutputText(result.outputText())
                    .setErrorCode(result.errorCode())
                    .setErrorMessage(result.errorMessage())
                    .setMetrics(metrics(command.getModelId(), result))
                    .build();
            send(DeviceToBrain.newBuilder().setTaskResult(response).build());
        });
    }

    private void dispatch(ExecuteShard command) {
        submitAttempt(command.getTaskId(), command.getAttemptId(), command.getTimeoutMs(), () -> {
            TaskExecutionResult result;
            long start = System.nanoTime();
            try {
                result = taskExecutor.executeShard(command);
            } catch (Exception failure) {
                result = executionFailure(failure, start);
            }
            PartialTaskResult response = PartialTaskResult.newBuilder()
                    .setTaskId(command.getTaskId())
                    .setAttemptId(command.getAttemptId())
                    .setShardId(command.getShardId())
                    .setDeviceId(configuration.deviceId())
                    .setSuccess(result.success())
                    .setOutputText(result.outputText())
                    .setErrorCode(result.errorCode())
                    .setErrorMessage(result.errorMessage())
                    .setMetrics(metrics(command.getModelId(), result))
                    .build();
            send(DeviceToBrain.newBuilder().setPartialTaskResult(response).build());
        });
    }

    private void dispatch(ExecutePipelineStage command) {
        submitAttempt(command.getTaskId(), command.getAttemptId(), command.getTimeoutMs(), () -> {
            TaskExecutionResult result;
            long start = System.nanoTime();
            try {
                result = taskExecutor.executePipelineStage(command);
            } catch (Exception failure) {
                result = executionFailure(failure, start);
            }
            PipelineStageResult.Builder response = PipelineStageResult.newBuilder()
                    .setTaskId(command.getTaskId())
                    .setAttemptId(command.getAttemptId())
                    .setStageId(command.getStageId())
                    .setDeviceId(configuration.deviceId())
                    .setSuccess(result.success())
                    .setOutputText(result.outputText())
                    .setErrorCode(result.errorCode())
                    .setErrorMessage(result.errorMessage())
                    .setMetrics(metrics(command.getModelId(), result));
            if (result.boundary() != null) {
                response.setOutputBoundary(result.boundary());
            }
            send(DeviceToBrain.newBuilder().setPipelineStageResult(response).build());
        });
    }

    private void submitAttempt(
            String taskId,
            String attemptId,
            long timeoutMs,
            AttemptWork work) {
        activeTasks.compute(taskId, (key, count) -> {
            if (count == null) {
                return new AtomicInteger(1);
            }
            count.incrementAndGet();
            return count;
        });
        FutureTask<Void> attempt = new FutureTask<>(() -> {
            try {
                work.run();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            } catch (Exception failure) {
                sendExecutionFailure(taskId, attemptId, failure);
            } finally {
                attempts.remove(attemptId);
                activeTasks.computeIfPresent(taskId, (key, count) ->
                        count.decrementAndGet() <= 0 ? null : count);
            }
            return null;
        });
        attempts.put(attemptId, attempt);
        taskPool.execute(attempt);
        if (timeoutMs > 0) {
            timeoutPool.schedule(() -> cancel(attemptId), timeoutMs, TimeUnit.MILLISECONDS);
        }
    }

    private void cancel(String attemptId) {
        Future<?> attempt = attempts.remove(attemptId);
        if (attempt != null) {
            attempt.cancel(true);
        }
    }

    private void sendExecutionFailure(String taskId, String attemptId, Exception failure) {
        if (Thread.currentThread().isInterrupted()) {
            return;
        }
        TaskResult result = TaskResult.newBuilder()
                .setTaskId(taskId)
                .setAttemptId(attemptId)
                .setDeviceId(configuration.deviceId())
                .setSuccess(false)
                .setErrorCode("EXECUTION_FAILED")
                .setErrorMessage(String.valueOf(failure.getMessage()))
                .build();
        send(DeviceToBrain.newBuilder().setTaskResult(result).build());
    }

    private ExecutionMetrics metrics(String modelId, TaskExecutionResult result) {
        ExecutionDetails details = result.details() == null
                ? new ExecutionDetails(modelId, "", "unknown", "", "", null, null)
                : result.details();
        ExecutionMetrics.Builder metrics = ExecutionMetrics.newBuilder()
                .setModelId(details.modelId().isEmpty() ? modelId : details.modelId())
                .setModelVersion(details.modelVersion())
                .setRuntimeName(details.runtimeName())
                .setRuntimeVersion(details.runtimeVersion())
                .setAccelerator(details.accelerator())
                .setExecutionLatencyMs((int) Math.min(result.latencyMs(), Integer.MAX_VALUE))
                .setErrorCode(result.errorCode())
                .setErrorMessage(result.errorMessage());
        if (details.observedMemoryDeltaMb() != null) {
            metrics.setObservedMemoryDeltaMb(details.observedMemoryDeltaMb());
        }
        if (details.observedThermalDelta() != null) {
            metrics.setObservedThermalDelta(details.observedThermalDelta());
        }
        return metrics.build();
    }

    private static TaskExecutionResult executionFailure(Exception failure, long start) {
        long latency = Math.max(1L, (System.nanoTime() - start) / 1_000_000L);
        return TaskExecutionResult.failure(
                "EXECUTION_FAILED",
                String.valueOf(failure.getMessage()),
                latency,
                null);
    }

    private void send(DeviceToBrain message) {
        if (outbound == null || closed.get()) {
            throw new IllegalStateException("gRPC stream is not available");
        }
        synchronized (outboundLock) {
            outbound.onNext(message);
        }
    }

    @FunctionalInterface
    private interface AttemptWork {
        void run() throws Exception;
    }
}

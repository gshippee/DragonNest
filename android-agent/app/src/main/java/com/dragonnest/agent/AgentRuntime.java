package com.dragonnest.agent;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public final class AgentRuntime {
    private static final long MAX_BACKOFF_SECONDS = 60;
    private final ScheduledExecutorService executor =
            Executors.newSingleThreadScheduledExecutor();
    private final Supplier<AgentConnection> connectionFactory;
    private final EnrollmentStore enrollmentStore;
    private final AndroidTelemetry telemetry;
    private final ClientDebugLog debugLog;
    private volatile AgentConnection connection;
    private volatile boolean stopping;
    private volatile ScheduledFuture<?> heartbeatFuture;
    private long reconnectBackoffSeconds = 1;

    public AgentRuntime(
            Supplier<AgentConnection> connectionFactory,
            EnrollmentStore enrollmentStore,
            AndroidTelemetry telemetry,
            ClientDebugLog debugLog) {
        this.connectionFactory = connectionFactory;
        this.enrollmentStore = enrollmentStore;
        this.telemetry = telemetry;
        this.debugLog = debugLog;
    }

    public void start() {
        debugLog.add("Agent runtime started");
        AgentStatusRepository.update(AgentConnectionState.CONNECTING, "Connecting to DragonNest");
        executor.execute(this::connect);
    }

    public void onNetworkChanged() {
        debugLog.add("Network state changed");
        executor.execute(() -> {
            if (connection != null && connection.isConnected()) {
                sendHeartbeat();
            } else {
                connect();
            }
        });
    }

    public void stop() {
        stopping = true;
        debugLog.add("Agent runtime stopping");
        AgentStatusRepository.update(AgentConnectionState.STOPPED, "Agent stopped");
        executor.execute(() -> {
            if (connection != null) {
                try {
                    connection.sendShutdown("android_service_stopped");
                } catch (Exception ignored) {
                    // Shutdown is best-effort; the Brain heartbeat timeout is authoritative.
                } finally {
                    connection.close();
                }
            }
            executor.shutdownNow();
        });
    }

    private void connect() {
        if (stopping || (connection != null && connection.isConnected())) {
            return;
        }
        AgentConnection next = null;
        try {
            debugLog.add("Connecting to Brain");
            AgentStatusRepository.update(AgentConnectionState.CONNECTING, "Connecting to DragonNest");
            next = connectionFactory.get();
            String replacementCredential = next.connect(enrollmentStore.load());
            if (replacementCredential != null && !replacementCredential.isBlank()) {
                enrollmentStore.save(replacementCredential);
            }
            connection = next;
            reconnectBackoffSeconds = 1;
            debugLog.add("Brain registration accepted");
            AgentStatusRepository.update(AgentConnectionState.CONNECTED, "Connected through DragonNest");
            sendHeartbeat();
        } catch (Exception failure) {
            debugLog.add("Brain connection failed: " + failureSummary(failure));
            AgentStatusRepository.update(AgentConnectionState.RETRYING, failureSummary(failure));
            if (next != null) {
                next.close();
            }
            scheduleReconnect();
        }
    }

    private void sendHeartbeat() {
        if (stopping || connection == null || !connection.isConnected()) {
            scheduleReconnect();
            return;
        }
        try {
            connection.sendHeartbeat(telemetry.sample(
                    connection.activeTaskIds(),
                    connection.warmModelIds()));
            if (heartbeatFuture != null) {
                heartbeatFuture.cancel(false);
            }
            heartbeatFuture = executor.schedule(
                    this::sendHeartbeat,
                    connection.heartbeatIntervalMs(),
                    TimeUnit.MILLISECONDS);
        } catch (Exception failure) {
            connection.close();
            connection = null;
            scheduleReconnect();
        }
    }

    private void scheduleReconnect() {
        if (stopping) {
            return;
        }
        long delay = reconnectBackoffSeconds;
        debugLog.add("Retrying Brain connection in " + delay + "s");
        AgentStatusRepository.update(
                AgentConnectionState.RETRYING,
                "Retrying in " + delay + "s");
        reconnectBackoffSeconds = Math.min(reconnectBackoffSeconds * 2, MAX_BACKOFF_SECONDS);
        executor.schedule(this::connect, delay, TimeUnit.SECONDS);
    }

    private static String failureSummary(Exception failure) {
        String detail = failure.getMessage();
        return failure.getClass().getSimpleName()
                + (detail == null || detail.isBlank() ? "" : ": " + detail);
    }
}

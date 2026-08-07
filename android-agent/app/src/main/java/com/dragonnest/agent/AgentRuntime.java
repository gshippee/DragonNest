package com.dragonnest.agent;

import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Supplier;

public final class AgentRuntime {
    private static final long MAX_BACKOFF_SECONDS = 60;
    private static final long SIMULATION_HEARTBEAT_DELAY_MS = 50;
    private final ScheduledExecutorService executor;
    private final Supplier<AgentConnection> connectionFactory;
    private final CredentialLoader credentialLoader;
    private final CredentialSaver credentialSaver;
    private final TelemetrySampler telemetrySampler;
    private final Consumer<String> debugLogger;
    private final AtomicBoolean simulationHeartbeatQueued = new AtomicBoolean();
    private volatile AgentConnection connection;
    private volatile boolean stopping;
    private volatile ScheduledFuture<?> heartbeatFuture;
    private long reconnectBackoffSeconds = 1;

    public AgentRuntime(
            Supplier<AgentConnection> connectionFactory,
            EnrollmentStore enrollmentStore,
            AndroidTelemetry telemetry,
            ClientDebugLog debugLog) {
        this(
                connectionFactory,
                enrollmentStore::load,
                enrollmentStore::save,
                telemetry::sample,
                debugLog::add,
                Executors.newSingleThreadScheduledExecutor());
    }

    AgentRuntime(
            Supplier<AgentConnection> connectionFactory,
            CredentialLoader credentialLoader,
            CredentialSaver credentialSaver,
            TelemetrySampler telemetrySampler,
            Consumer<String> debugLogger,
            ScheduledExecutorService executor) {
        this.connectionFactory = connectionFactory;
        this.credentialLoader = credentialLoader;
        this.credentialSaver = credentialSaver;
        this.telemetrySampler = telemetrySampler;
        this.debugLogger = debugLogger;
        this.executor = executor;
    }

    public void start() {
        debugLogger.accept("Agent runtime started");
        AgentStatusRepository.update(AgentConnectionState.CONNECTING, "Connecting to DragonNest");
        executor.execute(this::connect);
    }

    public void onNetworkChanged() {
        debugLogger.accept("Network state changed");
        executor.execute(() -> {
            if (connection != null && connection.isConnected()) {
                sendHeartbeat();
            } else {
                connect();
            }
        });
    }

    /**
     * Publishes changed demo telemetry promptly without replacing the active
     * gRPC stream. Rapid slider events share one pending executor task.
     */
    public void onSimulationChanged() {
        if (stopping || !simulationHeartbeatQueued.compareAndSet(false, true)) {
            return;
        }
        try {
            executor.schedule(() -> {
                simulationHeartbeatQueued.set(false);
                if (!stopping && connection != null && connection.isConnected()) {
                    sendHeartbeat();
                }
            }, SIMULATION_HEARTBEAT_DELAY_MS, TimeUnit.MILLISECONDS);
        } catch (RejectedExecutionException ignored) {
            simulationHeartbeatQueued.set(false);
        }
    }

    public void stop() {
        stopping = true;
        debugLogger.accept("Agent runtime stopping");
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
            debugLogger.accept("Connecting to Brain");
            AgentStatusRepository.update(AgentConnectionState.CONNECTING, "Connecting to DragonNest");
            next = connectionFactory.get();
            String replacementCredential = next.connect(credentialLoader.load());
            if (replacementCredential != null && !replacementCredential.isBlank()) {
                credentialSaver.save(replacementCredential);
            }
            connection = next;
            reconnectBackoffSeconds = 1;
            debugLogger.accept("Brain registration accepted");
            AgentStatusRepository.update(AgentConnectionState.CONNECTED, "Connected through DragonNest");
            sendHeartbeat();
        } catch (Exception failure) {
            debugLogger.accept("Brain connection failed: " + failureSummary(failure));
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
            connection.sendHeartbeat(telemetrySampler.sample(
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
        debugLogger.accept("Retrying Brain connection in " + delay + "s");
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

    @FunctionalInterface
    interface CredentialLoader {
        String load() throws Exception;
    }

    @FunctionalInterface
    interface CredentialSaver {
        void save(String credential) throws Exception;
    }

    @FunctionalInterface
    interface TelemetrySampler {
        TelemetrySnapshot sample(
                List<String> activeTasks,
                List<String> warmModels);
    }
}

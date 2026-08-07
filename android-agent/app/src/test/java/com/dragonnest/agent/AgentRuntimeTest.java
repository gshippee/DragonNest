package com.dragonnest.agent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public final class AgentRuntimeTest {
    @Test
    public void simulationChangeSendsFreshTelemetryWithoutReconnect() throws Exception {
        FakeConnection connection = new FakeConnection();
        AtomicLong availableMemoryMb = new AtomicLong(6_000);
        AgentRuntime runtime = new AgentRuntime(
                () -> connection,
                () -> "device-credential",
                ignored -> { },
                (activeTasks, warmModels) -> snapshot(
                        availableMemoryMb.get(), activeTasks, warmModels),
                ignored -> { },
                Executors.newSingleThreadScheduledExecutor());

        try {
            runtime.start();
            TelemetrySnapshot initial = connection.heartbeats.poll(2, TimeUnit.SECONDS);
            assertNotNull("initial heartbeat was not sent", initial);
            assertEquals(6_000, initial.availableMemoryMb());

            availableMemoryMb.set(64);
            for (int update = 0; update < 20; update++) {
                runtime.onSimulationChanged();
            }

            TelemetrySnapshot refreshed = connection.heartbeats.poll(1, TimeUnit.SECONDS);
            assertNotNull("simulation heartbeat was not sent promptly", refreshed);
            assertEquals(64, refreshed.availableMemoryMb());
            assertTrue(refreshed.simulated());
            assertEquals("simulation update must reuse the gRPC stream", 1, connection.connectCalls.get());
            assertNull(
                    "slider burst should be coalesced into one immediate heartbeat",
                    connection.heartbeats.poll(200, TimeUnit.MILLISECONDS));
        } finally {
            runtime.stop();
            assertTrue(connection.shutdownSent.await(2, TimeUnit.SECONDS));
        }
    }

    private static TelemetrySnapshot snapshot(
            long availableMemoryMb,
            List<String> activeTasks,
            List<String> warmModels) {
        return new TelemetrySnapshot(
                80,
                false,
                0.2f,
                availableMemoryMb,
                0.1f,
                0.1f,
                0.1f,
                TelemetrySnapshot.UNKNOWN,
                5,
                List.copyOf(activeTasks),
                List.copyOf(warmModels),
                true,
                true);
    }

    private static final class FakeConnection implements AgentConnection {
        private final AtomicInteger connectCalls = new AtomicInteger();
        private final LinkedBlockingQueue<TelemetrySnapshot> heartbeats =
                new LinkedBlockingQueue<>();
        private final java.util.concurrent.CountDownLatch shutdownSent =
                new java.util.concurrent.CountDownLatch(1);
        private volatile boolean connected;

        @Override
        public String connect(String enrollmentCredential) {
            connectCalls.incrementAndGet();
            connected = true;
            return "";
        }

        @Override
        public void sendHeartbeat(TelemetrySnapshot telemetry) {
            heartbeats.add(telemetry);
        }

        @Override
        public void sendShutdown(String reason) {
            shutdownSent.countDown();
        }

        @Override
        public boolean isConnected() {
            return connected;
        }

        @Override
        public List<String> activeTaskIds() {
            return List.of();
        }

        @Override
        public List<String> warmModelIds() {
            return List.of("android-mock-v1");
        }

        @Override
        public long heartbeatIntervalMs() {
            return 60_000;
        }

        @Override
        public void close() {
            connected = false;
        }
    }
}

package com.dragonnest.agent;

import java.util.List;

public interface AgentConnection extends AutoCloseable {
    void connect(String enrollmentCredential) throws Exception;

    void sendHeartbeat(TelemetrySnapshot telemetry) throws Exception;

    void sendShutdown(String reason) throws Exception;

    boolean isConnected();

    List<String> activeTaskIds();

    List<String> warmModelIds();

    long heartbeatIntervalMs();

    @Override
    void close();
}

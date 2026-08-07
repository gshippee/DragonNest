package com.dragonnest.agent;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.net.ConnectivityManager;
import android.net.Network;
import android.os.IBinder;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class AgentForegroundService extends Service {
    public static final String ACTION_RELOAD = "com.dragonnest.agent.RELOAD";
    public static final String ACTION_UPDATE_SIMULATION = "com.dragonnest.agent.UPDATE_SIMULATION";
    private static final String CHANNEL_ID = "dragonnest-agent";
    private static final int NOTIFICATION_ID = 4101;
    private volatile AgentRuntime runtime;
    private AgentConfiguration configuration;
    private AndroidTelemetry telemetry;
    private ClientDebugLog debugLog;
    private ConnectivityManager connectivityManager;
    private ConnectivityManager.NetworkCallback networkCallback;
    private ExecutorService runtimeBootstrap;
    private volatile boolean runtimeStarting;
    private volatile boolean destroyed;

    @Override
    public void onCreate() {
        super.onCreate();
        startForeground(NOTIFICATION_ID, createNotification());
        configuration = new AgentConfiguration(this);
        telemetry = new AndroidTelemetry(this);
        debugLog = new ClientDebugLog(this);
        debugLog.add("Foreground service created");
        runtimeBootstrap = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "dragonnest-runtime-bootstrap");
            thread.setDaemon(true);
            return thread;
        });

        connectivityManager = getSystemService(ConnectivityManager.class);
        networkCallback = new ConnectivityManager.NetworkCallback() {
            @Override
            public void onAvailable(Network network) {
                if (runtime != null) {
                    runtime.onNetworkChanged();
                }
            }

            @Override
            public void onLost(Network network) {
                if (runtime != null) {
                    runtime.onNetworkChanged();
                }
            }
        };
        connectivityManager.registerDefaultNetworkCallback(networkCallback);
    }

    private void startRuntime() {
        runtimeStarting = true;
        runtimeBootstrap.execute(() -> {
            try {
                AndroidRuntimeCatalog runtimeCatalog = AndroidRuntimeCatalog.create(this);
                if (destroyed) {
                    return;
                }
                AgentProfile profile = new AgentProfile(this, configuration, runtimeCatalog);
                AgentRuntime created = new AgentRuntime(
                        () -> new GrpcAgentConnection(
                                configuration,
                                profile,
                                runtimeCatalog,
                                debugLog),
                        new EnrollmentStore(this),
                        telemetry,
                        debugLog);
                runtime = created;
                created.start();
            } catch (RuntimeException failure) {
                debugLog.add("Runtime bootstrap failed: " + failure.getMessage());
            } finally {
                runtimeStarting = false;
            }
        });
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_UPDATE_SIMULATION.equals(intent.getAction())) {
            telemetry.setSimulation(configuration.simulation());
            if (runtime != null) {
                runtime.onSimulationChanged();
            }
            debugLog.add("Simulation updated without reconnecting");
            return START_STICKY;
        }
        if (intent != null && ACTION_RELOAD.equals(intent.getAction())) {
            if (runtime != null) {
                runtime.stop();
                runtime = null;
            }
        }
        if (!new EnrollmentStore(this).hasCredential()
                || new UserProfileStore(this).load() == null) {
            debugLog.add("Agent not started: enrollment or profile is missing");
            stopSelf();
            return START_NOT_STICKY;
        }
        if (runtime == null && !runtimeStarting) {
            debugLog.add("Starting device agent");
            telemetry.setSimulation(configuration.simulation());
            startRuntime();
        }
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        destroyed = true;
        if (connectivityManager != null && networkCallback != null) {
            connectivityManager.unregisterNetworkCallback(networkCallback);
        }
        if (runtime != null) {
            runtime.stop();
        }
        if (runtimeBootstrap != null) {
            runtimeBootstrap.shutdownNow();
        }
        if (debugLog != null) {
            debugLog.add("Foreground service stopped");
        }
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private Notification createNotification() {
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(
                CHANNEL_ID,
                "DragonNest",
                NotificationManager.IMPORTANCE_LOW));
        return new Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("PersonaCare is ready")
                .setContentText("Connected through DragonNest")
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setOngoing(true)
                .build();
    }

}

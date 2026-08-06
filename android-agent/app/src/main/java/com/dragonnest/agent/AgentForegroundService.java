package com.dragonnest.agent;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.net.ConnectivityManager;
import android.net.Network;
import android.os.IBinder;

public final class AgentForegroundService extends Service {
    public static final String ACTION_RELOAD = "com.dragonnest.agent.RELOAD";
    public static final String ACTION_UPDATE_SIMULATION = "com.dragonnest.agent.UPDATE_SIMULATION";
    private static final String CHANNEL_ID = "dragonnest-agent";
    private static final int NOTIFICATION_ID = 4101;
    private AgentRuntime runtime;
    private AgentConfiguration configuration;
    private AndroidTelemetry telemetry;
    private ClientDebugLog debugLog;
    private ConnectivityManager connectivityManager;
    private ConnectivityManager.NetworkCallback networkCallback;

    @Override
    public void onCreate() {
        super.onCreate();
        startForeground(NOTIFICATION_ID, createNotification());
        configuration = new AgentConfiguration(this);
        telemetry = new AndroidTelemetry(this);
        debugLog = new ClientDebugLog(this);
        debugLog.add("Foreground service created");

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
        AndroidRuntimeCatalog runtimeCatalog = AndroidRuntimeCatalog.create(this);
        AgentProfile profile = new AgentProfile(this, configuration, runtimeCatalog);
        runtime = new AgentRuntime(
                () -> new GrpcAgentConnection(
                        configuration,
                        profile,
                        runtimeCatalog,
                        debugLog),
                new EnrollmentStore(this),
                telemetry,
                debugLog);
        runtime.start();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_UPDATE_SIMULATION.equals(intent.getAction())) {
            telemetry.setSimulation(configuration.simulation());
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
        if (runtime == null) {
            debugLog.add("Starting device agent");
            telemetry.setSimulation(configuration.simulation());
            startRuntime();
        }
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        if (connectivityManager != null && networkCallback != null) {
            connectivityManager.unregisterNetworkCallback(networkCallback);
        }
        if (runtime != null) {
            runtime.stop();
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

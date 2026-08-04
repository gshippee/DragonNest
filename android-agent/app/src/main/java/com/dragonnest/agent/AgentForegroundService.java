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
    private static final String CHANNEL_ID = "dragonnest-agent";
    private static final int NOTIFICATION_ID = 4101;
    private AgentRuntime runtime;
    private AgentConfiguration configuration;
    private AndroidTelemetry telemetry;
    private ConnectivityManager connectivityManager;
    private ConnectivityManager.NetworkCallback networkCallback;

    @Override
    public void onCreate() {
        super.onCreate();
        startForeground(NOTIFICATION_ID, createNotification());
        configuration = new AgentConfiguration(this);
        telemetry = new AndroidTelemetry(this);

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
                        runtimeCatalog),
                new EnrollmentStore(this),
                telemetry);
        runtime.start();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_RELOAD.equals(intent.getAction())) {
            if (runtime != null) {
                runtime.stop();
                runtime = null;
            }
        }
        if (runtime == null) {
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
                "DragonNest Agent",
                NotificationManager.IMPORTANCE_LOW));
        return new Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("DragonNest Agent")
                .setContentText("Connected device agent is running")
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setOngoing(true)
                .build();
    }

}

package com.dragonnest.agent;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

public final class BootReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        if (Intent.ACTION_BOOT_COMPLETED.equals(intent.getAction())) {
            if (new EnrollmentStore(context).hasCredential()
                    && new UserProfileStore(context).load() != null) {
                context.startForegroundService(new Intent(context, AgentForegroundService.class));
            }
        }
    }
}

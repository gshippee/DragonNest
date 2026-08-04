package com.dragonnest.agent;

import android.content.Context;

/** Genie-specific executor selected only after artifact and vendor bridge validation. */
public final class GenieAndroidTaskExecutor extends VendorAndroidTaskExecutor {
    public GenieAndroidTaskExecutor(
            Context context, AndroidModelArtifact artifact, AndroidRuntimeBridge bridge) {
        super(context, artifact, bridge);
    }
}

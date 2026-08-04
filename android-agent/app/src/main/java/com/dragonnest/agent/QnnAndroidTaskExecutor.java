package com.dragonnest.agent;

import android.content.Context;

/** QNN-specific executor selected only after artifact and vendor bridge validation. */
public final class QnnAndroidTaskExecutor extends VendorAndroidTaskExecutor {
    public QnnAndroidTaskExecutor(
            Context context, AndroidModelArtifact artifact, AndroidRuntimeBridge bridge) {
        super(context, artifact, bridge);
    }
}

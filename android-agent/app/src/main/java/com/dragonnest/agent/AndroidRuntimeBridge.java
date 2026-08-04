package com.dragonnest.agent;

import android.content.Context;

/**
 * API implemented by the QNN or Genie vendor AAR packaged into a release build.
 *
 * The base Agent intentionally does not link Qualcomm's licensed libraries. A
 * bridge owns those JNI bindings and can therefore be substituted as Qualcomm's
 * SDK API changes without changing the gRPC protocol.
 */
public interface AndroidRuntimeBridge {
    String runtimeName();

    String runtimeVersion();

    /** Must load/probe the model and return false when it cannot execute on this device. */
    boolean isAvailable(Context context, AndroidModelArtifact artifact);

    RuntimeExecutionResult execute(
            Context context,
            AndroidModelArtifact artifact,
            RuntimeExecutionRequest request) throws Exception;
}

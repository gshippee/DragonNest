package com.dragonnest.agent.vendor;

import android.content.Context;

import com.dragonnest.agent.AndroidModelArtifact;
import com.dragonnest.agent.AndroidRuntimeBridge;
import com.dragonnest.agent.RuntimeExecutionRequest;
import com.dragonnest.agent.RuntimeExecutionResult;
import com.dragonnest.proto.BoundaryTensor;
import com.dragonnest.proto.PipelineOperation;
import com.google.protobuf.ByteString;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Direct QAIRT/QNN context-binary bridge for indexed pipeline stages.
 *
 * Native handles are keyed by task/pipeline/stage, so KV tensors remain with
 * the stage that owns them. Only the named activation boundary is returned to
 * DragonNest. The open-source build contains no Qualcomm binaries; the JNI
 * target is enabled only by the existing explicit QAIRT hardware-build flag.
 */
public final class QnnRuntimeBridge implements AndroidRuntimeBridge {
    private record SessionKey(String taskId, String pipelineId, int stageIndex) { }

    private static final boolean NATIVE_LIBRARY_LOADED = loadNativeLibrary();
    private final Map<SessionKey, Long> sessions = new ConcurrentHashMap<>();

    @Override
    public String runtimeName() {
        return "qnn";
    }

    @Override
    public String runtimeVersion() {
        return NATIVE_LIBRARY_LOADED ? nativeRuntimeVersion() : "unavailable";
    }

    @Override
    public boolean isAvailable(Context context, AndroidModelArtifact artifact) {
        return NATIVE_LIBRARY_LOADED
                && nativeExecutionReady()
                && artifact.runtime().equals("qnn")
                && Files.isRegularFile(artifact.artifactPath())
                && nativeProbe(artifact.artifactPath().toString());
    }

    @Override
    public RuntimeExecutionResult execute(
            Context context, AndroidModelArtifact artifact, RuntimeExecutionRequest request)
            throws Exception {
        if (!NATIVE_LIBRARY_LOADED) {
            throw new IllegalStateException("DragonNest QNN JNI library is not packaged");
        }
        SessionKey key = new SessionKey(
                request.taskId(), request.pipelineId(), request.stageIndex());
        if (request.operation() == PipelineOperation.PIPELINE_RESET
                || request.operation() == PipelineOperation.PIPELINE_CANCEL) {
            release(key);
            return new RuntimeExecutionResult("", null, "htp");
        }

        boolean ephemeral = request.operation()
                == PipelineOperation.PIPELINE_OPERATION_UNSPECIFIED;
        long handle;
        if (request.operation() == PipelineOperation.PIPELINE_PREFILL || ephemeral) {
            release(key);
            handle = nativeCreateSession(
                    artifact.artifactPath().toString(),
                    artifact.runtimeOptionsJson(),
                    request.pipelineId(),
                    request.stageIndex());
            if (handle == 0) {
                throw new IllegalStateException("QNN context/session creation failed");
            }
            sessions.put(key, handle);
        } else {
            Long existing = sessions.get(key);
            if (existing == null) {
                throw new IllegalStateException("QNN decode requested before stage prefill");
            }
            handle = existing;
        }

        try {
            BoundaryTensor input = request.inputBoundary();
            NativeStageResult nativeResult = nativeExecute(
                    handle,
                    request.operation().getNumber(),
                    request.requestText(),
                    request.tokenId(),
                    input == null ? "" : input.getTensorName(),
                    input == null ? "" : input.getDtype(),
                    input == null ? new int[0] : input.getShapeList().stream()
                            .mapToInt(Integer::intValue).toArray(),
                    input == null ? new byte[0] : input.getData().toByteArray(),
                    request.finalStage());
            BoundaryTensor boundary = null;
            if (nativeResult.boundaryData.length > 0) {
                String checksum = "sha256:" + hex(
                        MessageDigest.getInstance("SHA-256")
                                .digest(nativeResult.boundaryData));
                boundary = BoundaryTensor.newBuilder()
                        .setTensorName(nativeResult.boundaryName)
                        .setDtype(nativeResult.boundaryDtype)
                        .addAllShape(Arrays.stream(nativeResult.boundaryShape)
                                .boxed().toList())
                        .setData(ByteString.copyFrom(nativeResult.boundaryData))
                        .setChecksum(checksum)
                        .build();
            }
            return new RuntimeExecutionResult(
                    "", boundary, "htp", nativeResult.nextTokenId,
                    nativeResult.eos, nativeResult.tokenText);
        } finally {
            if (ephemeral) {
                release(key);
            }
        }
    }

    private void release(SessionKey key) {
        Long handle = sessions.remove(key);
        if (handle != null) {
            nativeReleaseSession(handle);
        }
    }

    @Override
    public void releaseTask(String taskId) {
        sessions.keySet().stream()
                .filter(key -> key.taskId().equals(taskId))
                .toList()
                .forEach(this::release);
    }

    private static boolean loadNativeLibrary() {
        try {
            System.loadLibrary("dragonnest_qnn_jni");
            return true;
        } catch (UnsatisfiedLinkError unavailable) {
            return false;
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder output = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            output.append(String.format("%02x", value & 0xff));
        }
        return output.toString();
    }

    /** Value object populated by JNI after one prompt or decode graph execution. */
    public static final class NativeStageResult {
        public final String boundaryName;
        public final String boundaryDtype;
        public final int[] boundaryShape;
        public final byte[] boundaryData;
        public final Integer nextTokenId;
        public final boolean eos;
        public final String tokenText;

        public NativeStageResult(
                String boundaryName,
                String boundaryDtype,
                int[] boundaryShape,
                byte[] boundaryData,
                Integer nextTokenId,
                boolean eos,
                String tokenText) {
            this.boundaryName = boundaryName;
            this.boundaryDtype = boundaryDtype;
            this.boundaryShape = boundaryShape;
            this.boundaryData = boundaryData;
            this.nextTokenId = nextTokenId;
            this.eos = eos;
            this.tokenText = tokenText;
        }
    }

    private static native boolean nativeProbe(String contextBinaryPath);

    /** Prevents staged bytes from becoming schedulable before execution is bound. */
    private static native boolean nativeExecutionReady();

    private static native long nativeCreateSession(
            String contextBinaryPath, String runtimeOptionsJson,
            String pipelineId, int stageIndex);

    private static native NativeStageResult nativeExecute(
            long handle, int operation, String prompt, int tokenId,
            String boundaryName, String boundaryDtype, int[] boundaryShape,
            byte[] boundaryData, boolean finalStage);

    private static native void nativeReleaseSession(long handle);

    private static native String nativeRuntimeVersion();
}

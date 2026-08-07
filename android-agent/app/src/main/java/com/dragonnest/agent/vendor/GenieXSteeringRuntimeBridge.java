package com.dragonnest.agent.vendor;

import android.content.Context;
import android.util.Log;

import com.dragonnest.agent.AndroidModelArtifact;
import com.dragonnest.agent.AndroidRuntimeBridge;
import com.dragonnest.agent.RuntimeExecutionRequest;
import com.dragonnest.agent.RuntimeExecutionResult;
import com.dragonnest.geniexsteeringlab.GenieXBridge;
import com.dragonnest.proto.SteeringSpec;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Runtime activation-steering adapter for the forked GenieX closure.
 *
 * <p>This bridge is deliberately <em>additive</em>. It answers to its own
 * runtime name ({@code genie_aux}) and loads its own native closure
 * ({@code libgnxfrk*.so}), so the stock {@link GenieXRuntimeBridge} and the
 * Qualcomm GenieX 0.3.5 AAR that the physically accepted Base/Balanced path
 * runs on are untouched by its presence.
 *
 * <p>Steering is bound as GenieX auxiliary tensors: {@code alpha} float32
 * {@code [1]} and {@code steering_vector} float32 {@code [1,1,1024]}, injected
 * at the layer the compiled bundle exposes. A request whose {@link
 * SteeringSpec} names an unknown vector, an unsupported layer, or a mode other
 * than {@code runtime_vector} is rejected rather than quietly executed
 * unsteered — a wrong-but-plausible answer is worse than a failure here.
 *
 * <p>A physical runtime error always propagates; this bridge never falls back
 * to DragonNest's mock executor and never substitutes prompt conditioning.
 */
public final class GenieXSteeringRuntimeBridge implements AndroidRuntimeBridge {
    private static final String TAG = "DragonNestGenieXAux";

    /** Filename of the validated unit vector staged inside the bundle. */
    private static final String VECTOR_FILE = "steering_vector_layer7_unit.bin";
    /** Sidecar that makes the bundle self-declare its auxiliary inputs. */
    private static final String AUX_SIDECAR = "aux_inputs.json";
    /** Qwen3-0.6B hidden width; the compiled steering_vector input is [1,1,1024]. */
    private static final int HIDDEN_SIZE = 1024;
    /** The vector is stored L2-normalised; reject anything that is not. */
    private static final float UNIT_NORM_TOLERANCE = 1e-3f;
    private static final String RUNTIME_VECTOR_MODE = "runtime_vector";

    private final Object lock = new Object();
    private String loadedBundlePath;
    private float[] vector;

    @Override
    public String runtimeName() {
        return "genie_aux";
    }

    @Override
    public String runtimeVersion() {
        return "GenieX-fork-aux-0.3.5 / QAIRT-2.45";
    }

    @Override
    public boolean isAvailable(Context context, AndroidModelArtifact artifact) {
        if (!runtimeName().equals(artifact.runtime())) {
            return false;
        }
        try {
            synchronized (lock) {
                ensureLoaded(context, artifact);
            }
            return true;
        } catch (Throwable failure) {
            Log.e(TAG, "Steering probe failed for " + artifact.modelId(), failure);
            return false;
        }
    }

    @Override
    public RuntimeExecutionResult execute(
            Context context,
            AndroidModelArtifact artifact,
            RuntimeExecutionRequest request) throws Exception {
        if (request.inputBoundary() != null) {
            throw new IllegalArgumentException(
                    "GenieX full-model bundles do not accept pipeline boundary tensors");
        }
        // The native handle is process-wide and not thread-safe; serializing
        // here also keeps the aux-write deltas attributable to one request.
        synchronized (lock) {
            ensureLoaded(context, artifact);
            SteeringSpec steering = request.steering();
            boolean steered = steering != null && steering.getEnabled();
            float alpha = 0.0f;
            if (steered) {
                validateSteering(artifact, steering);
                alpha = steering.getAlpha();
            }

            String raw = GenieXBridge.generate(
                    request.requestText(),
                    alpha,
                    steered ? vector : null,
                    steered,
                    generationLimit(artifact, request));
            JSONObject response = new JSONObject(raw);
            if (!response.optBoolean("ok", false)) {
                throw new IllegalStateException(
                        "GenieX steering generation failed: "
                                + response.optString("error", "unknown error"));
            }
            String text = response.optString("text", "").trim();
            if (text.isEmpty()) {
                throw new IllegalStateException("GenieX steering returned an empty response");
            }

            long prefillWrites = response.optLong("delta_prefill_writes", 0L);
            long decodeWrites = response.optLong("delta_decode_writes", 0L);
            if (steered && (prefillWrites <= 0 || decodeWrites <= 0)) {
                // Both graphs must have received the tensors. Returning text
                // that was not actually steered would be the worst outcome:
                // it looks like success.
                throw new IllegalStateException(
                        "steering requested but aux tensors were not bound to both phases "
                                + "(prefill=" + prefillWrites + ", decode=" + decodeWrites + ")");
            }
            JSONObject stats = response.optJSONObject("stats");
            Log.i(TAG, "task=" + request.taskId()
                    + " artifact=" + artifact.artifactId()
                    + " steered=" + steered
                    + " alpha=" + alpha
                    + " vector_id=" + (steered ? steering.getVectorId() : "")
                    + " layer=" + (steered ? steering.getTargetLayer() : -1)
                    + " prefill_aux_writes=" + prefillWrites
                    + " decode_aux_writes=" + decodeWrites
                    + " generated_tokens=" + response.optInt("generated_tokens", 0)
                    + " tps=" + response.optDouble("tps", 0.0)
                    + " context_loads=" + (stats == null ? -1 : stats.optLong("context_loads", -1))
                    + " ms=" + response.optLong("ms", 0L));
            return new RuntimeExecutionResult(text, null, "htp");
        }
    }

    /**
     * Rejects any steering request the compiled bundle and staged vector cannot
     * honour exactly. Everything checked here is advertised by the artifact
     * itself, so the device never claims a capability it does not hold.
     */
    private void validateSteering(AndroidModelArtifact artifact, SteeringSpec steering) {
        String mode = steering.getMode();
        if (!RUNTIME_VECTOR_MODE.equals(mode)) {
            throw new IllegalArgumentException(
                    "steering mode " + (mode.isEmpty() ? "(unset)" : mode)
                            + " is not supported by " + runtimeName()
                            + "; only " + RUNTIME_VECTOR_MODE + " is");
        }
        String vectorId = steering.getVectorId();
        if (!artifact.steeringVectorIds().contains(vectorId)) {
            throw new IllegalArgumentException(
                    "unknown steering vector " + (vectorId.isEmpty() ? "(unset)" : vectorId)
                            + "; " + artifact.modelId() + " holds "
                            + artifact.steeringVectorIds());
        }
        int layer = steering.getTargetLayer();
        if (!artifact.supportedSteeringLayers().contains(layer)) {
            throw new IllegalArgumentException(
                    "steering layer " + layer + " is not compiled into "
                            + artifact.modelId() + "; it exposes "
                            + artifact.supportedSteeringLayers());
        }
        float alpha = steering.getAlpha();
        if (Float.isNaN(alpha) || Float.isInfinite(alpha)) {
            throw new IllegalArgumentException("steering alpha is not finite: " + alpha);
        }
    }

    /** Loads the native closure and the model context exactly once. */
    private void ensureLoaded(Context context, AndroidModelArtifact artifact) throws IOException {
        Path directory = artifact.artifactPath();
        String bundlePath = directory.toAbsolutePath().toString();
        if (loadedBundlePath != null) {
            if (!loadedBundlePath.equals(bundlePath)) {
                // The forked shim owns a single process-wide handle bound to
                // the first bundle it loaded. Silently serving a different
                // artifact from it would misreport which model answered.
                throw new IllegalStateException(
                        "steering runtime already holds " + loadedBundlePath
                                + "; refusing to serve " + bundlePath);
            }
            return;
        }
        if (!Files.isDirectory(directory)) {
            throw new IOException("steering artifact is not a directory: " + directory);
        }
        if (!Files.isRegularFile(directory.resolve(AUX_SIDECAR))) {
            throw new IOException(
                    "bundle does not declare auxiliary inputs (" + AUX_SIDECAR + " missing): "
                            + directory + " -- a prompt-only or unsteerable bundle cannot "
                            + "serve runtime_vector");
        }
        float[] loaded = readUnitVector(directory.resolve(VECTOR_FILE));

        String nativeLibDir = context.getApplicationInfo().nativeLibraryDir;
        JSONObject result;
        try {
            result = new JSONObject(GenieXBridge.init(nativeLibDir, bundlePath));
        } catch (Exception failure) {
            throw new IOException("forked GenieX init failed for " + bundlePath, failure);
        }
        if (!result.optBoolean("ok", false)) {
            throw new IOException("forked GenieX init failed: "
                    + result.optString("error", "unknown error"));
        }
        vector = loaded;
        loadedBundlePath = bundlePath;
        Log.i(TAG, "loaded steering context for " + artifact.modelId()
                + " load_ms=" + result.optLong("load_ms", -1)
                + " hidden_size=" + loaded.length);
    }

    /**
     * Reads the layer-7 concise-vs-verbose vector as little-endian float32 and
     * checks the shape and normalisation the compiled input expects. A vector
     * of the wrong width or norm would still "work" numerically while meaning
     * something entirely different, so it is refused.
     */
    // Package-private rather than private so the JVM unit tests can exercise
    // the rejection paths without a device: everything else in this class
    // reaches the native closure, which only exists on arm64 hardware.
    static float[] readUnitVector(Path path) throws IOException {
        if (!Files.isRegularFile(path)) {
            throw new IOException("steering vector is missing: " + path);
        }
        byte[] raw = Files.readAllBytes(path);
        if (raw.length != HIDDEN_SIZE * Float.BYTES) {
            throw new IOException("steering vector must be " + HIDDEN_SIZE
                    + " float32 values (" + (HIDDEN_SIZE * Float.BYTES) + " bytes), got "
                    + raw.length + " bytes: " + path);
        }
        float[] values = new float[HIDDEN_SIZE];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values);
        double sumOfSquares = 0.0;
        for (float value : values) {
            if (Float.isNaN(value) || Float.isInfinite(value)) {
                throw new IOException("steering vector contains a non-finite value: " + path);
            }
            sumOfSquares += (double) value * value;
        }
        double norm = Math.sqrt(sumOfSquares);
        if (Math.abs(norm - 1.0) > UNIT_NORM_TOLERANCE) {
            throw new IOException("steering vector is not unit L2 norm (got " + norm + "): " + path);
        }
        return values;
    }

    private static int generationLimit(
            AndroidModelArtifact artifact, RuntimeExecutionRequest request)
            throws JSONException {
        if (request.maxNewTokens() > 0) {
            return Math.min(request.maxNewTokens(), artifact.maxContextTokens());
        }
        int configured = new JSONObject(artifact.runtimeOptionsJson()).optInt("max_new_tokens", 96);
        return Math.max(1, Math.min(configured, artifact.maxContextTokens()));
    }
}

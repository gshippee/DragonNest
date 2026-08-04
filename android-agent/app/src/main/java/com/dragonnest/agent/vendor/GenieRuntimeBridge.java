package com.dragonnest.agent.vendor;

import android.content.Context;

import com.dragonnest.agent.AndroidModelArtifact;
import com.dragonnest.agent.AndroidRuntimeBridge;
import com.dragonnest.agent.RuntimeExecutionRequest;
import com.dragonnest.agent.RuntimeExecutionResult;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * QAIRT 2.48 Genie C API bridge. It is packaged only in hardware builds that
 * set DRAGONNEST_QAIRT_SDK_ROOT and provide the matching arm64 shared objects.
 */
public final class GenieRuntimeBridge implements AndroidRuntimeBridge {
    private static final boolean NATIVE_LIBRARY_LOADED = loadNativeLibrary();

    @Override
    public String runtimeName() {
        return "genie";
    }

    @Override
    public String runtimeVersion() {
        return NATIVE_LIBRARY_LOADED ? nativeRuntimeVersion() : "unavailable";
    }

    @Override
    public boolean isAvailable(Context context, AndroidModelArtifact artifact) {
        if (!NATIVE_LIBRARY_LOADED || !artifact.runtime().equals("genie")) {
            return false;
        }
        try {
            return nativeProbe(loadConfig(artifact.artifactPath()));
        } catch (Exception unavailable) {
            return false;
        }
    }

    @Override
    public RuntimeExecutionResult execute(
            Context context, AndroidModelArtifact artifact, RuntimeExecutionRequest request)
            throws Exception {
        if (request.inputBoundary() != null) {
            throw new UnsupportedOperationException(
                    "Genie bundles do not expose layer-pipeline boundary tensors");
        }
        if (!NATIVE_LIBRARY_LOADED) {
            throw new IllegalStateException("DragonNest Genie JNI library is not packaged");
        }
        String response = nativeExecute(loadConfig(artifact.artifactPath()), request.requestText());
        return new RuntimeExecutionResult(response, null, "htp");
    }

    private static boolean loadNativeLibrary() {
        try {
            System.loadLibrary("dragonnest_genie_jni");
            return true;
        } catch (UnsatisfiedLinkError unavailable) {
            return false;
        }
    }

    private static String loadConfig(Path bundleDirectory) throws IOException, JSONException {
        Path configPath = bundleDirectory.resolve("genie_config.json");
        if (!Files.isDirectory(bundleDirectory) || !Files.isRegularFile(configPath)) {
            throw new IOException("Genie artifact must be a bundle containing genie_config.json");
        }
        JSONObject root = new JSONObject(
                new String(Files.readAllBytes(configPath), StandardCharsets.UTF_8));
        JSONObject dialog = root.getJSONObject("dialog");
        JSONObject tokenizer = dialog.getJSONObject("tokenizer");
        tokenizer.put("path", resolve(bundleDirectory, tokenizer.getString("path")));

        JSONObject engine = dialog.getJSONObject("engine");
        if (engine.has("extensions")) {
            engine.put("extensions", resolve(bundleDirectory, engine.getString("extensions")));
        }
        JSONArray binaries = engine.getJSONObject("model")
                .getJSONObject("binary")
                .getJSONArray("ctx-bins");
        for (int index = 0; index < binaries.length(); index++) {
            binaries.put(index, resolve(bundleDirectory, binaries.getString(index)));
        }
        return root.toString();
    }

    private static String resolve(Path bundleDirectory, String path) throws IOException {
        Path resolved = Path.of(path);
        if (!resolved.isAbsolute()) {
            resolved = bundleDirectory.resolve(resolved).normalize();
        }
        if (!resolved.startsWith(bundleDirectory.toAbsolutePath().normalize())
                || !Files.isRegularFile(resolved)) {
            throw new IOException("Genie bundle references a missing or unsafe asset: " + path);
        }
        return resolved.toString();
    }

    private static native boolean nativeProbe(String configJson);

    private static native String nativeExecute(String configJson, String prompt);

    private static native String nativeRuntimeVersion();
}

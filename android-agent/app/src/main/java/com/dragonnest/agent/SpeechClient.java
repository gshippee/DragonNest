package com.dragonnest.agent;

import android.content.Context;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * Fetches spoken audio for a reply from the Brain's dashboard API.
 *
 * <p>Synthesis happens on the Brain host, not here. MeloTTS ships as QNN
 * context binaries compiled for a specific target, and the bundle DragonNest
 * uses is built for Snapdragon X Elite (v73) -- it cannot be deserialized by
 * this phone's HTP (SM8750/v79). So the phone sends the reply text, the laptop
 * runs the model on its NPU, and the phone plays back the .wav that comes
 * back.
 *
 * <p>This deliberately goes over the dashboard's HTTP API rather than the gRPC
 * control plane: the audio is a response to a UI gesture, not a routed fabric
 * task, and {@code TaskResult} carries no binary field.
 */
public final class SpeechClient {

    /** Synthesis failed on the Brain, or the Brain could not be reached. */
    public static class SpeechException extends IOException {
        SpeechException(String message) {
            super(message);
        }

        SpeechException(String message, Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * The Brain host's NPU is busy with the pinned language model. Speech
     * yields to it rather than competing for the same DSP session, so this is
     * a "try again in a moment", not a failure.
     */
    public static final class SpeechBusyException extends SpeechException {
        SpeechBusyException(String message) {
            super(message);
        }
    }

    /** The Brain host is not provisioned for speech (no model, no runtime). */
    public static final class SpeechUnavailableException extends SpeechException {
        SpeechUnavailableException(String message) {
            super(message);
        }
    }

    // Matches the server's own cap on SpeechRequest.text.
    private static final int MAX_TEXT_LENGTH = 4000;
    private static final int CONNECT_TIMEOUT_MS = 10_000;
    // Synthesis is a fresh qnn-net-run process per graph call plus melo/BERT
    // init, measured at ~16s for a short reply and longer for a multi-chunk
    // one, so this is generous by design.
    private static final int READ_TIMEOUT_MS = 300_000;
    private static final int MAX_AUDIO_BYTES = 32 * 1024 * 1024;

    private final AgentConfiguration configuration;
    private final File cacheDir;

    public SpeechClient(Context context) {
        this(new AgentConfiguration(context), new File(context.getCacheDir(), "speech"));
    }

    SpeechClient(AgentConfiguration configuration, File cacheDir) {
        this.configuration = configuration;
        this.cacheDir = cacheDir;
    }

    URL endpoint() throws IOException {
        return endpointFor(
                configuration.useTls(), configuration.brainHost(), configuration.dashboardPort());
    }

    static URL endpointFor(boolean useTls, String host, int dashboardPort) throws IOException {
        String trimmed = host == null ? "" : host.trim();
        if (trimmed.isEmpty()) {
            throw new SpeechUnavailableException("No DragonNest server address is configured.");
        }
        String scheme = useTls ? "https" : "http";
        return new URL(scheme + "://" + trimmed + ":" + dashboardPort + "/api/speech");
    }

    /**
     * Return a playable .wav for {@code text}, fetching it only if this device
     * has not already cached that exact text. The Brain caches by content too;
     * this second cache just avoids the round trip.
     */
    public File synthesize(String text) throws IOException {
        String trimmed = text == null ? "" : text.trim();
        if (trimmed.isEmpty()) {
            throw new SpeechException("There is nothing to read aloud.");
        }
        if (trimmed.length() > MAX_TEXT_LENGTH) {
            trimmed = trimmed.substring(0, MAX_TEXT_LENGTH);
        }

        File cached = cachePath(trimmed);
        if (cached.isFile() && cached.length() > 0) {
            return cached;
        }
        byte[] audio = request(trimmed);
        writeAtomically(cached, audio);
        return cached;
    }

    private byte[] request(String text) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) endpoint().openConnection();
        try {
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("Accept", "audio/wav");

            byte[] body;
            try {
                body = new JSONObject().put("text", text).toString().getBytes(StandardCharsets.UTF_8);
            } catch (JSONException exc) {
                throw new SpeechException("Could not encode the request.", exc);
            }
            try (OutputStream out = connection.getOutputStream()) {
                out.write(body);
            }

            int status = connection.getResponseCode();
            if (status == HttpURLConnection.HTTP_OK) {
                try (InputStream in = connection.getInputStream()) {
                    return readAll(in);
                }
            }
            throw failureFor(status, detail(connection));
        } catch (SpeechException exc) {
            throw exc;
        } catch (IOException exc) {
            throw new SpeechException("Could not reach DragonNest to read this aloud.", exc);
        } finally {
            connection.disconnect();
        }
    }

    private static SpeechException failureFor(int status, String detail) {
        if (status == HttpURLConnection.HTTP_CONFLICT) {
            return new SpeechBusyException(
                    detail.isEmpty() ? "DragonNest is thinking. Try again in a moment." : detail);
        }
        if (status == HttpURLConnection.HTTP_UNAVAILABLE) {
            return new SpeechUnavailableException(
                    detail.isEmpty() ? "This DragonNest server cannot speak yet." : detail);
        }
        return new SpeechException(
                detail.isEmpty() ? "DragonNest could not read this aloud." : detail);
    }

    /** Pull FastAPI's {"detail": ...} out of an error body, if there is one. */
    private static String detail(HttpURLConnection connection) {
        InputStream errors = connection.getErrorStream();
        if (errors == null) {
            return "";
        }
        try (InputStream in = errors) {
            String body = new String(readAll(in), StandardCharsets.UTF_8);
            return new JSONObject(body).optString("detail", "");
        } catch (IOException | JSONException exc) {
            return "";
        }
    }

    private static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] chunk = new byte[16 * 1024];
        int read;
        while ((read = in.read(chunk)) != -1) {
            buffer.write(chunk, 0, read);
            if (buffer.size() > MAX_AUDIO_BYTES) {
                throw new SpeechException("The audio response was unexpectedly large.");
            }
        }
        return buffer.toByteArray();
    }

    File cachePath(String text) {
        return new File(cacheDir, digest(text) + ".wav");
    }

    private static String digest(String text) {
        try {
            byte[] hash = MessageDigest.getInstance("SHA-256")
                    .digest(text.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (int index = 0; index < 16; index++) {
                hex.append(String.format("%02x", hash[index]));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException exc) {
            throw new IllegalStateException("SHA-256 is required", exc);
        }
    }

    /**
     * Write through a temp file so a cancelled or failed write can never leave
     * a truncated .wav behind that later looks like a valid cache hit.
     */
    private static void writeAtomically(File destination, byte[] audio) throws IOException {
        File parent = destination.getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
            throw new SpeechException("Could not create the audio cache directory.");
        }
        File temporary = new File(destination.getPath() + ".tmp");
        try (FileOutputStream out = new FileOutputStream(temporary)) {
            out.write(audio);
            out.getFD().sync();
        }
        if (!temporary.renameTo(destination)) {
            temporary.delete();
            throw new SpeechException("Could not save the audio.");
        }
    }
}

package com.dragonnest.agent;

import org.json.JSONObject;
import org.json.JSONException;

import java.time.Instant;
import java.util.regex.Pattern;

public record EnrollmentPayload(
        String brainHost,
        int brainPort,
        boolean useTls,
        String sessionId,
        String credential,
        long expiresAtEpoch) {
    private static final Pattern HOST = Pattern.compile(
            "^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,251}[A-Za-z0-9])?$");

    public static EnrollmentPayload parse(String encoded) {
        if (encoded == null || encoded.isBlank() || encoded.length() > 4096) {
            throw new IllegalArgumentException("Invalid enrollment QR payload");
        }
        JSONObject payload;
        try {
            payload = new JSONObject(encoded);
        } catch (JSONException invalidJson) {
            throw new IllegalArgumentException("Invalid enrollment QR payload", invalidJson);
        }
        if (!"dragonnest.enrollment".equals(payload.optString("type"))) {
            throw new IllegalArgumentException("QR code is not a DragonNest enrollment");
        }
        if (payload.optInt("version", -1) != 1) {
            throw new IllegalArgumentException("Unsupported enrollment QR version");
        }
        String host = payload.optString("brain_host", "").trim();
        int port = payload.optInt("brain_port", -1);
        String sessionId = payload.optString("session_id", "").trim();
        String credential = payload.optString("credential", "").trim();
        long expires = payload.optLong("expires_at_epoch", 0);
        if (!HOST.matcher(host).matches()) {
            throw new IllegalArgumentException("Enrollment QR has an invalid Brain host");
        }
        if (port < 1 || port > 65535) {
            throw new IllegalArgumentException("Enrollment QR has an invalid gRPC port");
        }
        if (sessionId.isEmpty() || !credential.startsWith("dn_bootstrap_")) {
            throw new IllegalArgumentException("Enrollment QR credential is invalid");
        }
        if (expires <= Instant.now().getEpochSecond()) {
            throw new IllegalArgumentException("Enrollment QR has expired");
        }
        return new EnrollmentPayload(
                host,
                port,
                payload.optBoolean("use_tls", false),
                sessionId,
                credential,
                expires);
    }
}

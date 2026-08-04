package com.dragonnest.agent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.time.Instant;

public final class EnrollmentPayloadTest {
    @Test
    public void parsesValidEnrollment() {
        String payload = "{"
                + "\"type\":\"dragonnest.enrollment\","
                + "\"version\":1,"
                + "\"brain_host\":\"192.168.1.20\","
                + "\"brain_port\":50051,"
                + "\"use_tls\":false,"
                + "\"session_id\":\"session-1\","
                + "\"credential\":\"dn_bootstrap_secret\","
                + "\"expires_at_epoch\":"
                + (Instant.now().getEpochSecond() + 300)
                + "}";

        EnrollmentPayload parsed = EnrollmentPayload.parse(payload);

        assertEquals("192.168.1.20", parsed.brainHost());
        assertEquals(50051, parsed.brainPort());
        assertEquals("session-1", parsed.sessionId());
        assertTrue(parsed.credential().startsWith("dn_bootstrap_"));
    }

    @Test
    public void rejectsExpiredAndForeignQrCodes() {
        assertThrows(
                IllegalArgumentException.class,
                () -> EnrollmentPayload.parse("{\"type\":\"other\",\"version\":1}"));
        String expired = "{"
                + "\"type\":\"dragonnest.enrollment\","
                + "\"version\":1,"
                + "\"brain_host\":\"brain.local\","
                + "\"brain_port\":50051,"
                + "\"session_id\":\"session-1\","
                + "\"credential\":\"dn_bootstrap_secret\","
                + "\"expires_at_epoch\":1}";
        assertThrows(IllegalArgumentException.class, () -> EnrollmentPayload.parse(expired));
    }
}

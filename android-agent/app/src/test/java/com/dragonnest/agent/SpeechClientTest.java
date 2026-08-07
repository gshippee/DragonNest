package com.dragonnest.agent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

/**
 * Unit coverage for the parts of {@link SpeechClient} that do not need a
 * device: endpoint construction and content-addressed caching.
 */
public final class SpeechClientTest {

    @Test
    public void endpointUsesTheDashboardPortNotTheGrpcPort() throws IOException {
        assertEquals(
                "http://10.73.51.92:8080/api/speech",
                SpeechClient.endpointFor(false, "10.73.51.92", 8080).toString());
    }

    @Test
    public void endpointHonoursTls() throws IOException {
        assertEquals(
                "https://brain.local:8443/api/speech",
                SpeechClient.endpointFor(true, "brain.local", 8443).toString());
    }

    @Test
    public void endpointRejectsAnUnconfiguredHost() {
        try {
            SpeechClient.endpointFor(false, "   ", 8080);
            throw new AssertionError("expected an unavailable error");
        } catch (IOException exc) {
            assertTrue(exc instanceof SpeechClient.SpeechUnavailableException);
        }
    }

    @Test
    public void cacheIsKeyedByContentSoTheSameReplyIsFetchedOnce() {
        SpeechClient speech = new SpeechClient(null, new File("speech-cache"));
        assertEquals(
                speech.cachePath("Take one tablet daily.").getName(),
                speech.cachePath("Take one tablet daily.").getName());
        assertNotEquals(
                speech.cachePath("Take one tablet daily.").getName(),
                speech.cachePath("Take two tablets daily.").getName());
    }
}

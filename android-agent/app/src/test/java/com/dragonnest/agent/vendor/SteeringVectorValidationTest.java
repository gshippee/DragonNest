package com.dragonnest.agent.vendor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Shape and normalisation checks on the layer-7 steering vector.
 *
 * A vector of the wrong width or scale would still multiply cleanly and
 * produce fluent text -- it would just mean something other than the
 * calibrated concise/verbose axis. These are the checks that turn that silent
 * mis-steer into a refusal.
 */
public final class SteeringVectorValidationTest {
    private static final int HIDDEN = 1024;

    @Test
    public void acceptsTheValidatedUnitVector() throws Exception {
        Path path = Files.createTempDirectory("vector").resolve("v.bin");
        float[] values = new float[HIDDEN];
        values[0] = 1.0f; // unit L2 norm
        Files.write(path, encode(values));

        float[] loaded = GenieXSteeringRuntimeBridge.readUnitVector(path);

        assertEquals(HIDDEN, loaded.length);
        assertEquals(1.0f, loaded[0], 1e-6f);
    }

    @Test
    public void rejectsAVectorOfTheWrongWidth() throws Exception {
        Path path = Files.createTempDirectory("vector-short").resolve("v.bin");
        Files.write(path, encode(new float[HIDDEN - 1]));

        IOException failure = assertThrows(
                IOException.class, () -> GenieXSteeringRuntimeBridge.readUnitVector(path));
        assertTrue(failure.getMessage().contains("float32"));
    }

    @Test
    public void rejectsAVectorThatIsNotUnitNorm() throws Exception {
        Path path = Files.createTempDirectory("vector-norm").resolve("v.bin");
        float[] values = new float[HIDDEN];
        values[0] = 4.0f; // right shape, wrong scale: would silently mean 4x alpha
        Files.write(path, encode(values));

        IOException failure = assertThrows(
                IOException.class, () -> GenieXSteeringRuntimeBridge.readUnitVector(path));
        assertTrue(failure.getMessage().contains("unit L2 norm"));
    }

    @Test
    public void rejectsAMissingVector() throws Exception {
        Path path = Files.createTempDirectory("vector-absent").resolve("absent.bin");

        IOException failure = assertThrows(
                IOException.class, () -> GenieXSteeringRuntimeBridge.readUnitVector(path));
        assertTrue(failure.getMessage().contains("missing"));
        assertFalse(Files.exists(path));
    }


    private static byte[] encode(float[] values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().put(values);
        return buffer.array();
    }
}

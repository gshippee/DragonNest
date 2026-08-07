package com.dragonnest.geniexsteeringlab;

/** JNI boundary to the forked GenieX C ABI (libsteeringlab_jni.so). */
public final class GenieXBridge {
    static {
        System.loadLibrary("steeringlab_jni");
    }

    private GenieXBridge() {}

    /** Loads the model once; subsequent calls are no-ops. Returns JSON. */
    public static native String init(String nativeLibDir, String bundleDir);

    /**
     * One generation request. alpha/vector are bound as runtime aux inputs when
     * useAux is true; with useAux false the request is bit-identical to stock
     * GenieX. Returns JSON with text, latency, and per-phase aux write deltas.
     */
    public static native String generate(
            String prompt, float alpha, float[] vector, boolean useAux, int maxTokens);
}

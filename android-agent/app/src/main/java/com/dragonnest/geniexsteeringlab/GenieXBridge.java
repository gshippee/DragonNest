package com.dragonnest.geniexsteeringlab;

/**
 * JNI boundary to the forked GenieX public C ABI (libsteeringlab_jni.so).
 *
 * <p>This package name is not a style choice. DragonNest reuses the exact
 * {@code libsteeringlab_jni.so} that was physically proven on the Galaxy S25
 * Ultra rather than recompiling it, and JNI resolves native methods by a symbol
 * name derived from the declaring class
 * ({@code Java_com_dragonnest_geniexsteeringlab_GenieXBridge_generate}). Moving
 * this class into {@code com.dragonnest.agent.vendor} would silently break that
 * binding, so the class stays here and
 * {@code com.dragonnest.agent.vendor.GenieXSteeringRuntimeBridge} owns all
 * DragonNest-facing policy.
 *
 * <p>The shim only ever calls the forked <em>public</em> C ABI — {@code
 * geniex_init}, {@code geniex_llm_create}, {@code
 * geniex_llm_apply_chat_template}, {@code geniex_llm_generate}, {@code
 * geniex_llm_reset}, {@code geniex_llm_get_aux_stats} — and never reaches into
 * geniex-qairt internals. Runtime steering travels as
 * {@code geniex_LlmGenerateInput.aux_inputs}.
 *
 * <p>The native side holds one process-wide model handle guarded by a mutex, so
 * requests are serialized and the loaded context is reused across alpha changes
 * with no reload.
 */
public final class GenieXBridge {
    static {
        // Resolves the forked closure, whose sonames were renamed to
        // libgnxfrk*.so so they cannot collide with the stock GenieX AAR that
        // the accepted Base path runs on.
        System.loadLibrary("steeringlab_jni");
    }

    private GenieXBridge() { }

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

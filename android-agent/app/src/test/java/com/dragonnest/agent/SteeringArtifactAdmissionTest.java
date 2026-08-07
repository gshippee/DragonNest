package com.dragonnest.agent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Admission rules for the runtime-steering artifact, and the guard that keeps a
 * new artifact from taking down the models a device already serves.
 */
public final class SteeringArtifactAdmissionTest {
    private static final int HIDDEN = 1024;

    private static String manifest(String runtime, String extra) {
        return """
                {"models":[{
                  "model_id":"qwen3-0.6b-s25-runtime-steerable",
                  "model_version":"c1899de", "runtime":"%s",
                  "artifact_path":"steerable", "checksum":"sha256-tree:%s",
                  "tokenizer_id":"Qwen/Qwen3-0.6B", "precision":"w4a16",
                  "supported_accelerators":["htp"], "min_memory_mb":2048,
                  "max_context_tokens":512, "supports_steering":true,
                  "supports_data_parallel":true, "supports_layer_pipeline":false,
                  "model_family":"qwen3", "role":"small_chat",
                  "task_classes":["chat_qa"], "quality_score":0.7,
                  "runtime_version":"GenieX-fork-aux-0.3.5 / QAIRT-2.45"%s
                }]}""".formatted(runtime, "0".repeat(64), extra);
    }

    @Test
    public void advertisesRuntimeVectorSteeringForTheForkedRuntime() throws Exception {
        Path root = Files.createTempDirectory("dragonnest-steerable");
        String extra = """
                , "steering_mode":"runtime_vector",
                  "steering_vector_ids":["concise-vs-verbose-layer-7"],
                  "supported_steering_layers":[7], "behavior_profile_id":""\
                """;
        AndroidArtifactRegistry registry =
                AndroidArtifactRegistry.fromJson(manifest("genie_aux", extra), root);
        AndroidModelArtifact artifact = registry.all().get(0);

        assertEquals("genie_aux", artifact.runtime());
        assertTrue(artifact.capability().getSupportsSteering());
        assertEquals("runtime_vector", artifact.capability().getSteeringModes(0));
        assertEquals(
                "concise-vs-verbose-layer-7",
                artifact.capability().getSteeringVectorIds(0));
        assertEquals(7, (int) artifact.capability().getSupportedSteeringLayers(0));
        // It realizes Concise *and* Detailed by alpha, so it must not claim a
        // single behavior profile.
        assertEquals(0, artifact.capability().getBehaviorProfileIdsCount());
    }

    @Test
    public void unusableEntryIsSkippedWithoutDroppingTheWorkingModels() throws Exception {
        Path root = Files.createTempDirectory("dragonnest-mixed");
        Path base = root.resolve("base/model.bin");
        Files.createDirectories(base.getParent());
        Files.write(base, "base model".getBytes(StandardCharsets.UTF_8));
        String source = """
                {"models":[
                  {"model_id":"qwen3-0.6b-s25-base", "model_version":"v1",
                   "runtime":"genie", "artifact_path":"base/model.bin",
                   "checksum":"sha256:%s", "tokenizer_id":"Qwen/Qwen3-0.6B",
                   "precision":"w4a16", "supported_accelerators":["htp"],
                   "min_memory_mb":2048, "max_context_tokens":512,
                   "supports_steering":false, "supports_data_parallel":true,
                   "supports_layer_pipeline":false, "model_family":"qwen3",
                   "role":"small_chat", "task_classes":["chat_qa"],
                   "quality_score":0.7, "runtime_version":"GenieX-0.3.5"},
                  {"model_id":"from-the-future", "model_version":"v1",
                   "runtime":"some-unshipped-runtime", "artifact_path":"future",
                   "checksum":"sha256-tree:%s", "tokenizer_id":"t",
                   "precision":"w4a16", "supported_accelerators":["htp"],
                   "min_memory_mb":2048, "max_context_tokens":512,
                   "supports_steering":false, "supports_data_parallel":true,
                   "supports_layer_pipeline":false, "model_family":"qwen3",
                   "role":"small_chat", "task_classes":["chat_qa"],
                   "quality_score":0.7, "runtime_version":"x"}
                ]}""".formatted("0".repeat(64), "0".repeat(64));

        AndroidArtifactRegistry registry = AndroidArtifactRegistry.fromJson(source, root);

        // The whole manifest used to be thrown away here, which silently took
        // down a model the device was serving correctly.
        List<AndroidModelArtifact> admitted = registry.all();
        assertEquals(1, admitted.size());
        assertEquals("qwen3-0.6b-s25-base", admitted.get(0).modelId());
        assertEquals(1, registry.skippedEntries().size());
        assertTrue(registry.skippedEntries().get(0).contains("from-the-future"));
    }
}

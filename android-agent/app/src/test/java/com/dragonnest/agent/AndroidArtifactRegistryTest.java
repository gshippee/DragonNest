package com.dragonnest.agent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;

public final class AndroidArtifactRegistryTest {
    @Test
    public void acceptsVerifiedQnnArtifactAndBuildsCapability() throws Exception {
        Path root = Files.createTempDirectory("dragonnest-models");
        Path artifact = root.resolve("qwen/model.bin");
        Files.createDirectories(artifact.getParent());
        Files.write(artifact, "qnn model".getBytes(StandardCharsets.UTF_8));
        String checksum = sha256(Files.readAllBytes(artifact));
        String manifest = """
                {"models":[{
                  "model_id":"qwen-s25", "model_version":"s25-v1", "runtime":"qnn",
                  "artifact_path":"qwen/model.bin", "checksum":"sha256:%s",
                  "tokenizer_id":"Qwen/Qwen3-0.6B", "precision":"fp16",
                  "supported_accelerators":["htp"], "min_memory_mb":512,
                  "max_context_tokens":128, "supports_steering":false,
                  "supports_data_parallel":true, "supports_layer_pipeline":false,
                  "model_family":"qwen3", "role":"small_chat",
                  "task_classes":["chat_qa"], "quality_score":0.8,
                  "runtime_version":"QAIRT-test"
                }]}""".formatted(checksum);

        AndroidArtifactRegistry registry = AndroidArtifactRegistry.fromJson(manifest, root);
        AndroidModelArtifact parsed = registry.all().get(0);

        assertTrue(registry.isVerified(parsed));
        assertEquals("qwen-s25", parsed.capability().getModelId());
        assertEquals("qnn", parsed.capability().getRuntimeName());
        assertEquals("htp", parsed.capability().getSupportedAccelerators(0));
    }

    @Test
    public void rejectsEscapingPathsAndChecksumMismatches() throws Exception {
        Path root = Files.createTempDirectory("dragonnest-models");
        String pathEscape = """
                {"models":[{
                  "model_id":"bad", "model_version":"v1", "runtime":"qnn",
                  "artifact_path":"../outside.bin", "checksum":"sha256:%s",
                  "tokenizer_id":"tokenizer", "precision":"fp16",
                  "supported_accelerators":["htp"], "min_memory_mb":1,
                  "max_context_tokens":1, "supports_steering":false,
                  "supports_data_parallel":false, "supports_layer_pipeline":false,
                  "model_family":"qwen", "role":"small_chat", "task_classes":["chat_qa"]
                }]}""".formatted("0".repeat(64));
        assertThrows(
                IllegalArgumentException.class,
                () -> AndroidArtifactRegistry.fromJson(pathEscape, root));

        Path artifact = root.resolve("model.bin");
        Files.write(artifact, "contents".getBytes(StandardCharsets.UTF_8));
        String mismatch = pathEscape.replace("../outside.bin", "model.bin");
        AndroidArtifactRegistry registry = AndroidArtifactRegistry.fromJson(mismatch, root);
        assertFalse(registry.isVerified(registry.all().get(0)));
    }

    @Test
    public void acceptsEmbeddingOnlyIndexedPipelineStage() throws Exception {
        Path root = Files.createTempDirectory("dragonnest-pipeline");
        Path artifact = root.resolve("stage0.bin");
        Files.write(artifact, "stage-0".getBytes(StandardCharsets.UTF_8));
        String manifest = """
                {"models":[{
                  "model_id":"qwen3-1.7b-s0-s25", "model_version":"demo-v1",
                  "runtime":"qnn", "artifact_path":"stage0.bin",
                  "checksum":"sha256:%s", "tokenizer_id":"Qwen/Qwen3-1.7B",
                  "precision":"w4a16-name-w8a16-compile-observed",
                  "supported_accelerators":["htp"], "min_memory_mb":1024,
                  "max_context_tokens":512, "supports_steering":false,
                  "supports_data_parallel":false, "supports_layer_pipeline":true,
                  "model_family":"qwen3-1.7b", "role":"pipeline_segment",
                  "task_classes":["reasoning_analysis"],
                  "split_boundary":{"pipeline_id":"qwen3-1.7b-w4a16-demo-v1",
                    "stage_index":0, "stage_count":4, "total_layers":28,
                    "input_tensor":"input_ids", "output_tensor":"embedding",
                    "includes_embedding":true, "includes_lm_head":false,
                    "boundary_format":"qnn-raw-tensor-v1"}
                }]}""".formatted(sha256(Files.readAllBytes(artifact)));

        AndroidModelArtifact parsed = AndroidArtifactRegistry.fromJson(manifest, root)
                .all().get(0);

        assertEquals(0, parsed.segment().stageIndex());
        assertEquals(4, parsed.segment().stageCount());
        assertEquals(null, parsed.segment().transformerStartLayer());
        assertFalse(parsed.capability().getSegment().hasTransformerStartLayer());
        assertTrue(parsed.capability().getSegment().getIncludesEmbedding());
    }

    private static String sha256(byte[] source) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(source);
        StringBuilder output = new StringBuilder();
        for (byte value : digest) {
            output.append(String.format("%02x", value & 0xff));
        }
        return output.toString();
    }
}

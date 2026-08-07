package com.dragonnest.agent;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/** Loads only model artifacts rooted in the app-private DragonNest model directory. */
public final class AndroidArtifactRegistry {
    public static final String MODEL_DIRECTORY = "dragonnest-models";
    private static final Pattern CHECKSUM = Pattern.compile(
            "^sha256(?:-tree)?:[0-9a-fA-F]{64}$");
    private final Path root;
    private final Map<String, AndroidModelArtifact> artifacts;

    private AndroidArtifactRegistry(Path root, Map<String, AndroidModelArtifact> artifacts) {
        this.root = root;
        this.artifacts = Map.copyOf(artifacts);
    }

    public static AndroidArtifactRegistry loadInstalled(Context context) {
        Path root = context.getFilesDir().toPath().resolve(MODEL_DIRECTORY);
        Path manifest = root.resolve("manifest.json");
        if (!Files.isRegularFile(manifest)) {
            return new AndroidArtifactRegistry(root, Map.of());
        }
        try {
            return fromJson(
                    new String(Files.readAllBytes(manifest), StandardCharsets.UTF_8), root);
        } catch (IOException failure) {
            throw new IllegalStateException("Unable to read Android model manifest", failure);
        }
    }

    static AndroidArtifactRegistry fromJson(String source, Path modelRoot) {
        try {
            JSONObject manifest = new JSONObject(source);
            JSONArray models = manifest.optJSONArray("models");
            if (models == null) {
                throw new IllegalArgumentException("Model manifest must contain a models array");
            }
            Path root = modelRoot.toAbsolutePath().normalize();
            Map<String, AndroidModelArtifact> artifacts = new LinkedHashMap<>();
            for (int index = 0; index < models.length(); index++) {
                JSONObject item = models.optJSONObject(index);
                if (item == null) {
                    throw new IllegalArgumentException("Model manifest entry must be an object");
                }
                AndroidModelArtifact artifact = parse(item, root);
                if (artifacts.putIfAbsent(artifact.modelId(), artifact) != null) {
                    throw new IllegalArgumentException(
                            "Duplicate model_id in Android manifest: " + artifact.modelId());
                }
            }
            return new AndroidArtifactRegistry(root, artifacts);
        } catch (JSONException failure) {
            throw new IllegalArgumentException("Invalid Android model manifest JSON", failure);
        }
    }

    public List<AndroidModelArtifact> all() {
        return List.copyOf(artifacts.values());
    }

    public boolean isVerified(AndroidModelArtifact artifact) {
        try {
            verify(artifact);
            return true;
        } catch (IllegalArgumentException failure) {
            return false;
        }
    }

    public Path verify(AndroidModelArtifact artifact) {
        Path path = artifact.artifactPath().toAbsolutePath().normalize();
        if (!path.startsWith(root)) {
            throw new IllegalArgumentException("Model artifact escapes the app-private model directory");
        }
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Model artifact is missing: " + artifact.modelId());
        }
        String[] checksum = artifact.checksum().split(":", 2);
        boolean tree = "sha256-tree".equals(checksum[0]);
        if (tree != Files.isDirectory(path)) {
            throw new IllegalArgumentException("Checksum type does not match artifact: " + artifact.modelId());
        }
        if (!tree && !Files.isRegularFile(path)) {
            throw new IllegalArgumentException("Artifact must be a file: " + artifact.modelId());
        }
        String actual = tree ? hashTree(path) : hashFile(path);
        if (!checksum[1].equalsIgnoreCase(actual)) {
            throw new IllegalArgumentException("Checksum mismatch: " + artifact.modelId());
        }
        return path;
    }

    private static AndroidModelArtifact parse(JSONObject item, Path root) throws JSONException {
        String modelId = requiredString(item, "model_id");
        String runtime = requiredString(item, "runtime").toLowerCase(Locale.ROOT);
        if (!runtime.equals("qnn") && !runtime.equals("genie")) {
            throw new IllegalArgumentException("Unsupported Android runtime: " + runtime);
        }
        String relativePath = requiredString(item, "artifact_path");
        Path artifactPath = Path.of(relativePath);
        if (artifactPath.isAbsolute()) {
            throw new IllegalArgumentException("artifact_path must be relative to the Android model directory");
        }
        artifactPath = root.resolve(artifactPath).normalize();
        if (!artifactPath.startsWith(root)) {
            throw new IllegalArgumentException("artifact_path escapes the Android model directory");
        }
        String checksum = requiredString(item, "checksum");
        if (!CHECKSUM.matcher(checksum).matches()) {
            throw new IllegalArgumentException("Invalid SHA-256 checksum for " + modelId);
        }
        long minMemoryMb = item.getLong("min_memory_mb");
        int maxContextTokens = item.getInt("max_context_tokens");
        if (minMemoryMb < 0 || maxContextTokens <= 0) {
            throw new IllegalArgumentException("Invalid memory or context requirement for " + modelId);
        }
        float qualityScore = (float) item.optDouble("quality_score", 0.75);
        if (qualityScore < 0 || qualityScore > 1) {
            throw new IllegalArgumentException("quality_score must be between 0 and 1");
        }
        List<String> accelerators = stringList(item, "supported_accelerators", true);
        List<String> taskClasses = stringList(item, "task_classes", true);
        List<String> vectors = stringList(item, "steering_vector_ids", false);
        List<Integer> layers = integerList(item, "supported_steering_layers");
        boolean supportsSteering = item.optBoolean("supports_steering", !vectors.isEmpty());
        if (supportsSteering && vectors.isEmpty()) {
            throw new IllegalArgumentException("Steering model must declare steering_vector_ids");
        }
        String steeringMode = item.optString(
                "steering_mode", supportsSteering ? "runtime_vector" : "none");
        String behaviorProfileId = item.optString("behavior_profile_id", "");
        if (supportsSteering != steeringMode.equals("runtime_vector")) {
            throw new IllegalArgumentException(
                    "supports_steering is reserved for runtime_vector: " + modelId);
        }
        if ((steeringMode.equals("baked_profile") || steeringMode.equals("prompt_profile"))
                && behaviorProfileId.isBlank()) {
            throw new IllegalArgumentException(
                    steeringMode + " requires behavior_profile_id: " + modelId);
        }
        AndroidModelSegment segment = parseSegment(item.optJSONObject("split_boundary"));
        boolean supportsPipeline = item.optBoolean("supports_layer_pipeline", segment != null);
        if (supportsPipeline != (segment != null)) {
            throw new IllegalArgumentException(
                    "split_boundary and supports_layer_pipeline must agree for " + modelId);
        }
        return new AndroidModelArtifact(
                modelId,
                requiredString(item, "model_version"),
                runtime,
                artifactPath,
                checksum,
                requiredString(item, "tokenizer_id"),
                requiredString(item, "precision"),
                accelerators,
                minMemoryMb,
                maxContextTokens,
                supportsSteering,
                item.optBoolean("supports_data_parallel", false),
                supportsPipeline,
                requiredString(item, "model_family"),
                requiredString(item, "role"),
                taskClasses,
                qualityScore,
                vectors,
                layers,
                segment,
                item.optString("runtime_version", "unknown"),
                item.optString("artifact_id", modelId),
                steeringMode,
                behaviorProfileId,
                item.optString("target_compatibility_class", ""),
                item.optJSONObject("runtime_options") == null
                        ? "{}" : item.optJSONObject("runtime_options").toString());
    }

    private static AndroidModelSegment parseSegment(JSONObject item) throws JSONException {
        if (item == null) {
            return null;
        }
        int stageCount = item.optInt("stage_count", 0);
        int stageIndex = stageCount > 0 ? item.getInt("stage_index") : -1;
        Integer start = null;
        if (item.has("transformer_start_layer")) {
            start = Integer.valueOf(item.getInt("transformer_start_layer"));
        } else if (item.has("start_layer")) {
            start = Integer.valueOf(item.getInt("start_layer"));
        }
        Integer end = null;
        if (item.has("transformer_end_layer")) {
            end = Integer.valueOf(item.getInt("transformer_end_layer"));
        } else if (item.has("end_layer")) {
            end = Integer.valueOf(item.getInt("end_layer"));
        }
        int total = item.optInt("total_layers", 0);
        if (stageCount > 0 && (stageIndex < 0 || stageIndex >= stageCount)) {
            throw new IllegalArgumentException("Invalid pipeline stage index");
        }
        if ((start == null) != (end == null)) {
            throw new IllegalArgumentException("Incomplete transformer-layer range");
        }
        if (start != null && (start < 0 || start > end || (total > 0 && end > total))) {
            throw new IllegalArgumentException("Invalid transformer-layer range");
        }
        boolean includesEmbedding = item.optBoolean("includes_embedding", false);
        if (start == null && !includesEmbedding) {
            throw new IllegalArgumentException("Layerless stage must own embeddings");
        }
        return new AndroidModelSegment(
                requiredString(item, "pipeline_id"),
                stageIndex,
                stageCount,
                start,
                end,
                total,
                includesEmbedding,
                item.optBoolean("includes_lm_head", false),
                item.optString("input_tensor", ""),
                item.optString("output_tensor", ""),
                requiredString(item, "boundary_format"));
    }

    private static String requiredString(JSONObject item, String field) throws JSONException {
        String value = item.optString(field, "").trim();
        if (value.isEmpty()) {
            throw new IllegalArgumentException("Missing required model field: " + field);
        }
        return value;
    }

    private static List<String> stringList(JSONObject item, String field, boolean required)
            throws JSONException {
        JSONArray values = item.optJSONArray(field);
        if (values == null) {
            if (required) {
                throw new IllegalArgumentException("Missing required model list: " + field);
            }
            return List.of();
        }
        List<String> output = new ArrayList<>();
        for (int index = 0; index < values.length(); index++) {
            String value = values.optString(index, "").trim();
            if (value.isEmpty()) {
                throw new IllegalArgumentException("Invalid empty item in " + field);
            }
            output.add(value);
        }
        if (required && output.isEmpty()) {
            throw new IllegalArgumentException("Required model list is empty: " + field);
        }
        return output;
    }

    private static List<Integer> integerList(JSONObject item, String field) throws JSONException {
        JSONArray values = item.optJSONArray(field);
        if (values == null) {
            return List.of();
        }
        List<Integer> output = new ArrayList<>();
        for (int index = 0; index < values.length(); index++) {
            int value = values.getInt(index);
            if (value < 0) {
                throw new IllegalArgumentException("Invalid negative item in " + field);
            }
            output.add(value);
        }
        return output;
    }

    private static String hashFile(Path path) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (var input = Files.newInputStream(path)) {
                byte[] buffer = new byte[1024 * 1024];
                for (int read; (read = input.read(buffer)) != -1;) {
                    digest.update(buffer, 0, read);
                }
            }
            return hex(digest.digest());
        } catch (IOException | NoSuchAlgorithmException failure) {
            throw new IllegalArgumentException("Unable to hash artifact", failure);
        }
    }

    private static String hashTree(Path root) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (Stream<Path> entries = Files.walk(root)) {
                entries.filter(Files::isRegularFile)
                        .sorted(Comparator.comparing(
                                path -> root.relativize(path).toString().replace('\\', '/')))
                        .forEach(path -> updateTreeDigest(digest, root, path));
            }
            return hex(digest.digest());
        } catch (IOException | NoSuchAlgorithmException failure) {
            throw new IllegalArgumentException("Unable to hash artifact tree", failure);
        }
    }

    private static void updateTreeDigest(MessageDigest digest, Path root, Path path) {
        digest.update(root.relativize(path).toString().replace('\\', '/').getBytes(StandardCharsets.UTF_8));
        digest.update((byte) 0);
        try (var input = Files.newInputStream(path)) {
            byte[] buffer = new byte[1024 * 1024];
            for (int read; (read = input.read(buffer)) != -1;) {
                digest.update(buffer, 0, read);
            }
            digest.update((byte) 0);
        } catch (IOException failure) {
            throw new IllegalArgumentException("Unable to hash artifact tree", failure);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder output = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            output.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return output.toString();
    }
}

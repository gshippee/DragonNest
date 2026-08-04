package com.dragonnest.agent;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.stream.Stream;

/** Installs a signed APK's optional model assets into app-private storage once. */
final class AndroidModelAssetInstaller {
    private static final String ASSET_ROOT = "models";

    private AndroidModelAssetInstaller() { }

    static void installIfAbsent(Context context) throws IOException {
        Path destination = context.getFilesDir().toPath()
                .resolve(AndroidArtifactRegistry.MODEL_DIRECTORY);
        if (isCompleteInstall(destination)) {
            return;
        }
        String[] topLevel = context.getAssets().list(ASSET_ROOT);
        if (topLevel == null || topLevel.length == 0) {
            return;
        }
        Path staging = destination.resolveSibling(destination.getFileName() + ".installing");
        deleteTree(staging);
        copyTree(context.getAssets(), ASSET_ROOT, staging);
        Files.write(staging.resolve(".installed"), new byte[]{'1'});
        deleteTree(destination);
        try {
            Files.move(staging, destination, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException unavailable) {
            Files.move(staging, destination);
        }
    }

    private static boolean isCompleteInstall(Path destination) {
        return Files.isRegularFile(destination.resolve("manifest.json"))
                && Files.isRegularFile(destination.resolve(".installed"));
    }

    private static void copyTree(AssetManager assets, String source, Path destination)
            throws IOException {
        String[] children = assets.list(source);
        if (children != null && children.length > 0) {
            Files.createDirectories(destination);
            for (String child : children) {
                copyTree(assets, source + "/" + child, destination.resolve(child));
            }
            return;
        }
        Files.createDirectories(destination.getParent());
        try (InputStream input = assets.open(source);
                OutputStream output = Files.newOutputStream(destination)) {
            byte[] buffer = new byte[1024 * 1024];
            for (int read; (read = input.read(buffer)) != -1;) {
                output.write(buffer, 0, read);
            }
        }
    }

    private static void deleteTree(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return;
        }
        try (Stream<Path> entries = Files.walk(directory)) {
            entries.sorted(Comparator.reverseOrder()).forEach(path -> {
                try {
                    Files.delete(path);
                } catch (IOException failure) {
                    throw new IllegalStateException("Unable to replace Android model assets", failure);
                }
            });
        } catch (IllegalStateException failure) {
            if (failure.getCause() instanceof IOException ioFailure) {
                throw ioFailure;
            }
            throw failure;
        }
    }
}

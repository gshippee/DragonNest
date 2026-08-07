import org.gradle.api.file.SourceDirectorySet
import org.gradle.api.plugins.ExtensionAware

plugins {
    id("com.android.application")
    id("com.google.protobuf")
    id("org.jetbrains.kotlin.android")
}

val qairtSdkRoot = providers.gradleProperty("qairtSdkRoot")
    .orElse(providers.environmentVariable("DRAGONNEST_QAIRT_SDK_ROOT"))
    .orNull
val includeModelArtifacts = providers.gradleProperty("includeModelArtifacts")
    .orElse("false")
    .map { it.toBoolean() }
    .get()
val includeS25GenieXRuntime = providers.gradleProperty("includeS25GenieXRuntime")
    .orElse(providers.environmentVariable("DRAGONNEST_ANDROID_INCLUDE_S25_GENIEX"))
    .orElse("false")
    .map { it.toBoolean() }
    .get()

android {
    namespace = "com.dragonnest.agent"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.dragonnest.agent"
        minSdk = if (includeS25GenieXRuntime) 27 else 26
        targetSdk = 35
        versionCode = 8
        versionName = "0.1.7"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField(
            "boolean",
            "DRAGONNEST_ENABLE_MOCK_RUNTIME",
            (!includeS25GenieXRuntime).toString(),
        )
    }

    compileOptions {
        isCoreLibraryDesugaringEnabled = true
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    // Genie context binaries are memory-mapped at runtime and can be several
    // gigabytes. Compressing them wastes heap during packaging and blocks mmap.
    androidResources {
        noCompress += "bin"
    }

    // The open-source APK remains buildable without Qualcomm's SDK. A hardware
    // build opts in explicitly and compiles the JNI bridge for physical arm64 devices.
    if (includeModelArtifacts && !qairtSdkRoot.isNullOrBlank()) {
        defaultConfig {
            ndk {
                abiFilters += "arm64-v8a"
            }
            externalNativeBuild {
                cmake {
                    arguments += "-DQAIRT_SDK_ROOT=$qairtSdkRoot"
                }
            }
        }
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
            }
        }
    }
}

if (includeModelArtifacts) {
    android.sourceSets.getByName("main").jniLibs.srcDir("../vendor/jniLibs")
    android.sourceSets.getByName("main").assets.srcDir("../vendor/model-assets")
}

// The forked GenieX closure that serves the "genie_aux" runtime-steering
// runtime. Its sonames were renamed to libgnxfrk*.so by
// scripts/artifact_tools/stage_steering_native_closure.py precisely so it can
// sit beside the stock geniex-android AAR instead of displacing it, so
// packaging it cannot change how the accepted Base path executes.
if (includeS25GenieXRuntime) {
    android.sourceSets.getByName("main").jniLibs.srcDir("../vendor/steering-jniLibs")
}

val mainProto = (
    android.sourceSets.getByName("main") as ExtensionAware
).extensions.getByName("proto") as SourceDirectorySet
mainProto.srcDir("../../proto")

// Neither protoc nor protoc-gen-grpc-java publish a native Windows-on-ARM
// binary; the windows-x86_64 one runs fine under Windows' built-in x64
// emulation. Pin to it explicitly on windows-aarch64 hosts instead of
// letting the protobuf-gradle-plugin auto-detect a classifier
// ("windows-aarch_64") that doesn't exist upstream.
val isWindowsOnArm = run {
    val osName = System.getProperty("os.name").lowercase()
    val osArch = System.getProperty("os.arch").lowercase()
    osName.contains("windows") && (osArch.contains("aarch64") || osArch.contains("arm64"))
}

protobuf {
    protoc {
        artifact = if (isWindowsOnArm) {
            "com.google.protobuf:protoc:3.25.5:windows-x86_64@exe"
        } else {
            "com.google.protobuf:protoc:3.25.5"
        }
    }
    plugins {
        create("grpc") {
            artifact = if (isWindowsOnArm) {
                "io.grpc:protoc-gen-grpc-java:1.68.1:windows-x86_64@exe"
            } else {
                "io.grpc:protoc-gen-grpc-java:1.68.1"
            }
        }
    }
    generateProtoTasks {
        all().configureEach {
            builtins {
                maybeCreate("java").apply {
                    option("lite")
                }
            }
            plugins {
                maybeCreate("grpc").apply {
                    option("lite")
                }
            }
        }
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.4")
    implementation("androidx.activity:activity-compose:1.10.0")
    implementation(composeBom)
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.navigation:navigation-compose:2.8.5")
    implementation("io.grpc:grpc-okhttp:1.68.1")
    implementation("io.grpc:grpc-protobuf-lite:1.68.1")
    implementation("io.grpc:grpc-stub:1.68.1")
    implementation("com.google.protobuf:protobuf-javalite:3.25.5")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("com.journeyapps:zxing-android-embedded:4.3.0")
    // Compile the fail-closed bridge in every build, but package Qualcomm's
    // licensed GenieX runtime closure only for an explicit hardware build.
    compileOnly("com.qualcomm.qti:geniex-android:0.3.5")
    if (includeS25GenieXRuntime) {
        implementation("com.qualcomm.qti:geniex-android:0.3.5")
    }
    compileOnly("org.apache.tomcat:annotations-api:6.0.53")
    implementation(fileTree("../vendor/aars") { include("*.aar", "*.jar") })

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    debugImplementation("androidx.compose.ui:ui-tooling")
    androidTestImplementation(composeBom)
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}

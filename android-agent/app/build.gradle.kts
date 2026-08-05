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
    .orElse("true")
    .map { it.toBoolean() }
    .get()

android {
    namespace = "com.dragonnest.agent"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.dragonnest.agent"
        minSdk = 26
        targetSdk = 35
        versionCode = 8
        versionName = "0.1.7"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
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

val mainProto = (
    android.sourceSets.getByName("main") as ExtensionAware
).extensions.getByName("proto") as SourceDirectorySet
mainProto.srcDir("../../proto")

protobuf {
    protoc {
        artifact = "com.google.protobuf:protoc:3.25.5"
    }
    plugins {
        create("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:1.68.1"
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
    implementation("com.journeyapps:zxing-android-embedded:4.3.0")
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

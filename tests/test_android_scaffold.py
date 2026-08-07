from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
ANDROID = "{http://schemas.android.com/apk/res/android}"


def test_android_agent_manifest_and_platform_hooks_are_present():
    android_root = ROOT / "android-agent"
    manifest = ElementTree.parse(
        android_root / "app/src/main/AndroidManifest.xml"
    ).getroot()
    permissions = {
        item.attrib[f"{ANDROID}name"] for item in manifest.findall("uses-permission")
    }
    application = manifest.find("application")
    assert application is not None
    service = application.find("service")
    receiver = application.find("receiver")

    assert "android.permission.FOREGROUND_SERVICE" in permissions
    assert "android.permission.ACCESS_NETWORK_STATE" in permissions
    assert "android.permission.CAMERA" in permissions
    assert service is not None
    assert service.attrib[f"{ANDROID}foregroundServiceType"] == "dataSync"
    assert receiver is not None

    sources = android_root / "app/src/main/java/com/dragonnest/agent"
    runtime = (sources / "AgentRuntime.java").read_text(encoding="utf-8")
    foreground = (sources / "AgentForegroundService.java").read_text(
        encoding="utf-8"
    )
    enrollment = (sources / "EnrollmentStore.java").read_text(encoding="utf-8")
    connection = (sources / "GrpcAgentConnection.java").read_text(encoding="utf-8")
    settings = (sources / "AgentSettingsActivity.kt").read_text(encoding="utf-8")
    app = (sources / "PersonaCareApp.kt").read_text(encoding="utf-8")
    view_model = (sources / "PersonaCareViewModel.kt").read_text(encoding="utf-8")
    payload = (sources / "EnrollmentPayload.java").read_text(encoding="utf-8")
    inventory = (sources / "AndroidHardwareInventory.java").read_text(
        encoding="utf-8"
    )
    runtime_catalog = (sources / "AndroidRuntimeCatalog.java").read_text(
        encoding="utf-8"
    )
    artifacts = (sources / "AndroidArtifactRegistry.java").read_text(
        encoding="utf-8"
    )
    executor = (sources / "MockAndroidTaskExecutor.java").read_text(
        encoding="utf-8"
    )
    genie_bridge = (
        sources / "vendor/GenieRuntimeBridge.java"
    ).read_text(encoding="utf-8")
    qnn_bridge = (
        sources / "vendor/QnnRuntimeBridge.java"
    ).read_text(encoding="utf-8")
    genie_jni = (
        android_root / "app/src/main/cpp/genie_jni.cpp"
    ).read_text(encoding="utf-8")

    assert "reconnectBackoffSeconds * 2" in runtime
    assert "sendShutdown" in runtime
    assert "registerDefaultNetworkCallback" in foreground
    assert "runtime.onNetworkChanged()" in foreground
    assert "runtime.onSimulationChanged()" in foreground
    assert 'KeyStore.getInstance(ANDROID_KEYSTORE)' in enrollment
    assert "BrainControlGrpc.newStub" in connection
    assert "setRegisterDevice" in connection
    assert "getDeviceCredential" in connection
    assert "setTaskResult" in connection
    assert "setPartialTaskResult" in connection
    assert "setPipelineStageResult" in connection
    assert "Opening gRPC stream to" in connection
    assert "RegistrationAccepted" in connection
    assert "Android mock result" in executor
    assert "PersonaCareApp(viewModel)" in settings
    assert "Scan enrollment QR" in app
    assert "EnrollmentCaptureActivity.scanIntent" in app
    assert "About you and your preferences" in app
    assert "Keep on phone" in app
    assert "Ran on" in app
    assert "UserProfileStore" in view_model
    assert "BrainTaskClient" in view_model
    assert "EnrollmentCaptureActivity" in (
        android_root / "app/src/main/AndroidManifest.xml"
    ).read_text(encoding="utf-8")
    capture = (sources / "EnrollmentCaptureActivity.java").read_text(
        encoding="utf-8"
    )
    assert "requestPermissions(new String[]{Manifest.permission.CAMERA}" in capture
    assert "DecoratedBarcodeView" in capture
    assert "decodeSingle(new BarcodeCallback()" in capture
    assert ".setAction(SCAN_ACTION)" in capture
    assert "CAMERA_PERMISSION_REQUEST" in capture
    assert "dragonnest.enrollment" in payload
    assert "Build.SOC_MODEL" in inventory
    assert "setNpuStatus(runtimeCatalog.npuStatus())" in inventory
    assert "registry.isVerified(artifact)" in runtime_catalog
    assert "new QnnAndroidTaskExecutor" in runtime_catalog
    assert "new GenieAndroidTaskExecutor" in runtime_catalog
    assert "Checksum mismatch" in artifacts
    asset_installer = (sources / "AndroidModelAssetInstaller.java").read_text(
        encoding="utf-8"
    )
    assert "ATOMIC_MOVE" in asset_installer
    assert '".installed"' in asset_installer
    assert "nativeProbe" in genie_bridge
    assert "genie_config.json" in genie_bridge
    assert "nativeCreateSession" in qnn_bridge
    assert "nativeExecutionReady" in qnn_bridge
    assert "PIPELINE_PREFILL" in qnn_bridge
    assert "PIPELINE_RESET" in qnn_bridge
    assert "GenieDialog_query" in genie_jni
    gradle = (android_root / "app/build.gradle.kts").read_text(encoding="utf-8")
    assert 'jniLibs.srcDir("../vendor/jniLibs")' in gradle
    assert 'assets.srcDir("../vendor/model-assets")' in gradle
    assert "includeModelArtifacts" in gradle
    assert "compose = true" in gradle
    assert 'noCompress += "bin"' in gradle
    assert "setHardware(hardwareInventory.snapshot())" in (
        sources / "AgentProfile.java"
    ).read_text(encoding="utf-8")
    assert "setPersonalProfile(userProfile.registration())" in (
        sources / "AgentProfile.java"
    ).read_text(encoding="utf-8")

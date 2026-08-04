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
    settings = (sources / "AgentSettingsActivity.java").read_text(encoding="utf-8")
    payload = (sources / "EnrollmentPayload.java").read_text(encoding="utf-8")
    executor = (sources / "MockAndroidTaskExecutor.java").read_text(
        encoding="utf-8"
    )

    assert "reconnectBackoffSeconds * 2" in runtime
    assert "sendShutdown" in runtime
    assert "registerDefaultNetworkCallback" in foreground
    assert "runtime.onNetworkChanged()" in foreground
    assert 'KeyStore.getInstance(ANDROID_KEYSTORE)' in enrollment
    assert "BrainControlGrpc.newStub" in connection
    assert "setRegisterDevice" in connection
    assert "getDeviceCredential" in connection
    assert "setTaskResult" in connection
    assert "setPartialTaskResult" in connection
    assert "setPipelineStageResult" in connection
    assert "Android mock result" in executor
    assert "IntentIntegrator" in settings
    assert "Scan enrollment QR" in settings
    assert "dragonnest.enrollment" in payload

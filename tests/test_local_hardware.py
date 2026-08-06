from __future__ import annotations

import dataclasses
import subprocess

from dragon_nest import local_hardware
from dragon_nest.dashboard import HardwareInventoryPayload
from dragon_nest.models import HardwareInventory


def test_npu_name_for_matches_known_chipset_strings():
    assert local_hardware._npu_name_for("Qualcomm(R) Snapdragon(R) X 12-core X1E80100") == "Hexagon NPU (v73)"
    assert local_hardware._npu_name_for("QCS2210") == "Hexagon DSP (v66)"
    assert local_hardware._npu_name_for("Intel(R) Core(TM) i7-1365U") == ""


def test_soc_info_only_claims_qualcomm_when_the_string_says_so():
    assert local_hardware._soc_info("Snapdragon X Elite X1E80100") == ("Qualcomm", "Snapdragon X Elite X1E80100")
    assert local_hardware._soc_info("QCS9075") == ("Qualcomm", "QCS9075")
    assert local_hardware._soc_info("Intel(R) Core(TM) i7-1365U") == ("", "")
    assert local_hardware._soc_info("") == ("", "")


def test_cpu_abi_maps_common_architectures(monkeypatch):
    monkeypatch.setattr(local_hardware.platform, "machine", lambda: "AMD64")
    assert local_hardware._cpu_abi() == ("x86_64",)
    monkeypatch.setattr(local_hardware.platform, "machine", lambda: "aarch64")
    assert local_hardware._cpu_abi() == ("arm64-v8a",)
    monkeypatch.setattr(local_hardware.platform, "machine", lambda: "armv7l")
    assert local_hardware._cpu_abi() == ("armeabi-v7a",)
    monkeypatch.setattr(local_hardware.platform, "machine", lambda: "")
    assert local_hardware._cpu_abi() == ()


def test_npu_status_is_unavailable_without_an_sdk():
    assert local_hardware._npu_status(None) == "unavailable"


def test_npu_status_is_not_probed_when_sdk_present_but_no_validator(monkeypatch):
    monkeypatch.setattr(local_hardware.shutil, "which", lambda name: None)
    assert local_hardware._npu_status("/opt/qcom/aistack/qairt/2.31.0") == "not_probed"


def test_npu_status_is_available_when_validator_confirms_htp(monkeypatch):
    monkeypatch.setattr(local_hardware.shutil, "which", lambda name: "/usr/bin/qnn-platform-validator")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="Backend HTP: PASS\n", stderr="")

    monkeypatch.setattr(local_hardware.subprocess, "run", fake_run)
    assert local_hardware._npu_status("/opt/qcom/aistack/qairt/2.31.0") == "available"


def test_npu_status_is_unavailable_when_validator_finds_nothing(monkeypatch):
    monkeypatch.setattr(local_hardware.shutil, "which", lambda name: "/usr/bin/qnn-platform-validator")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="Backend CPU: PASS\n", stderr="")

    monkeypatch.setattr(local_hardware.subprocess, "run", fake_run)
    assert local_hardware._npu_status("/opt/qcom/aistack/qairt/2.31.0") == "unavailable"


def test_probe_local_hardware_returns_a_hardware_inventory_shaped_payload():
    info = local_hardware.probe_local_hardware()

    assert set(info.keys()) == {"hardware", "platform", "total_memory_mb", "display_name"}
    assert info["platform"] in {"windows", "linux", "macos"}
    assert isinstance(info["total_memory_mb"], int)
    assert isinstance(info["display_name"], str)

    hardware_fields = {f.name for f in dataclasses.fields(HardwareInventory)}
    assert set(info["hardware"].keys()) == hardware_fields
    assert info["hardware"]["npu_status"] in {"available", "not_probed", "unavailable"}

    # Round-trips through the dashboard's pydantic payload and the dataclass
    # DeviceRegistration uses, exactly as api_register_rest_device does.
    payload = HardwareInventoryPayload(**info["hardware"])
    HardwareInventory(**payload.model_dump())

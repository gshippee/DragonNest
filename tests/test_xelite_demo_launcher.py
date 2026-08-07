from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_same_host_xelite_launcher_uses_verified_worker_and_safe_timeout():
    launcher = (ROOT / "scripts/run_xelite_demo.ps1").read_text(encoding="utf-8")

    assert "run_xelite_worker.ps1" in launcher
    assert "-Brain '127.0.0.1:50051'" in launcher
    assert "--address 0.0.0.0:50051" in launcher
    assert "--http-host 0.0.0.0" in launcher
    assert "--default-task-timeout-ms 75000" in launcher
    assert launcher.count("Start-Process") == 2
    assert "-WindowStyle Normal" in launcher
    assert "New-EnrollmentToken" in launcher
    assert 'EnrollmentToken -eq "dev-token"' in launcher
    assert "New-NetFirewallRule" not in launcher
    assert "run_openai_adapter" not in launcher

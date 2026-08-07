from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dragon_nest.runtime import melotts_runner
from dragon_nest.runtime.melotts_worker import RESULT_MARKER


@pytest.fixture
def interpreter(monkeypatch):
    """Point the runner at a file that exists so it gets past its pre-flight
    check; the subprocess itself is always stubbed."""
    monkeypatch.setattr(melotts_runner, "tts_python_executable", lambda: Path(__file__))


def _completed(returncode: int, stdout: str, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _marked(payload: dict) -> str:
    return f"{RESULT_MARKER} {json.dumps(payload)}\n"


def _argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_parse_result_finds_marker_past_model_load_noise():
    """torch/melo/jieba write to stdout while loading, so the result is found
    by marker rather than by assuming it is the only line of output."""
    stdout = (
        "Building prefix dict from the default dictionary ...\n"
        "Loading model cost 0.7 seconds.\n"
        + _marked({"ok": True, "output": "out.wav", "sample_rate": 44100, "chunks": 2})
    )
    assert melotts_runner._parse_result(stdout)["chunks"] == 2


def test_parse_result_none_stdout_raises_clean_error():
    """subprocess.run's .stdout can come back None on Windows under DSP session
    contention even at returncode 0; that must not leak an AttributeError."""
    with pytest.raises(melotts_runner.TtsError, match="no result line"):
        melotts_runner._parse_result(None)


def test_parse_result_rejects_unreadable_payload():
    with pytest.raises(melotts_runner.TtsError, match="unreadable"):
        melotts_runner._parse_result(f"{RESULT_MARKER} not-json\n")


def test_synthesize_writes_wav_and_cleans_up_its_text_file(tmp_path, monkeypatch, interpreter):
    output = tmp_path / "speech.wav"
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["text"] = Path(_argument(command, "--text-file")).read_text(encoding="utf-8")
        Path(_argument(command, "--output")).write_bytes(b"RIFFfake")
        return _completed(0, _marked({"ok": True, "output": str(output), "chunks": 1}))

    monkeypatch.setattr(melotts_runner.subprocess, "run", fake_run)

    assert melotts_runner.synthesize("Hello there.", output) == output
    assert seen["text"] == "Hello there."
    assert _argument(seen["command"], "--backend") == "htp"
    # The text goes through a file (the Windows command-line length limit would
    # otherwise cap how long a response can be spoken) and must not be left behind.
    assert not Path(_argument(seen["command"], "--text-file")).exists()


def test_synthesize_maps_unavailable_worker_result_to_unavailable_error(
    tmp_path, monkeypatch, interpreter
):
    """A host missing the model files or qai_env is a 503, not a 500: there is
    nothing to retry until the machine is provisioned."""

    def fake_run(command, **kwargs):
        return _completed(
            melotts_runner.EXIT_UNAVAILABLE,
            _marked({"ok": False, "kind": "unavailable", "error": "MeloTTS model files missing"}),
        )

    monkeypatch.setattr(melotts_runner.subprocess, "run", fake_run)

    with pytest.raises(melotts_runner.TtsUnavailableError, match="model files missing"):
        melotts_runner.synthesize("Hello.", tmp_path / "out.wav")


def test_synthesize_maps_a_synthesis_failure_to_a_plain_tts_error(
    tmp_path, monkeypatch, interpreter
):
    def fake_run(command, **kwargs):
        return _completed(1, _marked({"ok": False, "kind": "failed", "error": "decoder rejected chunk"}))

    monkeypatch.setattr(melotts_runner.subprocess, "run", fake_run)

    with pytest.raises(melotts_runner.TtsError, match="decoder rejected chunk") as caught:
        melotts_runner.synthesize("Hello.", tmp_path / "out.wav")
    assert not isinstance(caught.value, melotts_runner.TtsUnavailableError)


def test_synthesize_explains_the_adsp_fail_fast_exit_code(tmp_path, monkeypatch, interpreter):
    """0xC0000409 means qnn-net-run could not resolve the Hexagon skel library.
    It crashes after the graph appears to compile cleanly, so the error must
    name ADSP_LIBRARY_PATH rather than leaving a bare exit code."""

    def fake_run(command, **kwargs):
        return _completed(0xC0000409, "", "DspTransport.openSession qnn_open failed")

    monkeypatch.setattr(melotts_runner.subprocess, "run", fake_run)

    with pytest.raises(melotts_runner.TtsError, match="ADSP_LIBRARY_PATH"):
        melotts_runner.synthesize("Hello.", tmp_path / "out.wav")


def test_synthesize_reports_a_timeout_as_a_wedged_dsp_session(tmp_path, monkeypatch, interpreter):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=1, output="", stderr="")

    monkeypatch.setattr(melotts_runner.subprocess, "run", fake_run)

    with pytest.raises(melotts_runner.TtsError, match="NPU/DSP session"):
        melotts_runner.synthesize("Hello.", tmp_path / "out.wav", timeout_sec=1)


def test_synthesize_reports_a_missing_interpreter_as_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAGONNEST_TTS_PYTHON", str(tmp_path / "nope" / "python.exe"))
    with pytest.raises(melotts_runner.TtsUnavailableError, match="interpreter not found"):
        melotts_runner.synthesize("Hello.", tmp_path / "out.wav")


def test_synthesize_rejects_empty_text(tmp_path):
    with pytest.raises(melotts_runner.TtsError, match="no text"):
        melotts_runner.synthesize("   ", tmp_path / "out.wav")


def test_synthesize_fails_when_the_worker_claims_success_without_a_file(
    tmp_path, monkeypatch, interpreter
):
    def fake_run(command, **kwargs):
        return _completed(0, _marked({"ok": True, "output": "missing.wav"}))

    monkeypatch.setattr(melotts_runner.subprocess, "run", fake_run)

    with pytest.raises(melotts_runner.TtsError, match="wrote no file"):
        melotts_runner.synthesize("Hello.", tmp_path / "out.wav")

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from dragon_nest import dashboard
from dragon_nest.runtime import melotts_runner, npu_lease
from dragon_nest.transport.brain import BrainService


WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt "


@pytest.fixture
def speech_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "SPEECH_CACHE_DIR", tmp_path / "speech")
    return tmp_path / "speech"


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=dashboard.create_dashboard_app(BrainService())),
        base_url="http://test",
    )


def test_speech_endpoint_returns_wav_audio(monkeypatch, speech_cache):
    calls: list[str] = []

    def fake_synthesize(text, output_path, **kwargs):
        calls.append(text)
        Path(output_path).write_bytes(WAV_BYTES)
        return Path(output_path)

    monkeypatch.setattr(melotts_runner, "synthesize", fake_synthesize)

    async def scenario() -> httpx.Response:
        async with _client() as client:
            return await client.post("/api/speech", json={"text": "Take one tablet daily."})

    response = asyncio.run(scenario())

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == WAV_BYTES
    assert calls == ["Take one tablet daily."]


def test_speech_endpoint_serves_repeat_requests_from_cache(monkeypatch, speech_cache):
    """Synthesis is seconds-scale on the NPU, so replaying the same answer must
    not re-run the model."""
    calls: list[str] = []

    def fake_synthesize(text, output_path, **kwargs):
        calls.append(text)
        Path(output_path).write_bytes(WAV_BYTES)
        return Path(output_path)

    monkeypatch.setattr(melotts_runner, "synthesize", fake_synthesize)

    async def scenario() -> list[httpx.Response]:
        async with _client() as client:
            first = await client.post("/api/speech", json={"text": "Same answer."})
            second = await client.post("/api/speech", json={"text": "Same answer."})
            other = await client.post("/api/speech", json={"text": "Different answer."})
            return [first, second, other]

    first, second, other = asyncio.run(scenario())

    assert [first.status_code, second.status_code, other.status_code] == [200, 200, 200]
    assert second.content == first.content
    assert calls == ["Same answer.", "Different answer."]


def test_speech_endpoint_reports_an_unprovisioned_host_as_503(monkeypatch, speech_cache):
    def fake_synthesize(text, output_path, **kwargs):
        raise melotts_runner.TtsUnavailableError("speech interpreter not found")

    monkeypatch.setattr(melotts_runner, "synthesize", fake_synthesize)

    async def scenario() -> httpx.Response:
        async with _client() as client:
            return await client.post("/api/speech", json={"text": "Hello."})

    response = asyncio.run(scenario())

    assert response.status_code == 503
    assert "interpreter not found" in response.json()["detail"]


def test_speech_endpoint_reports_a_busy_npu_as_409_not_a_failure(monkeypatch, speech_cache):
    """Speech yields to the pinned language model rather than competing for the
    DSP session, so a busy NPU is "come back in a moment", not an error."""

    def fake_synthesize(text, output_path, **kwargs):
        raise npu_lease.NpuBusyError("the NPU is busy with qwen3-4b-genie")

    monkeypatch.setattr(melotts_runner, "synthesize", fake_synthesize)

    async def scenario() -> httpx.Response:
        async with _client() as client:
            return await client.post("/api/speech", json={"text": "Hello."})

    response = asyncio.run(scenario())

    assert response.status_code == 409
    assert response.headers["retry-after"] == "5"
    assert not list(speech_cache.glob("*.wav")), "a declined run must not cache audio"


def test_speech_endpoint_reports_a_synthesis_failure_as_500(monkeypatch, speech_cache):
    def fake_synthesize(text, output_path, **kwargs):
        raise melotts_runner.TtsError("decoder rejected chunk")

    monkeypatch.setattr(melotts_runner, "synthesize", fake_synthesize)

    async def scenario() -> httpx.Response:
        async with _client() as client:
            return await client.post("/api/speech", json={"text": "Hello."})

    response = asyncio.run(scenario())

    assert response.status_code == 500
    assert "decoder rejected chunk" in response.json()["detail"]


def test_speech_endpoint_rejects_empty_and_oversized_text(speech_cache):
    async def scenario() -> list[httpx.Response]:
        async with _client() as client:
            empty = await client.post("/api/speech", json={"text": ""})
            huge = await client.post("/api/speech", json={"text": "a" * 4001})
            return [empty, huge]

    empty, huge = asyncio.run(scenario())

    assert empty.status_code == 422
    assert huge.status_code == 422


def test_speech_cache_is_bounded(speech_cache):
    speech_cache.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        (speech_cache / f"{index:032x}.wav").write_bytes(WAV_BYTES)

    dashboard._prune_speech_cache(max_files=2)

    assert len(list(speech_cache.glob("*.wav"))) == 2

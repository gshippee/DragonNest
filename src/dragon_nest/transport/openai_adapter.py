"""HTTP shim that exposes an OpenAI-compatible chat API as a DragonNest
HTTP endpoint device (see transport/http_device.py for the contract this
implements: GET /health, GET /info, POST /execute, POST /cancel).

DragonNest's HttpDeviceSession speaks its own JSON contract, not the OpenAI
chat-completions shape, so a third-party OpenAI-compliant provider (e.g.
Cirrascale's Inference Cloud) cannot be registered directly as an HTTP
endpoint device. This adapter sits in between: the brain talks DragonNest's
contract to the adapter (over localhost), and the adapter translates that
into OpenAI-style /chat/completions calls against the real provider.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI


@dataclass(frozen=True)
class OpenAIAdapterModel:
    model_id: str
    max_context_tokens: int = 8192
    model_family: str = ""
    role: str = "general"
    task_classes: tuple[str, ...] = ("chat_qa",)


@dataclass(frozen=True)
class OpenAIAdapterConfig:
    base_url: str
    api_key_env: str
    models: tuple[OpenAIAdapterModel, ...]
    device_id: str = "openai-adapter"
    display_name: str = "OpenAI-compatible endpoint"
    platform: str = "cloud"
    request_timeout_seconds: float = 60.0
    runtime_name: str = "openai_adapter"


def _api_key(config: OpenAIAdapterConfig) -> str:
    return os.environ.get(config.api_key_env, "")


def _model_payload(model: OpenAIAdapterModel) -> dict[str, object]:
    return {
        "model_id": model.model_id,
        "model_family": model.model_family or model.model_id,
        "role": model.role,
        "task_classes": list(model.task_classes),
        "max_context_tokens": model.max_context_tokens,
        "warm": True,
        "quality_score": 0.7,
        "runtime_name": "http",
        "supported_accelerators": ["cloud"],
    }


def _task_error(task_id: str, attempt_id: str, code: str, message: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "success": False,
        "output_text": "",
        "error_code": code,
        "error_message": message,
    }


async def _chat_completion(
    client: httpx.AsyncClient,
    config: OpenAIAdapterConfig,
    model_id: str,
    request_text: str,
    timeout_seconds: float,
) -> dict[str, object]:
    api_key = _api_key(config)
    if not api_key:
        return {
            "success": False,
            "error_code": "MISSING_API_KEY",
            "error_message": f"environment variable {config.api_key_env!r} is not set",
        }
    started = time.monotonic()
    try:
        response = await client.post(
            f"{config.base_url.rstrip('/')}/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": request_text}],
                "stream": False,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=min(timeout_seconds, config.request_timeout_seconds),
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as exc:
        return {
            "success": False,
            "error_code": f"HTTP_{exc.response.status_code}",
            "error_message": exc.response.text[:500],
        }
    except (httpx.HTTPError, ValueError) as exc:
        return {"success": False, "error_code": "REQUEST_FAILED", "error_message": str(exc)}
    latency_ms = int((time.monotonic() - started) * 1000)
    choices = body.get("choices") or []
    if not choices:
        return {
            "success": False,
            "error_code": "EMPTY_RESPONSE",
            "error_message": "provider returned no choices",
        }
    output_text = str((choices[0].get("message") or {}).get("content", ""))
    return {"success": True, "output_text": output_text, "latency_ms": latency_ms}


def create_openai_adapter_app(config: OpenAIAdapterConfig) -> FastAPI:
    app = FastAPI(title="DragonNest OpenAI Adapter", version="0.1.0")
    app.state.http_client = httpx.AsyncClient()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.http_client.aclose()

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "battery_pct": -1,
            "charging": False,
            "thermal_level": -1,
            "cpu_utilization": -1,
            "accelerator_utilization": -1,
            "gpu_utilization": -1,
            "npu_utilization": -1,
            "available_memory_mb": 0,
            "network_rtt_ms": -1,
            "reachable": True,
        }

    @app.get("/info")
    async def info() -> dict[str, object]:
        return {
            "display_name": config.display_name,
            "device_type": "endpoint",
            "platform": config.platform,
            "total_memory_mb": 0,
            "models": [_model_payload(model) for model in config.models],
            "hardware": {},
        }

    async def _execute(payload: dict[str, object]) -> dict[str, object]:
        task_id = str(payload.get("task_id", ""))
        attempt_id = str(payload.get("attempt_id", ""))
        model_id = str(payload.get("model_id", "")) or config.models[0].model_id
        request_text = str(payload.get("request_text", ""))
        timeout_ms = int(payload.get("timeout_ms") or 30000)
        result = await _chat_completion(
            app.state.http_client, config, model_id, request_text, timeout_ms / 1000
        )
        if not result["success"]:
            body = _task_error(
                task_id, attempt_id, str(result["error_code"]), str(result["error_message"])
            )
        else:
            body = {
                "task_id": task_id,
                "attempt_id": attempt_id,
                "success": True,
                "output_text": result["output_text"],
            }
        body["metrics"] = {
            "model_id": model_id,
            "model_version": "",
            "runtime_name": config.runtime_name,
            "runtime_version": "",
            "accelerator": "cloud",
            "execution_latency_ms": int(result.get("latency_ms", 0)),
            "error_code": body.get("error_code", ""),
            "error_message": body.get("error_message", ""),
        }
        return body

    @app.post("/execute")
    async def execute(payload: dict[str, object]) -> dict[str, object]:
        return await _execute(payload)

    @app.post("/execute_shard")
    async def execute_shard(payload: dict[str, object]) -> dict[str, object]:
        return await _execute(payload)

    @app.post("/execute_pipeline_stage")
    async def execute_pipeline_stage(payload: dict[str, object]) -> dict[str, object]:
        body = await _execute(payload)
        body["final_stage"] = True
        return body

    @app.post("/cancel")
    async def cancel(payload: dict[str, object]) -> dict[str, object]:
        del payload
        return {}

    return app

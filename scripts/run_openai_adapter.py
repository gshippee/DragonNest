"""Run the OpenAI-compatible adapter so DragonNest's brain can register a
third-party OpenAI-compliant provider (e.g. Cirrascale's Inference Cloud) as
an HTTP endpoint device.

The brain's HTTP endpoint transport speaks DragonNest's own JSON contract
(/health, /info, /execute), not OpenAI's /chat/completions shape, so this
adapter runs as a small local translator between the two. Point the brain's
"Add Endpoint" dialog at this adapter's address (default
http://127.0.0.1:8090), not at the provider directly.

Example:
    DRAGONNEST_OPENAI_API_KEY=<your key> python scripts/run_openai_adapter.py \\
        --base-url https://aisuite.cirrascale.com/apis/v2 \\
        --model Llama-3.1-8B
"""

from __future__ import annotations

import argparse

import uvicorn

from dragon_nest.transport.openai_adapter import (
    OpenAIAdapterConfig,
    OpenAIAdapterModel,
    create_openai_adapter_app,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the DragonNest OpenAI-compatible endpoint adapter"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--base-url",
        default="https://aisuite.cirrascale.com/apis/v2",
        help="OpenAI-compliant API base URL (no trailing /chat/completions)",
    )
    parser.add_argument(
        "--api-key-env",
        default="DRAGONNEST_OPENAI_API_KEY",
        help="Environment variable holding the provider's API key",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=[],
        help="Model id to advertise; repeatable. Defaults to Llama-3.1-8B",
    )
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--device-id", default="cirrascale-inference-cloud")
    parser.add_argument("--display-name", default="Cirrascale Inference Cloud")
    args = parser.parse_args()

    model_ids = args.models or ["Llama-3.1-8B"]
    config = OpenAIAdapterConfig(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        models=tuple(
            OpenAIAdapterModel(model_id=model_id, max_context_tokens=args.max_context_tokens)
            for model_id in model_ids
        ),
        device_id=args.device_id,
        display_name=args.display_name,
    )
    app = create_openai_adapter_app(config)
    print(f"OpenAI adapter listening on http://{args.host}:{args.port}")
    print(f"Forwarding to {args.base_url} using API key from ${args.api_key_env}")
    print("Register this adapter's address in the DragonNest admin dashboard as the endpoint.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

"""Model provider registry for OmniFinan."""

from __future__ import annotations


PROVIDER_REGISTRY = {
    "gpt": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "deepseek": "deepseek",
}


def infer_provider(model_name: str, default: str = "deepseek") -> str:
    lower = model_name.lower()
    for key, provider in PROVIDER_REGISTRY.items():
        if key in lower:
            return provider
    return default

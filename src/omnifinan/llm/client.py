"""Unified LLM call helpers used by OmniFinan agents."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from pydantic import BaseModel
from pyomnix.consts import OMNIX_PATH
from pyomnix.agents.models_settings import ModelConfig
from pyomnix.omnix_logger import get_logger

T = TypeVar("T", bound=BaseModel)
logger = get_logger(__name__)


def _cache_root() -> Path:
    root = OMNIX_PATH / "financial" / "llm_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_cache_key(
    prompt: Any,
    model_name: str,
    provider_api: str,
    schema_name: str | None,
    temperature: float | None,
    seed: int | None,
) -> str:
    payload = {
        "prompt": str(prompt),
        "model_name": model_name,
        "provider_api": provider_api,
        "schema_name": schema_name,
        "temperature": temperature,
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _cache_file(cache_key: str) -> Path:
    return _cache_root() / f"{cache_key}.json"


def _load_cached_response(cache_key: str) -> dict[str, Any] | None:
    path = _cache_file(cache_key)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _store_cached_response(cache_key: str, payload: dict[str, Any]) -> None:
    with open(_cache_file(cache_key), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _normalize_prompt(prompt: Any) -> Any:
    if isinstance(prompt, PromptValue):
        return prompt.to_messages()
    if isinstance(prompt, str):
        return [HumanMessage(content=prompt)]
    if isinstance(prompt, BaseMessage):
        return [prompt]
    if isinstance(prompt, list):
        if prompt and isinstance(prompt[0], dict):
            normalized_messages: list[BaseMessage] = []
            for item in prompt:
                role = item.get("role", "user")
                content = item.get("content", "")
                if role == "system":
                    normalized_messages.append(HumanMessage(content=f"[SYSTEM] {content}"))
                else:
                    normalized_messages.append(HumanMessage(content=content))
            return normalized_messages
        return prompt
    return [HumanMessage(content=str(prompt))]


def create_default_response(model_class: type[T]) -> T:
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        elif hasattr(field.annotation, "__args__"):
            default_values[field_name] = field.annotation.__args__[0]
        else:
            default_values[field_name] = None
    return model_class(**default_values)


def call_llm(
    prompt: Any,
    model_name: str,
    provider_api: str,
    pydantic_model: type[T] | None = None,
    max_retries: int = 3,
    default_factory=None,
    agent_name: str | None = None,
    trace=None,
    scratchpad=None,
    temperature: float | None = None,
    seed: int | None = None,
    deterministic_mode: bool = False,
) -> T | str:
    schema_name = pydantic_model.__name__ if pydantic_model is not None else None
    cache_key = _build_cache_key(
        prompt=prompt,
        model_name=model_name,
        provider_api=provider_api,
        schema_name=schema_name,
        temperature=temperature,
        seed=seed,
    )
    if deterministic_mode:
        cached = _load_cached_response(cache_key)
        if cached is not None:
            if pydantic_model is not None:
                try:
                    return pydantic_model.model_validate(cached["payload"])
                except Exception:
                    pass
            else:
                return str(cached.get("payload", ""))

    model_factory = ModelConfig().setup_model_factory(provider_api).get(provider_api)
    if model_factory is None:
        logger.error("Model factory not found for provider_api=%s", provider_api)
        if default_factory:
            return default_factory()
        if pydantic_model is None:
            return ""
        return create_default_response(pydantic_model)
    try:
        model_kwargs: dict[str, Any] = {"model": model_name, "max_retries": max_retries}
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if seed is not None:
            model_kwargs["seed"] = seed
        try:
            model = model_factory(**model_kwargs)
        except TypeError:
            # Backward-compatible fallback for providers that do not accept
            # temperature/seed arguments.
            model = model_factory(model=model_name, max_retries=max_retries)
        runnable = model.with_structured_output(pydantic_model) if pydantic_model else model
        normalized_prompt = _normalize_prompt(prompt)
    except Exception as exc:
        logger.error("Error creating LLM runnable for %s: %s", agent_name or "agent", exc)
        if default_factory:
            return default_factory()
        if pydantic_model is None:
            return ""
        return create_default_response(pydantic_model)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = runnable.invoke(normalized_prompt)
            prompt_text = str(prompt)
            if pydantic_model is not None:
                output_text = str(result)
            else:
                output_text = str(result.content)
            approx_input_tokens = max(len(prompt_text) // 4, 1)
            approx_output_tokens = max(len(output_text) // 4, 1)
            if trace is not None and hasattr(trace, "mark_llm_usage"):
                trace.mark_llm_usage(
                    agent_name=agent_name or "unknown_agent",
                    model_name=model_name,
                    provider_api=provider_api,
                    input_tokens=approx_input_tokens,
                    output_tokens=approx_output_tokens,
                )
            if scratchpad is not None and hasattr(scratchpad, "append"):
                scratchpad.append(
                    "llm_usage",
                    {
                        "agent_name": agent_name or "unknown_agent",
                        "model_name": model_name,
                        "provider_api": provider_api,
                        "input_tokens": approx_input_tokens,
                        "output_tokens": approx_output_tokens,
                    },
                )
            if pydantic_model is not None:
                if deterministic_mode:
                    _store_cached_response(
                        cache_key,
                        {"payload": result.model_dump(), "schema_name": schema_name},
                    )
                return result
            if deterministic_mode:
                _store_cached_response(
                    cache_key,
                    {"payload": result.content, "schema_name": None},
                )
            return result.content
        except Exception as exc:  # pragma: no cover - provider runtime
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(0.5 * (2**attempt))
    logger.error(
        "LLM call failed after retries for %s: %s",
        agent_name or "agent",
        last_error,
    )
    if default_factory:
        return default_factory()
    if pydantic_model is None:
        return ""
    return create_default_response(pydantic_model)


def extract_json_from_response(content: str) -> dict[str, Any] | None:
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as exc:
        logger.warning("Error extracting JSON from response: %s", exc)
    return None

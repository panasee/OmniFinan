"""Runtime configuration for OmniFinan."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data_models import MarketType


@dataclass
class LLMRuntimeConfig:
    model_name: str = "deepseek-chat"
    provider_api: str = "deepseek"
    temperature: float = 0.2
    max_retries: int = 3
    language: str = "Chinese"


@dataclass
class RuntimeConfig:
    enable_scratchpad: bool = True
    scratchpad_dirname: str = "scratchpad"
    enable_observability: bool = True
    data_cache_ttl_seconds: int = 3600
    data_provider: str = "akshare"
    market_type: MarketType = MarketType.CHINA
    enabled_analysts: list[str] = field(default_factory=list)
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    debate_rounds: int = 1
    deterministic_mode: bool = True
    llm_seed: int = 7
    llm: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        market_raw = os.getenv("OMNIFINAN_MARKET_TYPE", MarketType.CHINA.value)
        try:
            market_type = MarketType(market_raw)
        except Exception:
            market_type = MarketType.CHINA
        analysts_raw = os.getenv("OMNIFINAN_ENABLED_ANALYSTS", "")
        enabled_analysts = [item.strip() for item in analysts_raw.split(",") if item.strip()]
        return cls(
            enable_scratchpad=os.getenv("OMNIFINAN_ENABLE_SCRATCHPAD", "1") == "1",
            scratchpad_dirname=os.getenv("OMNIFINAN_SCRATCHPAD_DIR", "scratchpad"),
            enable_observability=os.getenv("OMNIFINAN_ENABLE_OBSERVABILITY", "1") == "1",
            data_cache_ttl_seconds=int(os.getenv("OMNIFINAN_DATA_CACHE_TTL", "3600")),
            data_provider=os.getenv("OMNIFINAN_DATA_PROVIDER", "akshare"),
            market_type=market_type,
            enabled_analysts=enabled_analysts,
            llm_provider=os.getenv("OMNIFINAN_LLM_PROVIDER", "deepseek"),
            llm_model=os.getenv("OMNIFINAN_LLM_MODEL", "deepseek-chat"),
            debate_rounds=int(os.getenv("OMNIFINAN_DEBATE_ROUNDS", "1")),
            deterministic_mode=os.getenv("OMNIFINAN_DETERMINISTIC_MODE", "1") == "1",
            llm_seed=int(os.getenv("OMNIFINAN_LLM_SEED", "7")),
            llm=LLMRuntimeConfig(
                model_name=os.getenv("OMNIFINAN_MODEL_NAME", "deepseek-chat"),
                provider_api=os.getenv("OMNIFINAN_PROVIDER_API", "deepseek"),
                temperature=float(os.getenv("OMNIFINAN_MODEL_TEMPERATURE", "0.2")),
                max_retries=int(os.getenv("OMNIFINAN_MODEL_RETRIES", "3")),
                language=os.getenv("OMNIFINAN_LANGUAGE", "Chinese"),
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeConfig":
        file_path = Path(path)
        with open(file_path, encoding="utf-8") as handle:
            if file_path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml  # type: ignore
                except Exception as exc:
                    raise RuntimeError(
                        "PyYAML is required for YAML config loading. Install with `pip install pyyaml`."
                    ) from exc
                payload = yaml.safe_load(handle) or {}
            else:
                import json

                payload = json.load(handle) or {}
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimeConfig":
        merged = cls.from_env()
        llm_payload = payload.get("llm", {})
        merged.enable_scratchpad = payload.get("enable_scratchpad", merged.enable_scratchpad)
        merged.scratchpad_dirname = payload.get("scratchpad_dirname", merged.scratchpad_dirname)
        merged.enable_observability = payload.get(
            "enable_observability", merged.enable_observability
        )
        merged.data_cache_ttl_seconds = payload.get(
            "data_cache_ttl_seconds", merged.data_cache_ttl_seconds
        )
        merged.data_provider = payload.get("data_provider", merged.data_provider)
        merged.market_type = MarketType(payload.get("market_type", merged.market_type.value))
        merged.enabled_analysts = payload.get("enabled_analysts", merged.enabled_analysts)
        merged.llm_provider = payload.get("llm_provider", merged.llm_provider)
        merged.llm_model = payload.get("llm_model", merged.llm_model)
        merged.debate_rounds = int(payload.get("debate_rounds", merged.debate_rounds))
        merged.deterministic_mode = payload.get("deterministic_mode", merged.deterministic_mode)
        merged.llm_seed = int(payload.get("llm_seed", merged.llm_seed))
        merged.llm = LLMRuntimeConfig(
            model_name=llm_payload.get("model_name", merged.llm.model_name),
            provider_api=llm_payload.get("provider_api", merged.llm.provider_api),
            temperature=float(llm_payload.get("temperature", merged.llm.temperature)),
            max_retries=int(llm_payload.get("max_retries", merged.llm.max_retries)),
            language=llm_payload.get("language", merged.llm.language),
        )
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_scratchpad": self.enable_scratchpad,
            "scratchpad_dirname": self.scratchpad_dirname,
            "enable_observability": self.enable_observability,
            "data_cache_ttl_seconds": self.data_cache_ttl_seconds,
            "data_provider": self.data_provider,
            "market_type": self.market_type.value,
            "enabled_analysts": self.enabled_analysts,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "debate_rounds": self.debate_rounds,
            "deterministic_mode": self.deterministic_mode,
            "llm_seed": self.llm_seed,
            "llm": {
                "model_name": self.llm.model_name,
                "provider_api": self.llm.provider_api,
                "temperature": self.llm.temperature,
                "max_retries": self.llm.max_retries,
                "language": self.llm.language,
            },
        }

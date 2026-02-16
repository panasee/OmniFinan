"""Credential loading helpers for market data providers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pyomnix.consts import OMNIX_PATH


def _load_json_loose(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    # Tolerate trailing commas in user-managed config files.
    normalized = re.sub(r",\s*([}\]])", r"\1", raw)
    parsed = json.loads(normalized)
    return parsed if isinstance(parsed, dict) else {}


def load_provider_credentials() -> dict[str, Any]:
    cfg_path = OMNIX_PATH / "finn_api.json"
    if not cfg_path.exists():
        return {}
    try:
        return _load_json_loose(cfg_path)
    except Exception:
        return {}


def get_api_key(provider: str) -> str | None:
    payload = load_provider_credentials()
    node: Any = payload.get(provider)
    if not isinstance(node, dict):
        target = provider.strip().lower()
        for key, value in payload.items():
            if isinstance(key, str) and key.strip().lower() == target and isinstance(value, dict):
                node = value
                break
    if not isinstance(node, dict):
        return None
    val = node.get("api_key")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None

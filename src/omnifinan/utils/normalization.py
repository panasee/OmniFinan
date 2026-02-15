"""Normalization helpers for consistent numeric conventions across agents."""

from __future__ import annotations

from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def confidence_to_unit(value: Any, default: float = 0.0) -> float:
    """Normalize confidence to [0, 1].

    Accepts legacy formats:
    - unit interval: 0.0 ~ 1.0
    - percent number: 0 ~ 100
    - percent string: "85%"
    """
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            numeric = float(text)
        except ValueError:
            return default
    elif isinstance(value, int | float):
        numeric = float(value)
    else:
        return default

    if numeric > 1.0:
        numeric = numeric / 100.0
    return clamp(numeric, 0.0, 1.0)


def confidence_to_percent(value: Any, default: float = 0.0) -> float:
    return confidence_to_unit(value, default=default) * 100.0

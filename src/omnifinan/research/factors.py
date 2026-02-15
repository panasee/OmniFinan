"""Lightweight factor DSL helpers inspired by qlib."""

from __future__ import annotations

import pandas as pd


def ref(series: pd.Series, n: int) -> pd.Series:
    return series.shift(n)


def mean(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=n).mean()


def std(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=n).std()


def rank(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=n).apply(lambda x: pd.Series(x).rank().iloc[-1])


def apply_factor(name: str, df: pd.DataFrame, column: str = "close") -> pd.Series:
    """Apply a small set of built-in factors by name."""
    s = df[column]
    if name == "Ref($close,1)":
        return ref(s, 1)
    if name == "Mean($close,5)":
        return mean(s, 5)
    if name == "Std($close,20)":
        return std(s, 20)
    if name == "Rank($close,20)":
        return rank(s, 20)
    raise ValueError(f"Unsupported factor expression: {name}")

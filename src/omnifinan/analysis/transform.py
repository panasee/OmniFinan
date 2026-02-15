"""Data transforms and lightweight factor features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_returns(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """Add daily and log returns to a price frame."""
    out = df.copy()
    out["return_1d"] = out[close_col].pct_change()
    out["log_return_1d"] = (1 + out["return_1d"]).apply(
        lambda x: pd.NA if pd.isna(x) or x <= 0 else np.log(x)
    )
    return out


def add_rolling_features(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """Add lightweight rolling features inspired by qlib-style factors."""
    out = df.copy()
    out["ma_5"] = out[close_col].rolling(window=5, min_periods=5).mean()
    out["ma_20"] = out[close_col].rolling(window=20, min_periods=20).mean()
    out["volatility_20"] = out[close_col].pct_change().rolling(window=20, min_periods=20).std()
    out["zscore_20"] = (
        (out[close_col] - out[close_col].rolling(window=20, min_periods=20).mean())
        / out[close_col].rolling(window=20, min_periods=20).std()
    )
    return out


def build_feature_frame(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """Build a standardized research frame for downstream backtesting/models."""
    frame = add_basic_returns(df, close_col=close_col)
    frame = add_rolling_features(frame, close_col=close_col)
    return frame

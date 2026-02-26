"""Factor extraction and quick IC evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _filter_by_date_range(
    frame: pd.DataFrame,
    *,
    date_col: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter rows by inclusive date range; no-op when both bounds are None."""
    if start_date is None and end_date is None:
        return frame
    if date_col not in frame.columns:
        raise KeyError(f"date_col '{date_col}' not found in DataFrame.")

    out = frame.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce")
    mask = pd.Series(True, index=out.index)
    if start_date is not None:
        mask &= dt >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= dt <= pd.Timestamp(end_date)
    return out.loc[mask].copy()


@dataclass(slots=True)
class CustomFactorSpec:
    """Declarative spec for user-defined factors.

    Notes:
    - `func` receives either a grouped sub-DataFrame or the whole DataFrame
      depending on `group_col`.
    - The callable can use arbitrary math libraries (numpy/scipy/statsmodels/pywt, etc.).
    - Return value must be index-aligned with the input chunk or have the same length.
    """

    name: str
    func: Callable[..., Any]
    group_col: str | None = "symbol"
    sort_col: str | None = "date"
    kwargs: dict[str, Any] = field(default_factory=dict)


def _coerce_factor_output(
    output: Any,
    *,
    index: pd.Index,
    factor_name: str,
) -> pd.Series:
    """Normalize custom factor output to an index-aligned Series."""
    if isinstance(output, pd.Series):
        if output.index.equals(index):
            return output
        if len(output) == len(index):
            return pd.Series(output.to_numpy(), index=index, name=factor_name)
        raise ValueError(
            f"Custom factor '{factor_name}' returned Series with incompatible shape/index."
        )
    if isinstance(output, (np.ndarray, list, tuple)):
        if len(output) != len(index):
            raise ValueError(
                f"Custom factor '{factor_name}' returned length={len(output)} "
                f"but expected {len(index)}."
            )
        return pd.Series(output, index=index, name=factor_name)
    raise TypeError(
        f"Custom factor '{factor_name}' returned unsupported type: {type(output)}."
    )


def apply_custom_factor(
    df: pd.DataFrame,
    *,
    name: str,
    func: Callable[..., Any],
    group_col: str | None = "symbol",
    sort_col: str | None = "date",
    kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply one custom factor callable and append result as a new column.

    This is the main extensibility interface for advanced math factors:
    users can pass any callable that maps a DataFrame (or grouped chunk)
    to an index-aligned vector.
    """
    out = df.copy()
    fn_kwargs = dict(kwargs or {})

    if sort_col is not None and sort_col in out.columns:
        out = out.sort_values(sort_col)

    if group_col is None:
        series = _coerce_factor_output(
            func(out, **fn_kwargs),
            index=out.index,
            factor_name=name,
        )
        out[name] = series
        return out

    if group_col not in out.columns:
        raise KeyError(f"group_col '{group_col}' not found in DataFrame.")

    result = pd.Series(index=out.index, dtype=float, name=name)
    for _, g in out.groupby(group_col, group_keys=False):
        local = g
        if sort_col is not None and sort_col in local.columns:
            local = local.sort_values(sort_col)
        series = _coerce_factor_output(
            func(local, **fn_kwargs),
            index=local.index,
            factor_name=name,
        )
        result.loc[local.index] = series.to_numpy()

    out[name] = result
    return out


def apply_custom_factors(
    df: pd.DataFrame,
    factors: Sequence[CustomFactorSpec] | Mapping[str, Callable[..., Any]],
) -> pd.DataFrame:
    """Apply multiple custom factors with concise syntax."""
    out = df.copy()
    if isinstance(factors, Mapping):
        for name, func in factors.items():
            out = apply_custom_factor(out, name=name, func=func)
        return out

    for spec in factors:
        out = apply_custom_factor(
            out,
            name=spec.name,
            func=spec.func,
            group_col=spec.group_col,
            sort_col=spec.sort_col,
            kwargs=spec.kwargs,
        )
    return out


def add_candidate_factors(
    df: pd.DataFrame,
    *,
    group_col: str = "symbol",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    forward_horizon: int = 5,
) -> pd.DataFrame:
    """Create a compact set of technical candidate factors."""
    out = df.copy()
    g = out.groupby(group_col, group_keys=False)

    close = out[close_col]
    high = out[high_col]
    low = out[low_col]
    volume = out[volume_col]

    out["ret_1"] = g[close_col].pct_change(1)
    out["ret_5"] = g[close_col].pct_change(5)
    out["ret_20"] = g[close_col].pct_change(20)

    ma_5 = g[close_col].transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma_20 = g[close_col].transform(lambda s: s.rolling(20, min_periods=20).mean())
    ma_60 = g[close_col].transform(lambda s: s.rolling(60, min_periods=60).mean())
    out["mom_ma_5_20"] = ma_5 / ma_20 - 1
    out["mom_ma_20_60"] = ma_20 / ma_60 - 1

    out["volatility_20"] = g[close_col].transform(
        lambda s: s.pct_change().rolling(20, min_periods=20).std()
    )
    out["amplitude_1"] = (high - low) / close.replace(0, np.nan)

    vma_20 = g[volume_col].transform(lambda s: s.rolling(20, min_periods=20).mean())
    out["vol_ratio_20"] = volume / vma_20
    out["rev_5"] = -out["ret_5"]

    out[f"fwd_ret_{forward_horizon}"] = (
        g[close_col].shift(-forward_horizon) / out[close_col] - 1
    )
    return out


def zscore_by_date(
    df: pd.DataFrame,
    factor_cols: Iterable[str],
    *,
    date_col: str = "date",
) -> tuple[pd.DataFrame, list[str]]:
    """Cross-sectional z-score by date for each factor."""
    out = df.copy()
    z_cols: list[str] = []
    for col in factor_cols:
        daily_mean = out.groupby(date_col)[col].transform("mean")
        daily_std = out.groupby(date_col)[col].transform("std")
        z_col = f"{col}_z"
        out[z_col] = (out[col] - daily_mean) / daily_std.replace(0, np.nan)
        z_cols.append(z_col)
    return out, z_cols


def daily_ic(
    frame: pd.DataFrame,
    factor_col: str,
    *,
    label_col: str = "fwd_ret_5",
    date_col: str = "date",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    method: str = "pearson",
    min_obs: int = 3,
) -> pd.Series:
    """Daily cross-sectional correlation between factor and future returns."""
    frame = _filter_by_date_range(
        frame,
        date_col=date_col,
        start_date=start_date,
        end_date=end_date,
    )

    def _corr(g: pd.DataFrame) -> float:
        x = g[factor_col]
        y = g[label_col]
        valid = x.notna() & y.notna()
        if valid.sum() < min_obs:
            return np.nan
        return x[valid].corr(y[valid], method=method)

    subset = frame[[date_col, factor_col, label_col]]
    return subset.groupby(date_col, group_keys=False)[[factor_col, label_col]].apply(_corr)


def evaluate_factors(
    frame: pd.DataFrame,
    factor_cols: Iterable[str],
    *,
    label_col: str = "fwd_ret_5",
    date_col: str = "date",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    min_obs: int = 3,
) -> pd.DataFrame:
    """Summarize IC and RankIC statistics for multiple factors."""
    frame = _filter_by_date_range(
        frame,
        date_col=date_col,
        start_date=start_date,
        end_date=end_date,
    )
    rows: list[dict[str, float | int | str]] = []
    for col in factor_cols:
        ic_s = daily_ic(
            frame,
            col,
            label_col=label_col,
            date_col=date_col,
            start_date=start_date,
            end_date=end_date,
            method="pearson",
            min_obs=min_obs,
        )
        rank_ic_s = daily_ic(
            frame,
            col,
            label_col=label_col,
            date_col=date_col,
            start_date=start_date,
            end_date=end_date,
            method="spearman",
            min_obs=min_obs,
        )
        ic_std = ic_s.std()
        rank_ic_std = rank_ic_s.std()
        rows.append(
            {
                "factor": col,
                "ic_mean": ic_s.mean(),
                "ic_std": ic_std,
                "ic_ir": ic_s.mean() / ic_std if ic_std and not np.isnan(ic_std) else np.nan,
                "rank_ic_mean": rank_ic_s.mean(),
                "rank_ic_std": rank_ic_std,
                "rank_ic_ir": (
                    rank_ic_s.mean() / rank_ic_std
                    if rank_ic_std and not np.isnan(rank_ic_std)
                    else np.nan
                ),
                "obs_days": int(ic_s.notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ic_mean", ascending=False).reset_index(drop=True)

"""Cross-sectional factor backtest helpers."""

from __future__ import annotations

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


def build_cross_sectional_weights(
    frame: pd.DataFrame,
    *,
    score_col: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
    ret_col: str = "ret_1",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    quantile: float = 0.2,
    min_universe: int = 8,
    long_short: bool = True,
) -> pd.DataFrame:
    """Build daily target weights from cross-sectional factor ranks."""
    frame = _filter_by_date_range(
        frame,
        date_col=date_col,
        start_date=start_date,
        end_date=end_date,
    )
    records: list[dict[str, object]] = []
    for dt, g in frame.groupby(date_col):
        tradable = g[[symbol_col, score_col, ret_col]].dropna()
        n = len(tradable)
        if n < min_universe:
            continue

        k = max(1, int(n * quantile))
        ranked = tradable.sort_values(score_col)
        long_names = ranked.tail(k)[symbol_col].tolist()

        if long_short:
            short_names = ranked.head(k)[symbol_col].tolist()
            records.extend(
                {date_col: dt, symbol_col: sym, "target_w": 1.0 / k}
                for sym in long_names
            )
            records.extend(
                {date_col: dt, symbol_col: sym, "target_w": -1.0 / k}
                for sym in short_names
            )
        else:
            records.extend(
                {date_col: dt, symbol_col: sym, "target_w": 1.0 / k}
                for sym in long_names
            )
    return pd.DataFrame(records)


def run_daily_backtest(
    frame: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    ret_col: str = "ret_1",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cost_rate: float = 0.001,
) -> pd.DataFrame:
    """Run a daily backtest with one-day signal delay and transaction costs."""
    frame = _filter_by_date_range(
        frame,
        date_col=date_col,
        start_date=start_date,
        end_date=end_date,
    )
    weights = _filter_by_date_range(
        weights,
        date_col=date_col,
        start_date=start_date,
        end_date=end_date,
    )
    if weights.empty:
        raise RuntimeError("weights is empty; no valid trading dates")

    ret_tbl = frame[[date_col, symbol_col, ret_col]].dropna()
    w_pivot = (
        weights.pivot(index=date_col, columns=symbol_col, values="target_w")
        .sort_index()
        .fillna(0.0)
    )
    exec_w = w_pivot.shift(1).fillna(0.0)

    ret_pivot = (
        ret_tbl.pivot(index=date_col, columns=symbol_col, values=ret_col)
        .sort_index()
        .fillna(0.0)
    )

    dates = exec_w.index.intersection(ret_pivot.index)
    exec_w = exec_w.reindex(dates).fillna(0.0)
    ret_pivot = ret_pivot.reindex(dates).fillna(0.0)

    gross_ret = (exec_w * ret_pivot).sum(axis=1)
    turnover = exec_w.diff().abs().sum(axis=1)
    if len(turnover) > 0:
        turnover.iloc[0] = exec_w.iloc[0].abs().sum()
    cost = turnover * cost_rate
    net_ret = gross_ret - cost

    equity = (1.0 + net_ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    bench_ret = ret_pivot.replace(0.0, np.nan).mean(axis=1).fillna(0.0)
    bench_equity = (1.0 + bench_ret).cumprod()

    return pd.DataFrame(
        {
            "gross_ret": gross_ret,
            "cost": cost,
            "net_ret": net_ret,
            "turnover": turnover,
            "equity": equity,
            "drawdown": drawdown,
            "bench_ret": bench_ret,
            "bench_equity": bench_equity,
        }
    )


def perf_stats(
    ret: pd.Series,
    equity: pd.Series,
    *,
    annualization: int = 252,
) -> dict[str, float]:
    """Compute standard daily strategy performance statistics."""
    ret = ret.dropna()
    if ret.empty:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }

    total_return = float(equity.iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (annualization / len(ret)) - 1.0)
    annual_vol = float(ret.std() * np.sqrt(annualization))
    sharpe = float(annual_return / annual_vol) if annual_vol > 0 else np.nan
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    win_rate = float((ret > 0).mean())
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }

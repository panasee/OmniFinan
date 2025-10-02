from __future__ import annotations

from datetime import date
from typing import Final

import pandas as pd

__all__ = ["compute_rangebreaks", "filter_trading_days"]

try:
    import pandas_market_calendars as mcal  # type: ignore
except ImportError:  # pragma: no cover - allow import without hard dependency at import time
    mcal = None  # type: ignore


_MARKET_TO_CALENDAR: Final[dict[str, str]] = {
    "US": "NYSE",  # US equities (NYSE/NASDAQ share trading days)
    "HK": "HKEX",  # Hong Kong Exchange
    "A": "SSE",  # China A-shares (use Shanghai as baseline)
    "NA": "NA",  # Not applicable (like crypto and other data)
}


def _resolve_calendar_code(market: str) -> str:
    market_upper = str(market).upper()
    if market_upper not in _MARKET_TO_CALENDAR:
        raise ValueError("market must be one of {'US', 'A', 'HK', 'NA'}")
    return _MARKET_TO_CALENDAR[market_upper]


def _compute_trading_dates(
    start_dt: pd.Timestamp, end_dt: pd.Timestamp, calendar_code: str
) -> set[date]:
    if calendar_code == "NA":
        # Crypto trades 24/7, so all dates are trading days
        # Return all dates in the range (inclusive)
        rng = pd.date_range(start=start_dt.date(), end=end_dt.date(), freq="D")
        return {ts.date() for ts in rng}
    if mcal is None:  # lazy runtime check
        raise ImportError(
            "pandas-market-calendars is required for trading day filtering. "
            "Please install it: pip install pandas-market-calendars"
        )
    calendar = mcal.get_calendar(calendar_code)
    schedule = calendar.schedule(start_date=start_dt.date(), end_date=end_dt.date())
    # schedule index is DatetimeIndex of trading sessions in exchange tz
    return {ts.date() for ts in schedule.index}


def filter_trading_days(data_df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Filter rows to trading days for the given market.

    The input must contain a 'time' column convertible to pandas datetime. Rows with
    non-parsable timestamps are dropped. All remaining rows whose date is not a
    trading session for the selected market will be removed.

    Args:
        data_df: DataFrame with at least a 'time' column.
        market: One of 'US', 'A', 'HK'.

    Returns:
        A new DataFrame filtered to trading days, sorted by 'time'.
    """
    if "time" not in data_df.columns:
        raise ValueError("data_df must contain a 'time' column")

    df = data_df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])  # drop invalid timestamps
    if df.empty:
        return df

    calendar_code = _resolve_calendar_code(market)
    trading_dates = _compute_trading_dates(df["time"].min(), df["time"].max(), calendar_code)
    mask = df["time"].dt.date.isin(trading_dates)
    return df.loc[mask].sort_values("time").reset_index(drop=True)


def compute_rangebreaks(start_dt: pd.Timestamp, end_dt: pd.Timestamp, market: str) -> list[dict]:
    """Build plotly rangebreaks to hide weekends and market holidays on date axis."""
    calendar_code = _resolve_calendar_code(market)
    if calendar_code == "NA":
        return []

    rbs: list[dict] = [dict(bounds=["sat", "mon"])]  # hide weekends
    if mcal is None:
        return rbs

    calendar = mcal.get_calendar(calendar_code)
    schedule = calendar.schedule(start_date=start_dt.date(), end_date=end_dt.date())

    all_days = pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq="D")
    trading_dates = {ts.date() for ts in schedule.index}
    all_days_dates = [ts.date() for ts in all_days]
    non_trading = [pd.Timestamp(d) for d in all_days_dates if d not in trading_dates]

    if non_trading:
        rbs.append(dict(values=non_trading))  # hide exchange holidays
    return rbs

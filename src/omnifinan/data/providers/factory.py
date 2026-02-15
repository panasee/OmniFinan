"""Provider factory."""

from __future__ import annotations

from .akshare_provider import AkshareProvider
from .base import DataProvider
from .finnhub_provider import FinnhubProvider
from .yfinance_provider import YFinanceProvider


def create_data_provider(name: str | None) -> DataProvider:
    provider_name = (name or "akshare").strip().lower()
    if provider_name == "akshare":
        return AkshareProvider()
    if provider_name == "finnhub":
        return FinnhubProvider()
    if provider_name in {"yfinance", "yf", "yahoo"}:
        return YFinanceProvider()
    raise ValueError(f"Unsupported data provider: {name}")


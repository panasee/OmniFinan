"""Akshare-backed provider implementation."""

from __future__ import annotations

import time

from ...data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from ...unified_api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_macro_indicators,
    get_market_cap,
    get_prices,
    search_line_items,
)
from .base import DataProvider


class AkshareProvider(DataProvider):
    def __init__(self, min_request_interval_seconds: float = 0.05):
        self.min_request_interval_seconds = min_request_interval_seconds
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.min_request_interval_seconds:
            time.sleep(self.min_request_interval_seconds - elapsed)
        self._last_request_ts = time.time()

    def get_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> list[Price]:
        self._throttle()
        return get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

    def get_financial_metrics(
        self, ticker: str, end_date: str | None = None, period: str = "ttm", limit: int = 1
    ) -> list[FinancialMetrics]:
        self._throttle()
        return get_financial_metrics(ticker=ticker, end_date=end_date, period=period, limit=limit)

    def search_line_items(self, ticker: str, period: str = "ttm", limit: int = 10) -> list[LineItem]:
        self._throttle()
        return search_line_items(ticker=ticker, period=period, limit=limit)

    def get_company_news(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ) -> list[CompanyNews]:
        self._throttle()
        return get_company_news(symbol=ticker, start_date=start_date, end_date=end_date, limit=limit)

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        self._throttle()
        return get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            start_date=start_date,
            limit=limit,
        )

    def get_market_cap(self, ticker: str, end_date: str | None = None) -> float | None:
        self._throttle()
        return get_market_cap(ticker=ticker, end_date=end_date)

    def get_macro_indicators(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> dict:
        self._throttle()
        return get_macro_indicators(start_date=start_date, end_date=end_date)

    def get_macro_indicators_subset(
        self,
        series_keys: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        self._throttle()
        return get_macro_indicators(start_date=start_date, end_date=end_date, include_series=series_keys)

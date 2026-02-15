"""Data provider interfaces for market/fundamental data."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price


class DataProvider(ABC):
    @abstractmethod
    def get_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> list[Price]:
        raise NotImplementedError

    @abstractmethod
    def get_financial_metrics(
        self, ticker: str, end_date: str | None = None, period: str = "ttm", limit: int = 1
    ) -> list[FinancialMetrics]:
        raise NotImplementedError

    @abstractmethod
    def search_line_items(self, ticker: str, period: str = "ttm", limit: int = 10) -> list[LineItem]:
        raise NotImplementedError

    @abstractmethod
    def get_company_news(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ) -> list[CompanyNews]:
        raise NotImplementedError

    @abstractmethod
    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        raise NotImplementedError

    @abstractmethod
    def get_market_cap(self, ticker: str, end_date: str | None = None) -> float | None:
        raise NotImplementedError

    @abstractmethod
    def get_macro_indicators(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> dict:
        raise NotImplementedError

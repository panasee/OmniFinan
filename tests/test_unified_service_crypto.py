from __future__ import annotations

from pathlib import Path

from omnifinan.data.cache import DataCache
from omnifinan.data.providers.base import DataProvider
from omnifinan.data.providers.yfinance_provider import YFinanceProvider
from omnifinan.data.unified_service import UnifiedDataService
from omnifinan.data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, MarketType, Price


class _DummyProvider(DataProvider):
    def __init__(self):
        self.price_calls = 0

    def get_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> list[Price]:
        self.price_calls += 1
        return [
            Price(
                open=10.0,
                high=11.0,
                low=9.0,
                close=10.5,
                volume=100,
                amount=1050.0,
                time="2025-01-01",
                market=MarketType.US,
            )
        ]

    def get_financial_metrics(
        self, ticker: str, end_date: str | None = None, period: str = "ttm", limit: int = 1
    ) -> list[FinancialMetrics]:
        return []

    def search_line_items(self, ticker: str, period: str = "ttm", limit: int = 10) -> list[LineItem]:
        return []

    def get_company_news(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ) -> list[CompanyNews]:
        return []

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        return []

    def get_market_cap(self, ticker: str, end_date: str | None = None) -> float | None:
        return None

    def get_macro_indicators(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> dict:
        return {}


def _cache(tmp_path: Path) -> DataCache:
    return DataCache(root=tmp_path / "cache", max_entries_per_namespace=20)


def test_unified_service_routes_crypto_prices_to_yfinance(monkeypatch, tmp_path: Path):
    provider = _DummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))

    yfinance_calls = {"count": 0}

    def fake_get_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> list[Price]:
        _ = self, start_date, end_date, interval
        yfinance_calls["count"] += 1
        return [
            Price(
                open=100000.0,
                high=101000.0,
                low=99000.0,
                close=100500.0,
                volume=42,
                amount=4221000.0,
                time="2025-01-01",
                market=None,
            )
        ]

    monkeypatch.setattr(YFinanceProvider, "get_prices", fake_get_prices)
    rows = service.get_prices("BTC-USD", "2025-01-01", "2025-01-01")

    assert len(rows) == 1
    assert rows[0]["close"] == 100500.0
    assert yfinance_calls["count"] == 1
    assert provider.price_calls == 0


def test_unified_service_non_crypto_uses_primary_provider(tmp_path: Path):
    provider = _DummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    rows = service.get_prices("AAPL", "2025-01-01", "2025-01-01")

    assert len(rows) == 1
    assert rows[0]["close"] == 10.5
    assert provider.price_calls == 1

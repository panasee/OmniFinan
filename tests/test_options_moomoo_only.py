from __future__ import annotations

from pathlib import Path

from omnifinan.data.cache import DataCache
from omnifinan.data.providers.base import DataProvider
from omnifinan.data.providers.moomoo_options_provider import MoomooOptionsProvider
from omnifinan.data.unified_service import UnifiedDataService
from omnifinan.data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from omnifinan.unified_api import get_futures_option_chain, get_stock_option_chain


class _DummyProvider(DataProvider):
    def get_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> list[Price]:
        return []

    def get_financial_metrics(
        self, ticker: str, end_date: str | None = None, period: str = "ttm", limit: int = 1
    ) -> list[FinancialMetrics]:
        return []

    def search_line_items(self, ticker: str, period: str = "ttm", limit: int = 10) -> list[LineItem]:
        return []

    def get_company_news_raw(
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

    def get_macro_indicators(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        return {}


def _cache(tmp_path: Path) -> DataCache:
    return DataCache(root=tmp_path / "cache", max_entries_per_namespace=20)


def test_unified_api_stock_options_accepts_moomoo_only(monkeypatch):
    def fake_chain(self, **kwargs):
        _ = self
        return {"meta": {"source": "moomoo"}, "data": [{"symbol": kwargs["symbol"]}], "raw": {}}

    monkeypatch.setattr(MoomooOptionsProvider, "get_stock_option_chain", fake_chain)

    out = get_stock_option_chain("AAPL", provider="auto")
    assert out["meta"]["source"] == "moomoo"

    try:
        get_stock_option_chain("AAPL", provider="marketdata")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "moomoo" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for removed marketdata provider")


def test_unified_service_futures_options_are_explicitly_unavailable(tmp_path: Path):
    svc = UnifiedDataService(provider=_DummyProvider(), cache=_cache(tmp_path))
    out = svc.get_futures_option_chain(symbol="ES", provider="auto", force=True)

    assert out["meta"]["source"] == "fixed_sources_unavailable"
    assert "not supported" in str(out["meta"]["error"]).lower()
    assert out["data"] == []


def test_unified_api_futures_options_are_explicitly_unavailable():
    out = get_futures_option_chain("ES", provider="auto")
    assert out["meta"]["source"] == "fixed_sources_unavailable"
    assert "not supported" in str(out["meta"]["error"]).lower()
    assert out["data"] == []

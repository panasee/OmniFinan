from __future__ import annotations

from pathlib import Path

from omnifinan.data.cache import DataCache
from omnifinan.data.providers.base import DataProvider
from omnifinan.data.providers.marketdata_provider import MarketDataOptionsProvider
from omnifinan.data.providers.yfinance_options_provider import YFinanceOptionsProvider
from omnifinan.data.unified_service import UnifiedDataService
from omnifinan.data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from omnifinan.unified_api import get_stock_option_chain


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


class _Resp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _cache(tmp_path: Path) -> DataCache:
    return DataCache(root=tmp_path / "cache", max_entries_per_namespace=20)


def test_marketdata_provider_parses_columnar_payload(monkeypatch):
    captured: dict = {}

    def fake_get(url, params=None, timeout=20):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _Resp(
            {
                "optionSymbol": ["AAPL260619C00200000", "AAPL260619P00200000"],
                "strike": [200.0, 200.0],
                "side": ["call", "put"],
                "bid": [1.23, 1.11],
            }
        )

    monkeypatch.setattr("omnifinan.data.providers.marketdata_provider.requests.get", fake_get)
    p = MarketDataOptionsProvider(api_key="demo-key", base_url="https://api.marketdata.app/v1", timeout=9)
    out = p.get_stock_option_chain(
        symbol="AAPL",
        expiration="2026-06-19",
        option_type="call",
        snapshot_mode="realtime",
    )

    assert captured["url"] == "https://api.marketdata.app/v1/options/chain/AAPL/"
    assert captured["params"]["token"] == "demo-key"
    assert captured["params"]["expiration"] == "2026-06-19"
    assert captured["params"]["type"] == "call"
    assert captured["timeout"] == 9
    assert out["meta"]["source"] == "marketdata"
    assert out["meta"]["snapshot_mode"] == "realtime"
    assert len(out["data"]) == 2
    assert out["data"][0]["optionSymbol"] == "AAPL260619C00200000"


def test_unified_service_option_chain_uses_cache(monkeypatch, tmp_path: Path):
    calls = {"n": 0}

    def fake_chain(self, **kwargs):
        _ = self
        calls["n"] += 1
        return {"meta": {"source": "marketdata"}, "data": [{"symbol": kwargs["symbol"]}]}

    monkeypatch.setattr(MarketDataOptionsProvider, "get_stock_option_chain", fake_chain)

    svc = UnifiedDataService(provider=_DummyProvider(), cache=_cache(tmp_path))
    one = svc.get_stock_option_chain("AAPL", expiration="2026-06-19")
    two = svc.get_stock_option_chain("AAPL", expiration="2026-06-19")

    assert calls["n"] == 1
    assert one["data"][0]["symbol"] == "AAPL"
    assert two["data"][0]["symbol"] == "AAPL"


def test_unified_api_options_provider_is_marketdata_only():
    try:
        get_stock_option_chain("AAPL", provider="marketdata")
    except RuntimeError:
        # API key missing is acceptable in this test.
        pass

    try:
        get_stock_option_chain("AAPL", provider="akshare")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "marketdata" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for non-marketdata provider")


def test_unified_api_options_auto_fallback_to_yfinance(monkeypatch):
    def fail_marketdata(self, **kwargs):
        _ = self, kwargs
        raise RuntimeError("quota exhausted")

    def fake_yf(self, **kwargs):
        _ = self
        return {"meta": {"source": "yfinance"}, "data": [{"symbol": kwargs["symbol"]}]}

    monkeypatch.setattr(MarketDataOptionsProvider, "get_stock_option_chain", fail_marketdata)
    monkeypatch.setattr(YFinanceOptionsProvider, "get_stock_option_chain", fake_yf)

    out = get_stock_option_chain("SPY", provider="auto")
    assert out["meta"]["source"] == "yfinance"
    assert out["data"][0]["symbol"] == "SPY"


def test_unified_service_option_chain_analytics_returns_skew(monkeypatch, tmp_path: Path):
    sample_chain = {
        "meta": {"source": "marketdata", "snapshot_date": "2026-03-03"},
        "data": [
            {
                "expiration": "2026-04-03",
                "dte": 30,
                "side": "put",
                "strike": 90.0,
                "iv": 0.24,
                "delta": -0.25,
                "openInterest": 5000,
            },
            {
                "expiration": "2026-04-03",
                "dte": 30,
                "side": "put",
                "strike": 100.0,
                "iv": 0.20,
                "delta": -0.50,
                "openInterest": 2000,
            },
            {
                "expiration": "2026-04-03",
                "dte": 30,
                "side": "call",
                "strike": 100.0,
                "iv": 0.20,
                "delta": 0.50,
                "openInterest": 1000,
            },
            {
                "expiration": "2026-04-03",
                "dte": 30,
                "side": "call",
                "strike": 110.0,
                "iv": 0.22,
                "delta": 0.25,
                "openInterest": 7000,
            },
        ],
        "raw": {},
    }

    def fake_get_chain(self, *args, **kwargs):
        _ = self, args, kwargs
        return sample_chain

    def fake_get_prices(self, ticker, start_date=None, end_date=None, interval="1d"):
        _ = self, ticker, start_date, end_date, interval
        prices = []
        px = 100.0
        for i in range(40):
            px = px * (1.0 + (0.002 if i % 2 == 0 else -0.001))
            prices.append({"time": f"2026-01-{(i % 28) + 1:02d}", "close": px})
        return prices

    monkeypatch.setattr(UnifiedDataService, "get_stock_option_chain", fake_get_chain)
    monkeypatch.setattr(UnifiedDataService, "get_prices", fake_get_prices)

    svc = UnifiedDataService(provider=_DummyProvider(), cache=_cache(tmp_path))
    out = svc.get_stock_option_chain_analytics(
        symbol="AAPL",
        underlying_price=100.0,
        iv_history=[0.12, 0.18, 0.20, 0.21, 0.24, 0.28],
    )

    assert out["meta"]["analytics_version"] == "options_analytics_v1"
    assert len(out["data"]) == 4
    skew = out["analytics"]["skew_by_expiry"][0]
    assert abs(skew["risk_reversal_25d"] - (0.22 - 0.24)) < 1e-9
    assert abs(skew["butterfly_25d"] - (0.5 * (0.22 + 0.24) - 0.20)) < 1e-9
    assert out["analytics"]["summary"]["iv_historical_percentile"] is not None
    assert out["analytics"]["implied_vs_realized"]["historical_volatility"] is not None
    assert out["analytics"]["max_pain"]["overall"]["max_pain_strike"] == 100.0
    assert out["analytics"]["levels"]["primary_support"]["strike"] == 90.0
    assert out["analytics"]["levels"]["primary_resistance"]["strike"] == 110.0


def test_unified_service_option_chain_for_a_hk_returns_unsupported(tmp_path: Path):
    svc = UnifiedDataService(provider=_DummyProvider(), cache=_cache(tmp_path))

    out_a = svc.get_stock_option_chain("600519")
    out_hk = svc.get_stock_option_chain("00700")
    ana_a = svc.get_stock_option_chain_analytics("600519")

    assert out_a["meta"]["source"] == "fixed_sources_unavailable"
    assert out_hk["meta"]["source"] == "fixed_sources_unavailable"
    assert "not supported" in out_a["meta"]["error"]
    assert ana_a["meta"]["source"] == "fixed_sources_unavailable"
    assert ana_a["analytics"]["summary"]["option_count"] == 0
    assert len(ana_a["analytics"]["errors"]) >= 1


def test_unified_service_crypto_pair_routes_to_base_for_stock_options(monkeypatch, tmp_path: Path):
    captured = {"symbol": None}

    def fake_yf(self, **kwargs):
        _ = self
        captured["symbol"] = kwargs["symbol"]
        return {"meta": {"source": "yfinance"}, "data": [{"symbol": kwargs["symbol"]}], "raw": {}}

    monkeypatch.setattr(YFinanceOptionsProvider, "get_stock_option_chain", fake_yf)

    svc = UnifiedDataService(provider=_DummyProvider(), cache=_cache(tmp_path))
    out = svc.get_stock_option_chain(symbol="BTC-USDT", provider="yfinance", force=True)

    assert captured["symbol"] == "BTC"
    assert out["data"][0]["symbol"] == "BTC"


def test_unified_service_crypto_pair_routes_to_base_for_futures_options(monkeypatch, tmp_path: Path):
    captured = {"symbol": None}

    def fake_md(self, **kwargs):
        _ = self
        captured["symbol"] = kwargs["symbol"]
        return {"meta": {"source": "marketdata"}, "data": [{"symbol": kwargs["symbol"]}], "raw": {}}

    monkeypatch.setattr(MarketDataOptionsProvider, "get_futures_option_chain", fake_md)

    svc = UnifiedDataService(provider=_DummyProvider(), cache=_cache(tmp_path))
    out = svc.get_futures_option_chain(symbol="ETH-USD", provider="marketdata", force=True)

    assert captured["symbol"] == "ETH"
    assert out["data"][0]["symbol"] == "ETH"

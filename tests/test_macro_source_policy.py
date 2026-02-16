from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path

from omnifinan.data.cache import DataCache
from omnifinan.data.providers.base import DataProvider
from omnifinan.data.unified_service import UnifiedDataService
from omnifinan.data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price


class _MacroDummyProvider(DataProvider):
    def __init__(self) -> None:
        self.calls = 0

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

    def get_macro_indicators(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        self.calls += 1
        today = date.today().strftime("%Y-%m-%d")
        return {
            "series": {
                "fed_policy_rate": {
                    "latest": {"date": today, "value": 5.0},
                    "observations": [{"date": today, "value": 5.0}],
                }
            },
            "latest": {"fed_policy_rate": 5.0},
            "snapshot_at": "2026-02-15T00:00:00Z",
            "source_policy": {"version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank"},
            "_args": {"start_date": start_date, "end_date": end_date},
        }


class _MacroSubsetDummyProvider(_MacroDummyProvider):
    def __init__(self) -> None:
        super().__init__()
        self.subset_calls = 0
        self.last_subset_keys: list[str] = []

    def get_macro_indicators_subset(
        self,
        series_keys: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        self.subset_calls += 1
        self.last_subset_keys = list(series_keys)
        series = {}
        for key in series_keys:
            series[key] = {
                "latest": {"date": "2026-02-15", "value": 1.0},
                "observations": [{"date": "2026-02-15", "value": 1.0}],
                "error": None,
                "source": "subset_refresh",
            }
        return {
            "series": series,
            "latest": {k: 1.0 for k in series.keys()},
            "snapshot_at": "2026-02-16T00:00:00Z",
        }


def _cache(tmp_path: Path) -> DataCache:
    return DataCache(root=tmp_path / "cache", max_entries_per_namespace=20)


def test_macro_source_policy_cache_and_history(tmp_path: Path):
    provider = _MacroDummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))

    one = service.get_macro_indicators(start_date="2025-01-01", end_date="2026-12-31")
    two = service.get_macro_indicators(start_date="2025-01-01", end_date="2026-12-31")

    # Second call is within 1 day and should be skipped.
    assert provider.calls == 1
    assert one["latest"]["fed_policy_rate"] == 5.0
    assert two["latest"]["fed_policy_rate"] == 5.0

    dataset_key = "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank__master"
    history = service.cache.get_dataset("macro_indicators_history", dataset_key)
    assert isinstance(history, list)
    assert len(history) == 1


def test_macro_refresh_when_series_latest_date_is_stale(tmp_path: Path):
    provider = _MacroDummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))

    params = {
        "scope": "master",
        "source_policy_version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank",
    }
    service.cache.set(
        "macro_indicators",
        params,
        {
            "series": {
                "fed_policy_rate": {
                    "latest": {"date": "2020-01-01", "value": 1.0},
                    "observations": [{"date": "2020-01-01", "value": 1.0}],
                    "source": "fred:FEDFUNDS",
                }
            },
            "latest": {"fed_policy_rate": 1.0},
            "snapshot_at": "2026-02-15T00:00:00Z",
        },
    )
    cache_file = service.cache._request_key_path("macro_indicators", params)
    stale_ts = (datetime.now() - timedelta(days=2)).timestamp()
    os.utime(cache_file, (stale_ts, stale_ts))

    _ = service.get_macro_indicators(start_date="2025-01-01", end_date="2025-12-31")
    assert provider.calls == 1


def test_macro_refresh_when_cache_has_errors_even_if_fresh(tmp_path: Path):
    provider = _MacroDummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    params = {
        "scope": "master",
        "source_policy_version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank",
    }
    service.cache.set(
        "macro_indicators",
        params,
        {
            "series": {
                "sofr": {
                    "latest": None,
                    "observations": [],
                    "error": "400 bad request",
                }
            },
            "latest": {"sofr": None},
            "snapshot_at": "2026-02-15T00:00:00Z",
        },
    )

    _ = service.get_macro_indicators(start_date="2025-01-01", end_date="2025-12-31")
    assert provider.calls == 1


def test_macro_window_is_subset_of_master(tmp_path: Path):
    provider = _MacroDummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))

    params = {
        "scope": "master",
        "source_policy_version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank",
    }
    service.cache.set(
        "macro_indicators",
        params,
        {
            "series": {
                "fed_policy_rate": {
                    "observations": [
                        {"date": "2025-01-01", "value": 5.0},
                        {"date": "2025-06-01", "value": 4.75},
                        {"date": "2025-12-01", "value": 4.5},
                    ],
                    "latest": {"date": "2025-12-01", "value": 4.5},
                }
            },
            "latest": {"fed_policy_rate": 4.5},
            "snapshot_at": "2026-02-15T00:00:00Z",
        },
    )
    out = service.get_macro_indicators(start_date="2025-05-01", end_date="2025-11-01")
    assert provider.calls == 0
    obs = out["series"]["fed_policy_rate"]["observations"]
    assert len(obs) == 1
    assert obs[0]["date"] == "2025-06-01"


def test_macro_restores_request_cache_from_dataset_master(tmp_path: Path):
    provider = _MacroDummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))

    today = date.today().strftime("%Y-%m-%d")
    dataset_key = "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank__master"
    snapshot = {
        "series": {
            "fed_policy_rate": {
                "latest": {"date": today, "value": 4.5},
                "observations": [{"date": today, "value": 4.5}],
            }
        },
        "latest": {"fed_policy_rate": 4.5},
        "snapshot_at": "2026-02-16T00:00:00Z",
    }
    service.cache.set_dataset("macro_indicators_history", dataset_key, [snapshot])

    out = service.get_macro_indicators(start_date="2025-01-01", end_date="2026-12-31")
    assert provider.calls == 0
    assert out["latest"]["fed_policy_rate"] == 4.5


def test_macro_refreshes_only_missing_series_when_subset_supported(tmp_path: Path):
    provider = _MacroSubsetDummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    params = {
        "scope": "master",
        "source_policy_version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank",
    }
    service.cache.set(
        "macro_indicators",
        params,
        {
            "series": {
                "sofr": {
                    "latest": None,
                    "observations": [],
                    "error": "temporary gateway timeout",
                    "source": "fred:SOFR",
                },
                "us_pmi_manufacturing": {
                    "latest": None,
                    "observations": [],
                    "error": "unavailable in current fixed providers (FRED/IMF/WB)",
                    "source": "fixed_sources_unavailable",
                },
            },
            "latest": {"sofr": None, "us_pmi_manufacturing": None},
            "snapshot_at": "2026-02-15T00:00:00Z",
        },
    )
    out = service.get_macro_indicators(start_date="2025-01-01", end_date="2026-12-31")
    assert provider.calls == 0
    assert provider.subset_calls == 1
    assert sorted(provider.last_subset_keys) == ["sofr", "us_pmi_manufacturing"]
    assert out["series"]["sofr"]["latest"]["value"] == 1.0

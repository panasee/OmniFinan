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


def test_macro_skips_refetch_when_recently_fetched(tmp_path: Path):
    """Series with old observation dates should NOT be re-fetched if fetched_at is recent."""
    provider = _MacroSubsetDummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
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
                    "latest": {"date": "2026-02-20", "value": 4.3},
                    "observations": [{"date": "2026-02-20", "value": 4.3}],
                    "source": "fred:SOFR",
                    "error": None,
                    "fetched_at": now_iso,
                },
                "us_treasury_10y": {
                    "latest": {"date": "2026-02-20", "value": 4.1},
                    "observations": [{"date": "2026-02-20", "value": 4.1}],
                    "source": "fred:DGS10",
                    "error": None,
                    "fetched_at": now_iso,
                },
            },
            "latest": {"sofr": 4.3, "us_treasury_10y": 4.1},
            "snapshot_at": "2026-02-20T00:00:00Z",
        },
    )
    _ = service.get_macro_indicators(start_date="2025-01-01", end_date="2026-12-31")
    assert provider.calls == 0
    assert provider.subset_calls == 0


def test_macro_refetches_when_fetched_at_is_old(tmp_path: Path):
    """Series with stale observation AND old fetched_at should be re-fetched."""
    provider = _MacroSubsetDummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    old_fetch = (datetime.now() - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S")
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
                    "latest": {"date": "2026-02-15", "value": 4.3},
                    "observations": [{"date": "2026-02-15", "value": 4.3}],
                    "source": "fred:SOFR",
                    "error": None,
                    "fetched_at": old_fetch,
                },
            },
            "latest": {"sofr": 4.3},
            "snapshot_at": "2026-02-20T00:00:00Z",
        },
    )
    _ = service.get_macro_indicators(start_date="2025-01-01", end_date="2026-12-31")
    assert provider.subset_calls == 1
    assert "sofr" in provider.last_subset_keys


def test_macro_error_series_with_recent_fetched_at_not_refetched(tmp_path: Path):
    """Series with an error but recent fetched_at should NOT be re-fetched."""
    provider = _MacroSubsetDummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
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
                    "fetched_at": now_iso,
                },
                "fed_policy_rate": {
                    "latest": {"date": date.today().strftime("%Y-%m-%d"), "value": 5.0},
                    "observations": [{"date": date.today().strftime("%Y-%m-%d"), "value": 5.0}],
                    "source": "fred:FEDFUNDS",
                    "error": None,
                    "fetched_at": now_iso,
                },
            },
            "latest": {"sofr": None, "fed_policy_rate": 5.0},
            "snapshot_at": "2026-02-20T00:00:00Z",
        },
    )
    _ = service.get_macro_indicators(start_date="2025-01-01", end_date="2026-12-31")
    assert provider.calls == 0
    assert provider.subset_calls == 0


# --- Providers that simulate akshare / network errors ---


class _MacroRaisingProvider(_MacroDummyProvider):
    """Simulates akshare or network error: get_macro_indicators always raises."""

    def get_macro_indicators(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        self.calls += 1
        raise RuntimeError("akshare interface timeout / 数据源暂时不可用")


class _MacroSubsetRaisingProvider(_MacroSubsetDummyProvider):
    """Subset refresh raises (e.g. akshare error); full fetch also raises."""

    def get_macro_indicators_subset(
        self,
        series_keys: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        self.subset_calls += 1
        self.last_subset_keys = list(series_keys)
        raise RuntimeError("akshare 接口异常")

    def get_macro_indicators(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        self.calls += 1
        raise RuntimeError("akshare 接口异常")


class _MacroPartialErrorProvider(_MacroDummyProvider):
    """Full fetch returns some series with data, some with error (e.g. akshare 部分接口失败)."""

    def get_macro_indicators(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        self.calls += 1
        date_in_window = "2026-02-15"
        return {
            "series": {
                "fed_policy_rate": {
                    "latest": {"date": date_in_window, "value": 5.25},
                    "observations": [{"date": date_in_window, "value": 5.25}],
                    "source": "fred:FEDFUNDS",
                    "error": None,
                },
                "china_lpr_1y": {
                    "latest": None,
                    "observations": [],
                    "source": "akshare:china_official:pboc",
                    "error": "akshare 接口超时",
                },
            },
            "latest": {"fed_policy_rate": 5.25, "china_lpr_1y": None},
            "snapshot_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_policy": {"version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank"},
        }


def test_macro_when_provider_raises_returns_existing_cache(tmp_path: Path):
    """当 provider 全量拉取抛异常（如 akshare 报错）时，应返回已有缓存，不覆盖、不重复拉取已有数据。"""
    provider = _MacroRaisingProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    params = {
        "scope": "master",
        "source_policy_version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank",
    }
    existing = {
        "series": {
            "fed_policy_rate": {
                "latest": {"date": "2026-02-01", "value": 4.5},
                "observations": [{"date": "2026-02-01", "value": 4.5}],
                "source": "fred:FEDFUNDS",
                "error": None,
                "fetched_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "sofr": {
                "latest": {"date": "2026-02-01", "value": 4.3},
                "observations": [{"date": "2026-02-01", "value": 4.3}],
                "source": "fred:SOFR",
                "error": None,
                "fetched_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            },
        },
        "latest": {"fed_policy_rate": 4.5, "sofr": 4.3},
        "snapshot_at": "2026-02-01T00:00:00Z",
    }
    service.cache.set("macro_indicators", params, existing)

    out = service.get_macro_indicators(start_date="2026-01-01", end_date="2026-02-28")
    assert provider.calls == 0
    assert out["latest"]["fed_policy_rate"] == 4.5
    assert out["latest"]["sofr"] == 4.3
    assert out["series"]["fed_policy_rate"]["observations"]
    assert out["series"]["sofr"]["observations"]


def test_macro_when_subset_and_full_both_raise_returns_existing(tmp_path: Path):
    """Subset 拉取异常后 fallback 全量，全量也异常时，仍返回已有缓存，不丢失本地数据。"""
    provider = _MacroSubsetRaisingProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    params = {
        "scope": "master",
        "source_policy_version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank",
    }
    existing = {
        "series": {
            "china_cpi_yoy": {
                "latest": {"date": "2025-06-15", "value": 0.8},
                "observations": [{"date": "2025-06-15", "value": 0.8}],
                "source": "akshare:china_official:nbs",
                "error": None,
                "fetched_at": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S"),
            },
        },
        "latest": {"china_cpi_yoy": 0.8},
        "snapshot_at": "2025-06-15T00:00:00Z",
    }
    service.cache.set("macro_indicators", params, existing)

    out = service.get_macro_indicators(start_date="2025-01-01", end_date="2026-02-28")
    assert out["latest"]["china_cpi_yoy"] == 0.8
    assert out["series"]["china_cpi_yoy"]["observations"]
    assert provider.subset_calls == 1
    assert provider.calls == 1


def test_macro_merge_preserves_existing_when_incoming_has_error_for_series(tmp_path: Path):
    """Merge 时若某 series 本次拉取为 error/空，应保留本地已有该 series 的数据，不重复拉取覆盖。"""
    provider = _MacroPartialErrorProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    params = {
        "scope": "master",
        "source_policy_version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank",
    }
    existing = {
        "series": {
            "fed_policy_rate": {
                "latest": {"date": "2026-01-15", "value": 4.5},
                "observations": [{"date": "2026-01-15", "value": 4.5}],
                "source": "fred:FEDFUNDS",
                "error": None,
            },
            "china_lpr_1y": {
                "latest": {"date": "2026-01-20", "value": 3.45},
                "observations": [{"date": "2026-01-20", "value": 3.45}],
                "source": "akshare:china_official:pboc",
                "error": None,
            },
        },
        "latest": {"fed_policy_rate": 4.5, "china_lpr_1y": 3.45},
        "snapshot_at": "2026-01-15T00:00:00Z",
    }
    service.cache.set("macro_indicators", params, existing)
    old_fetched = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S")
    for key in ("fed_policy_rate", "china_lpr_1y"):
        existing["series"][key]["fetched_at"] = old_fetched
    service.cache.set("macro_indicators", params, existing)

    out = service.get_macro_indicators(start_date="2026-01-01", end_date="2026-02-28")
    merged = out["series"]
    assert merged["fed_policy_rate"]["latest"]["value"] == 5.25
    assert merged["fed_policy_rate"]["observations"]
    assert merged["china_lpr_1y"]["latest"]["value"] == 3.45
    assert len(merged["china_lpr_1y"]["observations"]) == 1
    assert merged["china_lpr_1y"]["observations"][0]["date"] == "2026-01-20"


def test_macro_no_full_fetch_when_cache_fresh_and_complete(tmp_path: Path):
    """本地缓存齐全且未过期时，不发起全量拉取，保证已有数据不重复拉取。"""
    provider = _MacroDummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    today = date.today().strftime("%Y-%m-%d")
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
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
                    "latest": {"date": today, "value": 5.0},
                    "observations": [{"date": today, "value": 5.0}],
                    "source": "fred:FEDFUNDS",
                    "error": None,
                    "fetched_at": now_iso,
                },
                "sofr": {
                    "latest": {"date": today, "value": 4.3},
                    "observations": [{"date": today, "value": 4.3}],
                    "source": "fred:SOFR",
                    "error": None,
                    "fetched_at": now_iso,
                },
            },
            "latest": {"fed_policy_rate": 5.0, "sofr": 4.3},
            "snapshot_at": f"{today}T00:00:00Z",
        },
    )

    out = service.get_macro_indicators(start_date="2025-01-01", end_date="2026-12-31")
    assert provider.calls == 0
    assert out["latest"]["fed_policy_rate"] == 5.0
    assert out["latest"]["sofr"] == 4.3

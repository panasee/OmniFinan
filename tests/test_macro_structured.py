from __future__ import annotations

from pathlib import Path

from omnifinan.data.cache import DataCache
from omnifinan.data.providers.base import DataProvider
from omnifinan.data.unified_service import UnifiedDataService
from omnifinan.data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from omnifinan.unified_api import structure_macro_indicators


def _obs_monthly(values: list[float], *, start_year: int = 2024, start_month: int = 1):
    out = []
    y = start_year
    m = start_month
    for v in values:
        out.append({"date": f"{y:04d}-{m:02d}-01", "value": float(v)})
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def test_structure_macro_indicators_builds_llm_and_chart_friendly_payload():
    payload = {
        "snapshot_at": "2026-02-16T00:00:00Z",
        "source_policy": {"version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank"},
        "series": {
            "us_cpi_yoy": {
                "source": "fred:CPIAUCSL",
                "observations": _obs_monthly([float(i) for i in range(1, 15)]),  # 14 points
                "latest": {"date": "2025-02-01", "value": 14.0},
                "previous": {"date": "2025-01-01", "value": 13.0},
                "trend": "up",
                "error": None,
            },
            "sg_policy_rate": {
                "source": "world_bank:SGP:FR.INR.DPST",
                "observations": _obs_monthly([1.0, 1.0, 0.9, 0.8]),
                "latest": {"date": "2024-04-01", "value": 0.8},
                "previous": {"date": "2024-03-01", "value": 0.9},
                "trend": "down",
                "error": None,
            },
            "us_pmi_services": {
                "source": "fixed_sources_unavailable",
                "observations": [],
                "latest": None,
                "previous": None,
                "trend": "flat",
                "error": "unavailable in current fixed providers (FRED/IMF/WB)",
            },
        },
    }

    out = structure_macro_indicators(payload)
    assert out["meta"]["coverage"]["total_metrics"] == 3
    assert out["meta"]["coverage"]["ok_metrics"] == 2
    assert out["meta"]["coverage"]["error_metrics"] == 1

    cpi = out["metrics"]["us_cpi_yoy"]
    assert cpi["dimension"] == "inflation"
    assert cpi["country"] == "US"
    assert cpi["frequency"] == "monthly"
    assert cpi["mom"] is not None
    assert cpi["yoy"] is not None
    assert cpi["trend_short"] == "up"
    assert cpi["obs_count"] == 14

    pmi = out["metrics"]["us_pmi_services"]
    assert pmi["error"] is not None
    assert pmi["obs_count"] == 0

    assert len(out["chart_data"]["long"]) == 18
    assert any(r["key"] == "sg_policy_rate" for r in out["chart_data"]["long"])


class _MacroStructuredDummyProvider(DataProvider):
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
        return {
            "snapshot_at": "2026-02-16T00:00:00Z",
            "source_policy": {"version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank"},
            "series": {
                "us_m2": {
                    "source": "fred:M2SL",
                    "observations": _obs_monthly([100, 101, 102, 103, 104, 105]),
                    "latest": {"date": "2024-06-01", "value": 105.0},
                    "previous": {"date": "2024-05-01", "value": 104.0},
                    "trend": "up",
                    "error": None,
                }
            },
            "latest": {"us_m2": 105.0},
        }


def test_unified_service_macro_structured_view(tmp_path: Path):
    cache = DataCache(root=tmp_path / "cache", max_entries_per_namespace=20)
    svc = UnifiedDataService(provider=_MacroStructuredDummyProvider(), cache=cache)
    out = svc.get_macro_indicators_structured(start_date="2024-01-01", end_date="2024-12-31")
    assert isinstance(out, dict)
    assert out["meta"]["coverage"]["total_metrics"] >= 1
    assert "us_m2" in out["metrics"]

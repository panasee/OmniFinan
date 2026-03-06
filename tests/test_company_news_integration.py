from __future__ import annotations

from pathlib import Path

from omnifinan.data.cache import DataCache
from omnifinan.data.providers.base import DataProvider
from omnifinan.data.unified_service import UnifiedDataService
from omnifinan.data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price


class _DummyProvider(DataProvider):
    def __init__(self):
        self.news_calls = 0

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
        self.news_calls += 1
        return [
            CompanyNews(
                ticker=ticker,
                title="贵州茅台发布年度经营公告",
                source="上交所公告",
                date="2026-03-05 08:30:00",
                url="https://example.com/notice",
                publish_time="2026-03-05 08:30:00",
                content="公司披露年度经营数据。",
            )
        ]

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


def test_unified_service_integrates_search_news(monkeypatch, tmp_path: Path):
    provider = _DummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))

    monkeypatch.setattr(
        "omnifinan.data.unified_service.fetch_search_news",
        lambda **kwargs: [
            {
                "ticker": "AAPL",
                "market": "us",
                "title": "Apple launches new AI features for iPhone",
                "content": "Reuters reported Apple unveiled new AI features.",
                "source": "Reuters",
                "source_type": "wire",
                "source_weight": 0.8,
                "date": "2026-03-05 09:00:00",
                "publish_time": "2026-03-05 09:00:00",
                "published_at": "2026-03-05 09:00:00",
                "url": "https://www.reuters.com/apple-ai",
                "domain": "reuters.com",
                "search_provider": "tavily",
            },
            {
                "ticker": "AAPL",
                "market": "us",
                "title": "Apple unveils new AI features for iPhone lineup",
                "content": "Bloomberg also covered the launch event.",
                "source": "Bloomberg",
                "source_type": "wire",
                "source_weight": 0.8,
                "date": "2026-03-05 09:20:00",
                "publish_time": "2026-03-05 09:20:00",
                "published_at": "2026-03-05 09:20:00",
                "url": "https://www.bloomberg.com/apple-ai",
                "domain": "bloomberg.com",
                "search_provider": "brave",
            },
            {
                "ticker": "AAPL",
                "market": "us",
                "title": "Apple launches new AI features for iPhone",
                "content": "Social media discussion followed the announcement.",
                "source": "social media",
                "source_type": "social_media",
                "source_weight": 0.1,
                "date": "2026-03-05 10:00:00",
                "publish_time": "2026-03-05 10:00:00",
                "published_at": "2026-03-05 10:00:00",
                "url": "https://x.com/post/apple-ai",
                "domain": "x.com",
                "search_provider": "tavily",
            },
        ],
    )

    rows = service.get_company_news("AAPL", start_date="2026-03-01", end_date="2026-03-06", limit=5)

    assert provider.news_calls == 0
    assert len(rows) == 1
    assert rows[0]["consensus_passed"] is True
    assert rows[0]["source_count"] == 3
    assert rows[0]["high_weight_source_count"] == 2
    assert rows[0]["weighted_source_score"] >= 1.6
    assert len(rows[0]["sources"]) == 3


def test_unified_service_uses_provider_for_a_shares(monkeypatch, tmp_path: Path):
    provider = _DummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))

    monkeypatch.setattr(
        "omnifinan.data.unified_service.fetch_search_news",
        lambda **kwargs: [],
    )

    rows = service.get_company_news("600519", start_date="2026-03-01", end_date="2026-03-06", limit=5)

    assert provider.news_calls == 1
    assert len(rows) == 1
    assert rows[0]["title"] == "贵州茅台发布年度经营公告"
    assert rows[0]["primary_source"] == "上交所公告"
    assert rows[0]["official_confirmed"] is True


def test_unified_service_news_uses_integrated_cache(monkeypatch, tmp_path: Path):
    provider = _DummyProvider()
    service = UnifiedDataService(provider=provider, cache=_cache(tmp_path))
    calls = {"n": 0}

    def fake_fetch(**kwargs):
        _ = kwargs
        calls["n"] += 1
        return [
            {
                "ticker": "AAPL",
                "market": "us",
                "title": "Apple signs new enterprise AI deal",
                "content": "Reuters said Apple signed a new enterprise AI deal.",
                "source": "Reuters",
                "source_type": "wire",
                "source_weight": 0.8,
                "date": "2026-03-05 09:00:00",
                "publish_time": "2026-03-05 09:00:00",
                "published_at": "2026-03-05 09:00:00",
                "url": "https://www.reuters.com/apple-enterprise-ai",
                "domain": "reuters.com",
                "search_provider": "tavily",
            }
        ]

    monkeypatch.setattr("omnifinan.data.unified_service.fetch_search_news", fake_fetch)

    one = service.get_company_news("AAPL", start_date="2026-03-01", end_date="2026-03-06", limit=5)
    two = service.get_company_news("AAPL", start_date="2026-03-01", end_date="2026-03-06", limit=5)

    assert calls["n"] == 1
    assert len(one) == 1
    assert len(two) == 1
    assert one[0]["event_id"] == two[0]["event_id"]

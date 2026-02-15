"""Cached data service wrapping a concrete provider."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from .cache import DataCache
from .providers.base import DataProvider
from .providers.yfinance_provider import YFinanceProvider
from .symbols import is_crypto_ticker


class UnifiedDataService:
    def __init__(self, provider: DataProvider, cache: DataCache | None = None, ttl_seconds: int = 3600):
        self.provider = provider
        self.cache = cache or DataCache()
        self.ttl_seconds = ttl_seconds
        self._crypto_provider: DataProvider | None = None
        self.cache.cleanup_expired(ttl_seconds)

    def _get_crypto_provider(self) -> DataProvider:
        if self._crypto_provider is None:
            self._crypto_provider = YFinanceProvider()
        return self._crypto_provider

    def _ticker_key(self, ticker: str) -> str:
        return ticker.upper().strip()

    def _model_dump_list(self, items: list[Any]) -> list[dict[str, Any]]:
        dumped: list[dict[str, Any]] = []
        for item in items:
            if hasattr(item, "model_dump"):
                dumped.append(item.model_dump())
            elif isinstance(item, dict):
                dumped.append(item)
        return dumped

    def _parse_date(self, value: Any) -> date | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            head = value.strip()[:10]
            for fmt in ("%Y-%m-%d", "%Y%m%d"):
                try:
                    return datetime.strptime(head, fmt).date()
                except ValueError:
                    continue
        return None

    def _date_to_str(self, d: date) -> str:
        return d.strftime("%Y-%m-%d")

    def _merge_records(self, existing: list[dict[str, Any]], incoming: list[dict[str, Any]], key_field: str):
        merged: dict[str, dict[str, Any]] = {}
        for row in existing:
            key = row.get(key_field)
            if key is not None:
                merged[str(key)] = row
        for row in incoming:
            key = row.get(key_field)
            if key is not None:
                merged[str(key)] = row
        return list(merged.values())

    def _merge_news_records(
        self,
        existing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        def _news_key(item: dict[str, Any]) -> str | None:
            url = item.get("url")
            if url:
                return f"url::{url}"
            title = item.get("title", "")
            dt = item.get("date", "")
            if title or dt:
                return f"title_date::{title}::{dt}"
            return None

        for row in existing:
            key = _news_key(row)
            if key:
                merged[key] = row
        for row in incoming:
            key = _news_key(row)
            if key:
                merged[key] = row
        return list(merged.values())

    def _merge_insider_records(
        self,
        existing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        def _trade_key(item: dict[str, Any]) -> str:
            return "|".join(
                str(item.get(k, ""))
                for k in ("filing_date", "transaction_date", "insider_name", "shares_traded")
            )

        for row in existing:
            merged[_trade_key(row)] = row
        for row in incoming:
            merged[_trade_key(row)] = row
        return list(merged.values())

    def _sort_by_date(self, items: list[dict[str, Any]], date_field: str, descending: bool = False):
        return sorted(
            items,
            key=lambda x: self._parse_date(x.get(date_field)) or date.min,
            reverse=descending,
        )

    def _filter_by_date_range(
        self,
        items: list[dict[str, Any]],
        date_field: str,
        start_date: str | None,
        end_date: str | None,
    ) -> list[dict[str, Any]]:
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        out: list[dict[str, Any]] = []
        for item in items:
            d = self._parse_date(item.get(date_field))
            if d is None:
                continue
            if start and d < start:
                continue
            if end and d > end:
                continue
            out.append(item)
        return out

    def _today_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _cached_call(
        self,
        namespace: str,
        params: dict[str, Any],
        call_fn,
        *,
        cacheable: bool = True,
    ) -> Any:
        if cacheable:
            hit = self.cache.get(namespace, params, ttl_seconds=self.ttl_seconds)
            if hit is not None:
                return hit
        result = call_fn()
        if cacheable:
            self.cache.set(namespace, params, result)
        return result

    def get_prices(
        self,
        ticker: str,
        start_date: str | None,
        end_date: str | None,
        interval: str = "1d",
    ):
        dataset_key = f"{self._ticker_key(ticker)}__{interval}"
        stored = self.cache.get_dataset("prices", dataset_key) or []
        stored = self._sort_by_date(stored, "time")
        price_provider = self._get_crypto_provider() if is_crypto_ticker(ticker) else self.provider

        target_start = self._parse_date(start_date)
        target_end = self._parse_date(end_date or self._today_str())

        fetch_chunks: list[list[dict[str, Any]]] = []
        if not stored:
            initial = price_provider.get_prices(ticker, start_date, end_date, interval=interval)
            fetch_chunks.append(self._model_dump_list(initial))
        else:
            first_date = self._parse_date(stored[0].get("time"))
            last_date = self._parse_date(stored[-1].get("time"))

            if target_start and first_date and target_start < first_date:
                backfill_end = first_date - timedelta(days=1)
                if target_start <= backfill_end:
                    backfill = price_provider.get_prices(
                        ticker=ticker,
                        start_date=self._date_to_str(target_start),
                        end_date=self._date_to_str(backfill_end),
                        interval=interval,
                    )
                    fetch_chunks.append(self._model_dump_list(backfill))

            if target_end and last_date and target_end > last_date:
                append_start = last_date + timedelta(days=1)
                if append_start <= target_end:
                    append_rows = price_provider.get_prices(
                        ticker=ticker,
                        start_date=self._date_to_str(append_start),
                        end_date=self._date_to_str(target_end),
                        interval=interval,
                    )
                    fetch_chunks.append(self._model_dump_list(append_rows))

        for rows in fetch_chunks:
            if rows:
                stored = self._merge_records(stored, rows, key_field="time")
        stored = self._sort_by_date(stored, "time")
        if stored:
            self.cache.set_dataset("prices", dataset_key, stored)
        return self._filter_by_date_range(stored, "time", start_date, end_date)

    def get_financial_metrics(self, ticker: str, end_date: str | None, period: str = "ttm", limit: int = 1):
        dataset_key = f"{self._ticker_key(ticker)}__{period}"
        stored = self.cache.get_dataset("financial_metrics", dataset_key) or []
        stored = self._sort_by_date(stored, "report_period", descending=True)

        eligible = self._filter_by_date_range(stored, "report_period", None, end_date)
        if len(eligible) < limit:
            fetch_limit = max(limit, len(stored) + max(limit, 5))
            fetched = self._model_dump_list(
                self.provider.get_financial_metrics(
                    ticker=ticker,
                    end_date=end_date,
                    period=period,
                    limit=fetch_limit,
                )
            )
            if fetched:
                stored = self._merge_records(stored, fetched, key_field="report_period")
                stored = self._sort_by_date(stored, "report_period", descending=True)
                self.cache.set_dataset("financial_metrics", dataset_key, stored)
                eligible = self._filter_by_date_range(stored, "report_period", None, end_date)

        return eligible[:limit]

    def get_line_items(self, ticker: str, period: str = "ttm", limit: int = 10):
        dataset_key = f"{self._ticker_key(ticker)}__{period}"
        stored = self.cache.get_dataset("line_items", dataset_key) or []
        stored = self._sort_by_date(stored, "report_period", descending=True)
        if len(stored) < limit:
            fetch_limit = max(limit, len(stored) + max(limit, 5))
            fetched = self._model_dump_list(
                self.provider.search_line_items(
                    ticker=ticker,
                    period=period,
                    limit=fetch_limit,
                )
            )
            if fetched:
                stored = self._merge_records(stored, fetched, key_field="report_period")
                stored = self._sort_by_date(stored, "report_period", descending=True)
                self.cache.set_dataset("line_items", dataset_key, stored)
        return stored[:limit]

    def get_market_cap(self, ticker: str, end_date: str | None):
        params = {"ticker": ticker, "end_date": end_date}
        return self._cached_call(
            "market_cap",
            params,
            lambda: self.provider.get_market_cap(ticker=ticker, end_date=end_date),
        )

    def get_macro_indicators(self, start_date: str | None = None, end_date: str | None = None):
        params = {"start_date": start_date, "end_date": end_date}
        return self._cached_call(
            "macro_indicators",
            params,
            lambda: self.provider.get_macro_indicators(start_date=start_date, end_date=end_date),
        )

    def get_company_news(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ):
        dataset_key = self._ticker_key(ticker)
        stored = self.cache.get_dataset("company_news", dataset_key) or []
        stored = self._sort_by_date(stored, "date", descending=True)

        target_end = self._parse_date(end_date or self._today_str())
        if not stored:
            fetched = self._model_dump_list(
                self.provider.get_company_news(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    limit=max(limit, 30),
                )
            )
            if fetched:
                stored = self._merge_news_records(stored, fetched)
        else:
            newest = self._parse_date(stored[0].get("date"))
            oldest = self._parse_date(stored[-1].get("date"))

            if newest and target_end and newest < target_end:
                update_start = newest + timedelta(days=1)
                if update_start <= target_end:
                    updates = self._model_dump_list(
                        self.provider.get_company_news(
                            ticker=ticker,
                            start_date=self._date_to_str(update_start),
                            end_date=self._date_to_str(target_end),
                            limit=max(limit * 3, 30),
                        )
                    )
                    if updates:
                        stored = self._merge_news_records(stored, updates)

            wanted_start = self._parse_date(start_date)
            if wanted_start and oldest and wanted_start < oldest:
                backfill_end = oldest - timedelta(days=1)
                if wanted_start <= backfill_end:
                    backfill = self._model_dump_list(
                        self.provider.get_company_news(
                            ticker=ticker,
                            start_date=self._date_to_str(wanted_start),
                            end_date=self._date_to_str(backfill_end),
                            limit=max(limit * 3, 30),
                        )
                    )
                    if backfill:
                        stored = self._merge_news_records(stored, backfill)

        stored = self._sort_by_date(stored, "date", descending=True)
        if stored:
            self.cache.set_dataset("company_news", dataset_key, stored)

        filtered = self._filter_by_date_range(stored, "date", start_date, end_date)
        filtered = self._sort_by_date(filtered, "date", descending=True)
        return filtered[:limit]

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ):
        dataset_key = self._ticker_key(ticker)
        stored = self.cache.get_dataset("insider_trades", dataset_key) or []
        stored = self._sort_by_date(stored, "filing_date", descending=True)

        target_end = self._parse_date(end_date)
        if not stored:
            fetched = self._model_dump_list(
                self.provider.get_insider_trades(
                    ticker=ticker,
                    end_date=end_date,
                    start_date=start_date,
                    limit=limit,
                )
            )
            if fetched:
                stored = self._merge_insider_records(stored, fetched)
        else:
            newest = self._parse_date(stored[0].get("filing_date"))
            if newest and target_end and newest < target_end:
                append_start = newest + timedelta(days=1)
                if append_start <= target_end:
                    updates = self._model_dump_list(
                        self.provider.get_insider_trades(
                            ticker=ticker,
                            end_date=end_date,
                            start_date=self._date_to_str(append_start),
                            limit=limit,
                        )
                    )
                    if updates:
                        stored = self._merge_insider_records(stored, updates)

        stored = self._sort_by_date(stored, "filing_date", descending=True)
        if stored:
            self.cache.set_dataset("insider_trades", dataset_key, stored)

        filtered = self._filter_by_date_range(stored, "filing_date", start_date, end_date)
        filtered = self._sort_by_date(filtered, "filing_date", descending=True)
        return filtered[:limit]

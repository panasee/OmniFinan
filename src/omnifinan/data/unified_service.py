"""Cached data service wrapping a concrete provider."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from statistics import median
from typing import Any

from pyomnix.omnix_logger import get_logger

from .cache import DataCache
from .providers.base import DataProvider
from .providers.yfinance_provider import YFinanceProvider
from .symbols import is_crypto_ticker

logger = get_logger("unified_data_service")


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

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    return parsed.replace(tzinfo=None)
                return parsed
            except Exception:
                pass
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d"):
                try:
                    return datetime.strptime(text[:19], fmt)
                except ValueError:
                    continue
        return None

    def _interval_to_timedelta(self, interval: str) -> timedelta:
        text = str(interval or "1d").strip().lower()
        m = re.fullmatch(r"(\d+)\s*(m|h|d|w|wk|mo|q|y)", text)
        if not m:
            return timedelta(days=1)
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "m":
            return timedelta(minutes=n)
        if unit == "h":
            return timedelta(hours=n)
        if unit == "d":
            return timedelta(days=n)
        if unit in {"w", "wk"}:
            return timedelta(days=7 * n)
        if unit == "mo":
            return timedelta(days=30 * n)
        if unit == "q":
            return timedelta(days=90 * n)
        if unit == "y":
            return timedelta(days=365 * n)
        return timedelta(days=1)

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

    def _merge_macro_payloads(self, existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        if not existing:
            return incoming
        out = dict(existing)
        ex_series = existing.get("series", {}) if isinstance(existing.get("series"), dict) else {}
        in_series = incoming.get("series", {}) if isinstance(incoming.get("series"), dict) else {}
        merged_series: dict[str, dict[str, Any]] = {}

        all_keys = set(ex_series.keys()) | set(in_series.keys())
        for key in all_keys:
            a = ex_series.get(key, {}) if isinstance(ex_series.get(key), dict) else {}
            b = in_series.get(key, {}) if isinstance(in_series.get(key), dict) else {}
            a_obs = a.get("observations", []) if isinstance(a.get("observations"), list) else []
            b_obs = b.get("observations", []) if isinstance(b.get("observations"), list) else []
            obs_map: dict[str, dict[str, Any]] = {}
            for row in a_obs:
                if isinstance(row, dict) and row.get("date") is not None:
                    obs_map[str(row.get("date"))] = row
            for row in b_obs:
                if isinstance(row, dict) and row.get("date") is not None:
                    obs_map[str(row.get("date"))] = row
            merged_obs = sorted(obs_map.values(), key=lambda x: str(x.get("date", "")))
            merged = dict(a)
            merged.update(b)
            merged["observations"] = merged_obs
            if merged_obs:
                merged["latest"] = merged_obs[-1]
                merged["previous"] = merged_obs[-2] if len(merged_obs) > 1 else None
                # If we have valid observations after merge, treat transient fetch errors as resolved.
                merged["error"] = None
                if len(merged_obs) > 1:
                    prev_val = merged_obs[-2].get("value")
                    last_val = merged_obs[-1].get("value")
                    if isinstance(prev_val, int | float) and isinstance(last_val, int | float):
                        if last_val > prev_val:
                            merged["trend"] = "up"
                        elif last_val < prev_val:
                            merged["trend"] = "down"
                        else:
                            merged["trend"] = "flat"
            merged_series[key] = merged

        out.update(incoming)
        out["series"] = merged_series
        out["latest"] = {
            key: val.get("latest", {}).get("value")
            if isinstance(val, dict) and isinstance(val.get("latest"), dict)
            else None
            for key, val in merged_series.items()
        }
        return out

    def _is_non_empty_macro_payload(self, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        series = payload.get("series", {})
        if not isinstance(series, dict) or not series:
            return False
        for val in series.values():
            if not isinstance(val, dict):
                continue
            obs = val.get("observations", [])
            if isinstance(obs, list) and len(obs) > 0:
                return True
            latest = val.get("latest")
            if isinstance(latest, dict) and latest.get("value") is not None:
                return True
        return False

    def _macro_payload_has_gaps(self, payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return True
        series = payload.get("series", {})
        if not isinstance(series, dict) or not series:
            return True
        for key, val in series.items():
            if not isinstance(val, dict):
                return True
            if self._is_terminal_macro_unavailable(str(key), val):
                continue
            if val.get("error"):
                return True
            obs = val.get("observations", [])
            latest = val.get("latest")
            has_obs = isinstance(obs, list) and len(obs) > 0
            has_latest = isinstance(latest, dict) and latest.get("value") is not None
            if not has_obs and not has_latest:
                return True
        return False

    def _is_terminal_macro_unavailable(self, series_key: str, item: dict[str, Any]) -> bool:
        if series_key in {"us_pmi_manufacturing", "us_pmi_services"}:
            # PMI source migrated from fixed-unavailable placeholder to ISM public feed.
            return False
        source = str(item.get("source", ""))
        error = str(item.get("error", ""))
        if source == "fixed_sources_unavailable":
            return True
        return "unavailable in current fixed providers" in error

    def _macro_gap_keys(self, payload: dict[str, Any] | None) -> list[str]:
        if not isinstance(payload, dict):
            return []
        series = payload.get("series", {})
        if not isinstance(series, dict):
            return []
        gaps: list[str] = []
        for key, val in series.items():
            if not isinstance(val, dict):
                gaps.append(str(key))
                continue
            if self._is_terminal_macro_unavailable(str(key), val):
                continue
            if val.get("error"):
                gaps.append(str(key))
                continue
            obs = val.get("observations", [])
            latest = val.get("latest")
            has_obs = isinstance(obs, list) and len(obs) > 0
            has_latest = isinstance(latest, dict) and latest.get("value") is not None
            if not has_obs and not has_latest:
                gaps.append(str(key))
        return gaps

    def _macro_latest_date(self, item: dict[str, Any]) -> date | None:
        latest = item.get("latest")
        if isinstance(latest, dict):
            d = self._parse_date(latest.get("date"))
            if d is not None:
                return d
        obs = item.get("observations", [])
        if not isinstance(obs, list):
            return None
        dates = [self._parse_date(row.get("date")) for row in obs if isinstance(row, dict)]
        parsed = [d for d in dates if d is not None]
        return max(parsed) if parsed else None

    def _macro_cycle_days(self, series_key: str, item: dict[str, Any]) -> int | None:
        obs = item.get("observations", [])
        if isinstance(obs, list):
            parsed = sorted(
                {
                    d
                    for d in (self._parse_date(row.get("date")) for row in obs if isinstance(row, dict))
                    if d is not None
                }
            )
            if len(parsed) >= 2:
                deltas = [(parsed[i] - parsed[i - 1]).days for i in range(1, len(parsed))]
                positive = [d for d in deltas if d > 0]
                if positive:
                    inferred = int(round(float(median(positive))))
                    return max(1, min(400, inferred))

        key = series_key.lower()
        source = str(item.get("source", "")).lower()
        if source.startswith("world_bank:"):
            return 365
        if "gdp" in key or "quarter" in key:
            return 90
        if any(s in key for s in ("cpi", "ppi", "payroll", "retail", "industrial", "unemployment", "m2", "pmi")):
            return 30
        if any(
            s in key
            for s in (
                "sofr",
                "rate",
                "yield",
                "treasury",
                "spread",
                "vix",
                "equity",
                "commodity",
                "fx",
                "shibor",
                "lpr",
            )
        ):
            return 1
        return None

    def _macro_stale_keys(self, payload: dict[str, Any] | None, as_of: date) -> list[str]:
        if not isinstance(payload, dict):
            return []
        series = payload.get("series", {})
        if not isinstance(series, dict):
            return []
        stale: list[str] = []
        for key, val in series.items():
            if not isinstance(val, dict):
                stale.append(str(key))
                continue
            if self._is_terminal_macro_unavailable(str(key), val):
                continue
            if val.get("error"):
                stale.append(str(key))
                continue
            obs = val.get("observations", [])
            latest = val.get("latest")
            has_obs = isinstance(obs, list) and len(obs) > 0
            has_latest = isinstance(latest, dict) and latest.get("value") is not None
            if not has_obs and not has_latest:
                stale.append(str(key))
                continue
            last_date = self._macro_latest_date(val)
            if last_date is None:
                stale.append(str(key))
                continue
            cycle_days = self._macro_cycle_days(str(key), val)
            # Prefer per-series publication cycle, and only fallback to 30 days
            # when cycle inference is not available.
            stale_threshold_days = 30 if cycle_days is None else max(1, int(cycle_days) * 3)
            if (as_of - last_date).days > stale_threshold_days:
                stale.append(str(key))
        return stale

    def _macro_update_summary(self, before: dict[str, Any] | None, after: dict[str, Any]) -> str:
        if not isinstance(before, dict):
            return "initialized"
        before_series = before.get("series", {}) if isinstance(before.get("series"), dict) else {}
        after_series = after.get("series", {}) if isinstance(after.get("series"), dict) else {}
        changed: list[str] = []
        for key in sorted(set(before_series.keys()) | set(after_series.keys())):
            b = before_series.get(key, {}) if isinstance(before_series.get(key), dict) else {}
            a = after_series.get(key, {}) if isinstance(after_series.get(key), dict) else {}
            b_latest = b.get("latest", {}).get("value") if isinstance(b.get("latest"), dict) else None
            a_latest = a.get("latest", {}).get("value") if isinstance(a.get("latest"), dict) else None
            b_cnt = len(b.get("observations", [])) if isinstance(b.get("observations"), list) else 0
            a_cnt = len(a.get("observations", [])) if isinstance(a.get("observations"), list) else 0
            if b_latest != a_latest or b_cnt != a_cnt:
                changed.append(key)
        return ",".join(changed[:10]) if changed else ""

    def _filter_macro_payload_window(
        self,
        payload: dict[str, Any] | None,
        start_date: str | None,
        end_date: str | None,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return payload
        if not start_date and not end_date:
            return payload
        out = dict(payload)
        series = payload.get("series", {})
        if not isinstance(series, dict):
            return out
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        filtered_series: dict[str, dict[str, Any]] = {}
        latest_map: dict[str, Any] = {}
        for key, val in series.items():
            if not isinstance(val, dict):
                continue
            cloned = dict(val)
            obs = val.get("observations", [])
            if isinstance(obs, list):
                filtered_obs: list[dict[str, Any]] = []
                for row in obs:
                    if not isinstance(row, dict):
                        continue
                    d = self._parse_date(str(row.get("date")))
                    if d is None:
                        continue
                    if start and d < start:
                        continue
                    if end and d > end:
                        continue
                    filtered_obs.append(row)
                cloned["observations"] = filtered_obs
                if filtered_obs:
                    cloned["latest"] = filtered_obs[-1]
                    cloned["previous"] = filtered_obs[-2] if len(filtered_obs) > 1 else None
                    latest_map[key] = filtered_obs[-1].get("value")
                else:
                    # If window has no points, keep last known value for continuity.
                    if obs:
                        latest = val.get("latest")
                        previous = val.get("previous")
                        cloned["latest"] = latest if isinstance(latest, dict) else None
                        cloned["previous"] = previous if isinstance(previous, dict) else None
                        latest_map[key] = latest.get("value") if isinstance(latest, dict) else None
                        cloned["note"] = "window_no_points_using_last_known"
                    else:
                        latest = cloned.get("latest")
                        latest_map[key] = latest.get("value") if isinstance(latest, dict) else None
            filtered_series[key] = cloned
        out["series"] = filtered_series
        out["latest"] = latest_map
        return out

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
        force: bool = False,
    ):
        dataset_key = f"{self._ticker_key(ticker)}__{interval}"
        stored = self.cache.get_dataset("prices", dataset_key) or []
        stored = self._sort_by_date(stored, "time")
        price_provider = self._get_crypto_provider() if is_crypto_ticker(ticker) else self.provider

        target_start = self._parse_date(start_date)
        target_end = self._parse_date(end_date or self._today_str())
        now_dt = datetime.now()
        refresh_window = self._interval_to_timedelta(interval) * 2

        fetch_chunks: list[list[dict[str, Any]]] = []
        if force or not stored:
            initial = price_provider.get_prices(ticker, start_date, end_date, interval=interval)
            fetch_chunks.append(self._model_dump_list(initial))
        else:
            first_date = self._parse_date(stored[0].get("time"))
            last_date = self._parse_date(stored[-1].get("time"))
            last_dt = self._parse_datetime(stored[-1].get("time"))
            should_refresh_latest = (
                force
                or last_dt is None
                or (now_dt - last_dt) > refresh_window
            )

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

            if should_refresh_latest and target_end and last_date and target_end > last_date:
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

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str | None,
        period: str = "ttm",
        limit: int = 1,
        force: bool = False,
    ):
        dataset_key = f"{self._ticker_key(ticker)}__{period}"
        stored = self.cache.get_dataset("financial_metrics", dataset_key) or []
        stored = self._sort_by_date(stored, "report_period", descending=True)

        eligible = self._filter_by_date_range(stored, "report_period", None, end_date)
        latest_report_date = self._parse_date(stored[0].get("report_period")) if stored else None
        stale_financial = (
            latest_report_date is None
            or (date.today() - latest_report_date).days > 30
        )
        if force or stale_financial or len(eligible) < limit:
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

    def get_line_items(self, ticker: str, period: str = "ttm", limit: int = 10, force: bool = False):
        dataset_key = f"{self._ticker_key(ticker)}__{period}"
        stored = self.cache.get_dataset("line_items", dataset_key) or []
        stored = self._sort_by_date(stored, "report_period", descending=True)
        latest_report_date = self._parse_date(stored[0].get("report_period")) if stored else None
        stale_line_items = (
            latest_report_date is None
            or (date.today() - latest_report_date).days > 30
        )
        if force or stale_line_items or len(stored) < limit:
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

    def get_macro_indicators(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        force: bool = False,
    ):
        source_policy_version = "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank"
        dataset_key = f"{source_policy_version}__master"
        params = {
            "scope": "master",
            "source_policy_version": source_policy_version,
        }
        cached = self.cache.get("macro_indicators", params, ttl_seconds=self.ttl_seconds)
        payload = cached if isinstance(cached, dict) else None
        today = date.today()

        # Request-cache miss fallback: restore latest master snapshot from dataset history.
        if payload is None:
            history = self.cache.get_dataset("macro_indicators_history", dataset_key) or []
            if isinstance(history, list):
                candidates = [item for item in history if isinstance(item, dict)]
                if candidates:
                    candidates = sorted(candidates, key=lambda x: str(x.get("snapshot_at", "")))
                    payload = candidates[-1]
                    self.cache.set("macro_indicators", params, payload)
                    logger.info(
                        "Macro request cache restored from dataset master snapshot. snapshot_at=%s",
                        str(payload.get("snapshot_at", "")),
                    )

        if force:
            logger.info("Macro force refresh requested. window=%s~%s", start_date, end_date)

        # Series-level refresh policy:
        # - derive staleness from latest observation date + expected update cycle per series
        # - refresh only stale keys when subset endpoint is available
        if isinstance(payload, dict) and not force:
            has_non_empty = self._is_non_empty_macro_payload(payload)
            stale_keys = self._macro_stale_keys(payload, as_of=today)
            if has_non_empty and not stale_keys:
                logger.info(
                    "Macro fetch skipped: all series within expected update cycles. window=%s~%s",
                    start_date,
                    end_date,
                )
                return self._filter_macro_payload_window(payload, start_date, end_date)
            if stale_keys:
                if hasattr(self.provider, "get_macro_indicators_subset"):
                    logger.info(
                        "Macro master cache has stale series; refreshing subset only. stale=%d window=%s~%s",
                        len(stale_keys),
                        start_date,
                        end_date,
                    )
                    try:
                        fetch_subset = getattr(self.provider, "get_macro_indicators_subset")
                        partial = fetch_subset(series_keys=stale_keys, start_date=None, end_date=None)
                        if isinstance(partial, dict):
                            before = payload
                            payload = self._merge_macro_payloads(payload or {}, partial)
                            self.cache.set("macro_indicators", params, payload)
                            summary = self._macro_update_summary(before, payload)
                            if summary:
                                logger.info("Macro missing-series refresh updated: %s", summary)
                            else:
                                logger.info("Macro missing-series refresh had no data delta.")
                            return self._filter_macro_payload_window(payload, start_date, end_date)
                    except Exception:
                        logger.exception("Macro subset refresh failed; fallback to full refresh.")
                logger.info(
                    "Macro master cache has stale series and subset refresh unavailable; forcing full refresh. "
                    "stale=%d window=%s~%s",
                    len(stale_keys),
                    start_date,
                    end_date,
                )

        try:
            # Always refresh the master database using full available history; query windows are subsets.
            fresh = self.provider.get_macro_indicators(start_date=None, end_date=None)
            if isinstance(fresh, dict):
                before = payload
                payload = self._merge_macro_payloads(payload or {}, fresh)
                self.cache.set("macro_indicators", params, payload)
                summary = self._macro_update_summary(before, payload)
                if summary:
                    logger.info("Macro master data updated. query_window=%s~%s series=%s", start_date, end_date, summary)
                else:
                    logger.info("Macro master fetch completed with no data delta. query_window=%s~%s", start_date, end_date)
        except Exception:
            if payload is None:
                raise

        # Persist immutable snapshots for longitudinal analysis/tracking.
        if isinstance(payload, dict):
            snapshot_at = str(payload.get("snapshot_at", ""))
            history = self.cache.get_dataset("macro_indicators_history", dataset_key) or []
            if isinstance(history, list):
                already = any(
                    isinstance(item, dict) and str(item.get("snapshot_at", "")) == snapshot_at
                    for item in history
                )
                if not already:
                    history.append(payload)
                    history = history[-500:]
                    self.cache.set_dataset("macro_indicators_history", dataset_key, history)
        return self._filter_macro_payload_window(payload, start_date, end_date)

    def get_macro_indicators_structured(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        force: bool = False,
    ):
        from ..unified_api import structure_macro_indicators

        raw = self.get_macro_indicators(start_date=start_date, end_date=end_date, force=force)
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "source_policy_version": "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank",
            "view": "structured_v1",
        }
        cached = None if force else self.cache.get("macro_indicators_structured", params, ttl_seconds=self.ttl_seconds)
        if isinstance(cached, dict):
            return cached
        structured = structure_macro_indicators(raw if isinstance(raw, dict) else {})
        self.cache.set("macro_indicators_structured", params, structured)
        return structured

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

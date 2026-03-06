"""Cached data service wrapping a concrete provider."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from statistics import median
from typing import Any

from pyomnix.omnix_logger import get_logger

from ..analysis.options import compute_chain_analytics
from ..data_models import MarketType
from .cache import DataCache
from .news import fetch_search_news, integrate_news_rows
from .providers.base import DataProvider
from .providers.moomoo_options_provider import MoomooOptionsProvider
from .providers.yfinance_provider import YFinanceProvider
from .symbols import (
    is_crypto_ticker,
    is_non_option_equity_market_ticker,
    normalize_crypto_option_underlying,
)

logger = get_logger("unified_data_service")


class UnifiedDataService:
    def __init__(self, provider: DataProvider, cache: DataCache | None = None, ttl_seconds: int = 3600):
        self.provider = provider
        self.cache = cache or DataCache()
        self.ttl_seconds = ttl_seconds
        self._crypto_provider: DataProvider | None = None
        self._moomoo_options_provider: MoomooOptionsProvider | None = None
        self.cache.cleanup_expired(ttl_seconds)

    def _get_crypto_provider(self) -> DataProvider:
        if self._crypto_provider is None:
            self._crypto_provider = YFinanceProvider()
        return self._crypto_provider

    def _get_moomoo_options_provider(self) -> MoomooOptionsProvider:
        if self._moomoo_options_provider is None:
            self._moomoo_options_provider = MoomooOptionsProvider()
        return self._moomoo_options_provider

    def _ticker_key(self, ticker: str) -> str:
        return ticker.upper().strip()

    def _macro_source_policy_version(self) -> str:
        from ..unified_api import MACRO_SOURCE_POLICY_VERSION

        return MACRO_SOURCE_POLICY_VERSION

    def _retired_macro_series_keys(self) -> set[str]:
        from ..unified_api import RETIRED_MACRO_SERIES_KEYS

        return set(RETIRED_MACRO_SERIES_KEYS)

    def _prune_retired_macro_series(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return payload
        retired = self._retired_macro_series_keys()
        if not retired:
            return payload
        series = payload.get("series", {})
        latest = payload.get("latest", {})
        if not isinstance(series, dict) and not isinstance(latest, dict):
            return payload
        out = dict(payload)
        if isinstance(series, dict):
            out["series"] = {k: v for k, v in series.items() if k not in retired}
        if isinstance(latest, dict):
            out["latest"] = {k: v for k, v in latest.items() if k not in retired}
        return out

    def _retire_deprecated_macro_policy_caches(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        active_policy_version = self._macro_source_policy_version()
        deprecated_versions = (
            "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank",
            "fixed_sources_v2_with_dbnomics_proxies",
        )

        for deprecated_version in deprecated_versions:
            if deprecated_version == active_policy_version:
                continue
            self.cache.delete(
                "macro_indicators",
                {
                    "scope": "master",
                    "source_policy_version": deprecated_version,
                },
            )
            self.cache.delete(
                "macro_indicators_structured",
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "source_policy_version": deprecated_version,
                    "view": "structured_v1",
                },
            )
            self.cache.delete_dataset(
                "macro_indicators_history",
                f"{deprecated_version}__master",
            )

    def _unsupported_stock_option_payload(self, symbol: str, reason: str) -> dict[str, Any]:
        clean_symbol = str(symbol or "").strip().upper()
        return {
            "meta": {
                "source": "fixed_sources_unavailable",
                "asset_kind": "stock_option",
                "symbol": clean_symbol,
                "error": reason,
            },
            "data": [],
            "raw": {},
        }

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

    def _infer_contract_multiplier(self, symbol: str, contract_multiplier: float | None) -> float:
        if contract_multiplier is not None and contract_multiplier > 0:
            return float(contract_multiplier)
        s = str(symbol or "").strip().upper()
        if s in {".SPX", "SPX", "^SPX", "SPXW", ".NDX", "NDX", "^NDX", "NDXW"}:
            return 100.0
        return 100.0

    def _extract_latest_value(self, payload: dict[str, Any] | None, key: str) -> float | None:
        if not isinstance(payload, dict):
            return None
        series = payload.get("series", {}) if isinstance(payload.get("series"), dict) else {}
        item = series.get(key)
        if not isinstance(item, dict):
            return None
        latest = item.get("latest")
        if isinstance(latest, dict):
            val = latest.get("value")
            try:
                return float(val) if val is not None else None
            except Exception:
                return None
        obs = item.get("observations")
        if isinstance(obs, list) and obs:
            val = obs[-1].get("value") if isinstance(obs[-1], dict) else None
            try:
                return float(val) if val is not None else None
            except Exception:
                return None
        return None

    def _resolve_risk_free_rate(self, requested_rate: float | None) -> tuple[float, str]:
        if requested_rate is not None:
            return float(requested_rate), "input"

        try:
            macro = self.get_macro_indicators(force=False)
        except Exception:
            macro = None

        for key in ("us_treasury_2y", "us_treasury_10y"):
            val = self._extract_latest_value(macro, key)
            if val is not None:
                # Macro yields are usually in percent.
                rate = val / 100.0 if val > 1 else val
                return float(rate), f"macro:{key}"

        return 0.02, "fallback_default"

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

    def _detect_market(self, ticker: str) -> MarketType:
        from ..unified_api import detect_market, normalize_ticker

        return detect_market(normalize_ticker(ticker))

    def _normalize_raw_news(
        self,
        ticker: str,
        market: MarketType,
        items: list[Any],
    ) -> list[dict[str, Any]]:
        from .news import _coerce_company_news_rows

        return _coerce_company_news_rows(ticker, market, items)

    def _fetch_company_news_raw(
        self,
        ticker: str,
        market: MarketType,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if market in {MarketType.CHINA, MarketType.CHINA_SH, MarketType.CHINA_SZ, MarketType.CHINA_BJ}:
            return self._normalize_raw_news(
                ticker,
                market,
                self.provider.get_company_news_raw(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                ),
            )
        if market in {MarketType.US, MarketType.HK}:
            return fetch_search_news(
                ticker=ticker,
                market=market,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )
        return []

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
            return self._prune_retired_macro_series(incoming) or incoming
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
        return self._prune_retired_macro_series(out) or out

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

    def _min_refetch_seconds(self, cycle_days: int | None) -> int:
        """Minimum seconds between fetch attempts for a series, based on its update frequency."""
        if cycle_days is None:
            return 6 * 3600
        if cycle_days <= 1:
            return 6 * 3600
        if cycle_days <= 7:
            return 24 * 3600
        if cycle_days <= 30:
            return 3 * 24 * 3600
        if cycle_days <= 90:
            return 7 * 24 * 3600
        return 14 * 24 * 3600

    def _recently_fetched(
        self,
        val: dict[str, Any],
        cycle_days: int | None,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """Return True if the series was fetched recently enough that re-fetching is unnecessary."""
        fetched_at = val.get("fetched_at")
        if not fetched_at and isinstance(payload, dict):
            fetched_at = payload.get("snapshot_at")
        if not fetched_at:
            return False
        fetched_dt = self._parse_datetime(fetched_at)
        if fetched_dt is None:
            return False
        age_seconds = (datetime.now() - fetched_dt).total_seconds()
        cooldown = self._min_refetch_seconds(cycle_days)
        if val.get("fetched_at") is None and isinstance(payload, dict) and payload.get("snapshot_at"):
            cooldown = max(cooldown, 7 * 24 * 3600)
        return age_seconds < cooldown

    def _macro_stale_keys(self, payload: dict[str, Any] | None, as_of: date) -> list[str]:
        if not isinstance(payload, dict):
            return []
        series = payload.get("series", {})
        if not isinstance(series, dict):
            return []
        stale: list[str] = []
        cooldown_skipped: list[str] = []
        for key, val in series.items():
            if not isinstance(val, dict):
                stale.append(str(key))
                continue
            if self._is_terminal_macro_unavailable(str(key), val):
                continue
            cycle_days = self._macro_cycle_days(str(key), val)
            if val.get("error"):
                if not self._recently_fetched(val, cycle_days, payload):
                    stale.append(str(key))
                else:
                    cooldown_skipped.append(str(key))
                continue
            obs = val.get("observations", [])
            latest = val.get("latest")
            has_obs = isinstance(obs, list) and len(obs) > 0
            has_latest = isinstance(latest, dict) and latest.get("value") is not None
            if not has_obs and not has_latest:
                if not self._recently_fetched(val, cycle_days, payload):
                    stale.append(str(key))
                else:
                    cooldown_skipped.append(str(key))
                continue
            last_date = self._macro_latest_date(val)
            if last_date is None:
                if not self._recently_fetched(val, cycle_days, payload):
                    stale.append(str(key))
                else:
                    cooldown_skipped.append(str(key))
                continue
            stale_threshold_days = 30 if cycle_days is None else max(7, int(cycle_days) * 3)
            if (as_of - last_date).days > stale_threshold_days:
                if not self._recently_fetched(val, cycle_days, payload):
                    stale.append(str(key))
                else:
                    cooldown_skipped.append(str(key))
        if cooldown_skipped:
            logger.info(
                "Macro staleness check: %d series skipped (recently fetched): %s",
                len(cooldown_skipped),
                ",".join(cooldown_skipped[:10]),
            )
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
        return self._prune_retired_macro_series(out) or out

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
        self._retire_deprecated_macro_policy_caches(start_date=start_date, end_date=end_date)
        source_policy_version = self._macro_source_policy_version()
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
                    except Exception as exc:
                        logger.error(
                            "Macro subset refresh failed (force=False). Using cached payload. error=%s",
                            str(exc),
                        )
                        return self._filter_macro_payload_window(payload, start_date, end_date)
                logger.error(
                    "Macro subset refresh unavailable but stale series detected (force=False). "
                    "Using cached payload. stale=%d window=%s~%s",
                    len(stale_keys),
                    start_date,
                    end_date,
                )
                # Critical behavior: never auto full-refresh unless explicitly forced.
                # Full refresh can be dominated by slow providers (e.g., AkShare), and should be
                # a user-initiated operation.
                logger.error(
                    "Macro refresh degraded: provider lacks subset refresh; run with force=True to full refresh "+
                    "(may be slow) or implement get_macro_indicators_subset()."
                )
                return self._filter_macro_payload_window(payload, start_date, end_date)

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

        self._retire_deprecated_macro_policy_caches(start_date=start_date, end_date=end_date)
        raw = self.get_macro_indicators(start_date=start_date, end_date=end_date, force=force)
        source_policy_version = self._macro_source_policy_version()
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "source_policy_version": source_policy_version,
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
        self.cache.delete_dataset("company_news", dataset_key)
        market = self._detect_market(ticker)
        stored = self.cache.get_dataset("company_news_raw", dataset_key) or []
        stored_updated_at = self.cache.get_dataset_updated_at("company_news_raw", dataset_key)
        stored = self._sort_by_date(stored, "date", descending=True)

        target_end = self._parse_date(end_date or self._today_str())
        if not stored:
            fetched = self._fetch_company_news_raw(
                ticker=ticker,
                market=market,
                start_date=start_date,
                end_date=end_date,
                limit=max(limit, 30),
            )
            if fetched:
                stored = self._merge_news_records(stored, fetched)
                stored_updated_at = None
        else:
            newest = self._parse_date(stored[0].get("date"))
            oldest = self._parse_date(stored[-1].get("date"))
            recently_refreshed = (
                stored_updated_at is not None and (datetime.now() - stored_updated_at) <= timedelta(hours=24)
            )

            if newest and target_end and newest < target_end and not recently_refreshed:
                update_start = newest + timedelta(days=1)
                if update_start <= target_end:
                    updates = self._fetch_company_news_raw(
                        ticker=ticker,
                        market=market,
                        start_date=self._date_to_str(update_start),
                        end_date=self._date_to_str(target_end),
                        limit=max(limit * 3, 30),
                    )
                    if updates:
                        stored = self._merge_news_records(stored, updates)
                        stored_updated_at = None

            wanted_start = self._parse_date(start_date)
            if wanted_start and oldest and wanted_start < oldest and not recently_refreshed:
                backfill_end = oldest - timedelta(days=1)
                if wanted_start <= backfill_end:
                    backfill = self._fetch_company_news_raw(
                        ticker=ticker,
                        market=market,
                        start_date=self._date_to_str(wanted_start),
                        end_date=self._date_to_str(backfill_end),
                        limit=max(limit * 3, 30),
                    )
                    if backfill:
                        stored = self._merge_news_records(stored, backfill)
                        stored_updated_at = None

        stored = self._sort_by_date(stored, "date", descending=True)
        if stored:
            self.cache.set_dataset("company_news_raw", dataset_key, stored)
            stored_updated_at = self.cache.get_dataset_updated_at("company_news_raw", dataset_key)

        filtered = self._filter_by_date_range(stored, "date", start_date, end_date)
        filtered = self._sort_by_date(filtered, "date", descending=True)
        integrated_key = f"{dataset_key}__{start_date or 'none'}__{end_date or 'none'}__{limit}"
        cached_integrated = self.cache.get_dataset("company_news_integrated", integrated_key)
        integrated_updated_at = self.cache.get_dataset_updated_at("company_news_integrated", integrated_key)
        if (
            cached_integrated
            and integrated_updated_at is not None
            and stored_updated_at is not None
            and integrated_updated_at >= stored_updated_at
        ):
            return cached_integrated
        integrated = integrate_news_rows(
            ticker=ticker,
            market=market,
            rows=filtered,
            limit=limit,
        )
        dumped = self._model_dump_list(integrated)
        if dumped:
            self.cache.set_dataset("company_news_integrated", integrated_key, dumped)
        return dumped

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

    def get_stock_option_chain(
        self,
        symbol: str,
        expiration: str | None = None,
        option_type: str | None = None,
        strike: float | None = None,
        min_dte: int | None = None,
        max_dte: int | None = None,
        snapshot_mode: str = "prev_close",
        snapshot_date: str | None = None,
        provider: str = "auto",
        force: bool = False,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if is_non_option_equity_market_ticker(symbol):
            return self._unsupported_stock_option_payload(
                symbol,
                "stock options are not supported for China A-share/HK equities in current providers",
            )
        routed_symbol = normalize_crypto_option_underlying(symbol)
        params = {
            "symbol": routed_symbol,
            "requested_symbol": symbol,
            "expiration": expiration,
            "type": option_type,
            "strike": strike,
            "min_dte": min_dte,
            "max_dte": max_dte,
            "snapshot_mode": snapshot_mode,
            "snapshot_date": snapshot_date,
            "provider": provider,
            "extra_params": extra_params or {},
        }
        if not force:
            hit = self.cache.get("stock_option_chain", params, ttl_seconds=self.ttl_seconds)
            if isinstance(hit, dict):
                return hit

        p = str(provider or "auto").strip().lower()
        if p not in {"auto", "moomoo"}:
            raise ValueError("stock options provider must be one of {'auto', 'moomoo'}")

        payload = self._get_moomoo_options_provider().get_stock_option_chain(
            symbol=routed_symbol,
            expiration=expiration,
            option_type=option_type,
            strike=strike,
            min_dte=min_dte,
            max_dte=max_dte,
            snapshot_mode=snapshot_mode,
            snapshot_date=snapshot_date,
            extra_params=extra_params,
        )
        self.cache.set("stock_option_chain", params, payload)
        return payload

    def get_futures_option_chain(
        self,
        symbol: str,
        expiration: str | None = None,
        option_type: str | None = None,
        strike: float | None = None,
        min_dte: int | None = None,
        max_dte: int | None = None,
        snapshot_mode: str = "prev_close",
        snapshot_date: str | None = None,
        provider: str = "auto",
        force: bool = False,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        routed_symbol = normalize_crypto_option_underlying(symbol)
        params = {
            "symbol": routed_symbol,
            "requested_symbol": symbol,
            "expiration": expiration,
            "type": option_type,
            "strike": strike,
            "min_dte": min_dte,
            "max_dte": max_dte,
            "snapshot_mode": snapshot_mode,
            "snapshot_date": snapshot_date,
            "provider": provider,
            "extra_params": extra_params or {},
        }
        if not force:
            hit = self.cache.get("futures_option_chain", params, ttl_seconds=self.ttl_seconds)
            if isinstance(hit, dict):
                return hit
        p = str(provider or "auto").strip().lower()
        if p not in {"auto", "moomoo"}:
            raise ValueError("futures options provider must be one of {'auto', 'moomoo'}")
        payload = {
            "meta": {
                "source": "fixed_sources_unavailable",
                "asset_kind": "futures_option",
                "symbol": routed_symbol,
                "requested_symbol": symbol,
                "error": "futures option chains are not supported in the current moomoo-only provider stack",
            },
            "data": [],
            "raw": {},
        }
        self.cache.set("futures_option_chain", params, payload)
        return payload

    def get_stock_option_chain_analytics(
        self,
        symbol: str,
        expiration: str | None = None,
        option_type: str | None = None,
        strike: float | None = None,
        min_dte: int | None = None,
        max_dte: int | None = None,
        snapshot_mode: str = "prev_close",
        snapshot_date: str | None = None,
        provider: str = "auto",
        force: bool = False,
        extra_params: dict[str, Any] | None = None,
        underlying_price: float | None = None,
        risk_free_rate: float | None = None,
        dividend_yield: float = 0.0,
        iv_history: list[float] | None = None,
        price_history: list[dict[str, Any]] | None = None,
        include_realized_vol: bool = True,
        hv_lookback_days: int = 20,
        contract_multiplier: float | None = None,
    ) -> dict[str, Any]:
        if is_non_option_equity_market_ticker(symbol):
            payload = self._unsupported_stock_option_payload(
                symbol,
                "stock options analytics are unavailable for China A-share/HK equities",
            )
            return {
                "meta": {**payload["meta"], "analytics_version": "options_analytics_v1"},
                "data": [],
                "raw": {},
                "analytics": {
                    "summary": {
                        "option_count": 0,
                        "enriched_count": 0,
                        "underlying_price": None,
                        "median_iv": None,
                        "iv_historical_percentile": None,
                    },
                    "surface": [],
                    "term_structure": [],
                    "skew_by_expiry": [],
                    "smile_by_expiry": [],
                    "max_pain": {"overall": None, "by_expiry": []},
                    "levels": {},
                    "implied_vs_realized": {},
                    "errors": [payload["meta"]["error"]],
                },
            }
        routed_symbol = normalize_crypto_option_underlying(symbol)
        params = {
            "symbol": routed_symbol,
            "requested_symbol": symbol,
            "expiration": expiration,
            "type": option_type,
            "strike": strike,
            "min_dte": min_dte,
            "max_dte": max_dte,
            "snapshot_mode": snapshot_mode,
            "snapshot_date": snapshot_date,
            "provider": provider,
            "extra_params": extra_params or {},
            "underlying_price": underlying_price,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield,
            "contract_multiplier": contract_multiplier,
            "iv_history_len": len(iv_history) if isinstance(iv_history, list) else 0,
            "price_history_len": len(price_history) if isinstance(price_history, list) else 0,
            "include_realized_vol": include_realized_vol,
            "hv_lookback_days": hv_lookback_days,
            "analytics_version": "options_analytics_v1",
        }
        if not force:
            hit = self.cache.get("stock_option_chain_analytics", params, ttl_seconds=self.ttl_seconds)
            if isinstance(hit, dict):
                return hit

        payload = self.get_stock_option_chain(
            symbol=routed_symbol,
            expiration=expiration,
            option_type=option_type,
            strike=strike,
            min_dte=min_dte,
            max_dte=max_dte,
            snapshot_mode=snapshot_mode,
            snapshot_date=snapshot_date,
            provider=provider,
            force=force,
            extra_params=extra_params,
        )
        rows = payload.get("data", [])
        resolved_snapshot = snapshot_date or str(payload.get("meta", {}).get("snapshot_date") or "")
        hv_prices = price_history
        if include_realized_vol and not isinstance(hv_prices, list):
            try:
                end_dt = self._parse_date(resolved_snapshot) or date.today()
                # Pull enough bars to account for weekends/holidays.
                start_dt = end_dt - timedelta(days=max(hv_lookback_days * 3, 60))
                hv_prices = self._model_dump_list(
                    self.get_prices(
                        ticker=symbol,
                        start_date=self._date_to_str(start_dt),
                        end_date=self._date_to_str(end_dt),
                    )
                )
            except Exception:
                hv_prices = []
        resolved_risk_free_rate, risk_free_rate_source = self._resolve_risk_free_rate(risk_free_rate)
        resolved_contract_multiplier = self._infer_contract_multiplier(symbol, contract_multiplier)

        # Underlying spot resolution policy:
        # - Do NOT use options-provider injected underlying prices.
        # - Prefer OmniFinan price interfaces (get_prices) and use latest close.
        resolved_underlying_price = underlying_price
        resolved_underlying_price_source: str | None = None
        if resolved_underlying_price is None:
            try:
                if isinstance(hv_prices, list) and hv_prices:
                    last = hv_prices[-1]
                    close_v = last.get("close")
                    if close_v is not None:
                        resolved_underlying_price = float(close_v)
                        resolved_underlying_price_source = f"prices:close@{last.get('time')}"
            except Exception:
                pass

        if resolved_underlying_price is None:
            try:
                end_dt = self._parse_date(resolved_snapshot) or date.today()
                start_dt = end_dt - timedelta(days=10)
                px_rows = self._model_dump_list(
                    self.get_prices(
                        ticker=symbol,
                        start_date=self._date_to_str(start_dt),
                        end_date=self._date_to_str(end_dt),
                        interval="1d",
                        force=False,
                    )
                )
                if px_rows:
                    last = px_rows[-1]
                    close_v = last.get("close")
                    if close_v is not None:
                        resolved_underlying_price = float(close_v)
                        resolved_underlying_price_source = f"prices:close@{last.get('time')}"
            except Exception:
                pass

        analytics = compute_chain_analytics(
            rows if isinstance(rows, list) else [],
            underlying_price=resolved_underlying_price,
            risk_free_rate=resolved_risk_free_rate,
            dividend_yield=dividend_yield,
            snapshot_date=resolved_snapshot,
            price_history=hv_prices if isinstance(hv_prices, list) else [],
            iv_history=iv_history if isinstance(iv_history, list) else None,
            hv_lookback_days=hv_lookback_days,
            contract_multiplier=resolved_contract_multiplier,
        )

        result = {
            "meta": {
                **(payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}),
                "analytics_version": "options_analytics_v1",
                "risk_free_rate": resolved_risk_free_rate,
                "risk_free_rate_source": risk_free_rate_source,
                "contract_multiplier": resolved_contract_multiplier,
                "underlying_price_source": resolved_underlying_price_source,
            },
            "data": rows if isinstance(rows, list) else [],
            "raw": payload.get("raw", {}) if isinstance(payload.get("raw"), dict) else {},
            "analytics": analytics,
        }
        self.cache.set("stock_option_chain_analytics", params, result)
        return result

    def get_stock_option_gex(
        self,
        symbol: str,
        expiration: str | None = None,
        gex_expiration: str | None = None,
        option_type: str | None = None,
        strike: float | None = None,
        min_dte: int | None = None,
        max_dte: int | None = None,
        snapshot_mode: str = "prev_close",
        snapshot_date: str | None = None,
        provider: str = "auto",
        force: bool = False,
        extra_params: dict[str, Any] | None = None,
        underlying_price: float | None = None,
        risk_free_rate: float | None = None,
        dividend_yield: float = 0.0,
        iv_history: list[float] | None = None,
        price_history: list[dict[str, Any]] | None = None,
        include_realized_vol: bool = True,
        hv_lookback_days: int = 20,
        contract_multiplier: float | None = None,
    ) -> dict[str, Any]:
        """Return estimated GEX summary for a stock/index option chain.

        GEX is estimated from Black-Scholes gamma and chain open interest.
        """
        resolved_expiration = gex_expiration if gex_expiration is not None else expiration

        analytics_payload = self.get_stock_option_chain_analytics(
            symbol=symbol,
            expiration=resolved_expiration,
            option_type=option_type,
            strike=strike,
            min_dte=min_dte,
            max_dte=max_dte,
            snapshot_mode=snapshot_mode,
            snapshot_date=snapshot_date,
            provider=provider,
            force=force,
            extra_params=extra_params,
            underlying_price=underlying_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            iv_history=iv_history,
            price_history=price_history,
            include_realized_vol=include_realized_vol,
            hv_lookback_days=hv_lookback_days,
            contract_multiplier=contract_multiplier,
        )
        summary = (((analytics_payload.get("analytics") or {}).get("summary")) or {})
        meta = analytics_payload.get("meta") or {}
        rows = analytics_payload.get("data") if isinstance(analytics_payload.get("data"), list) else []

        spot = float(summary.get("underlying_price") or 0.0)
        rows_total = int(summary.get("option_count") or len(rows) or 0)
        rows_usable = int(summary.get("enriched_count") or 0)
        quality = float(rows_usable / rows_total) if rows_total > 0 else 0.0
        contract_mult = float(summary.get("contract_multiplier") or meta.get("contract_multiplier") or 100.0)
        risk_free = float(summary.get("risk_free_rate") or meta.get("risk_free_rate") or 0.02)

        # Build strike profile + term buckets from raw chain rows.
        def _norm_pdf(x: float) -> float:
            return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

        now_ts = datetime.utcnow().timestamp()
        sec_per_year = 365.0 * 24.0 * 3600.0
        net_by_strike: dict[float, float] = {}
        bucket_net = {"zero_dte": 0.0, "near_term_1w": 0.0, "long_term": 0.0}
        total_volume = 0.0
        total_abs_gex = 0.0
        total_vanna_exp = 0.0
        total_charm_exp = 0.0
        vanna_call_exp = 0.0
        vanna_put_exp = 0.0
        charm_call_exp = 0.0
        charm_put_exp = 0.0
        gex_inputs: list[tuple[float, float, float, float, float]] = []

        for r in rows:
            try:
                k = float(r.get("strike")) if r.get("strike") is not None else None
                iv = float(r.get("iv")) if r.get("iv") is not None else None
                oi = float(r.get("openInterest")) if r.get("openInterest") is not None else None
                exp = float(r.get("expiration")) if r.get("expiration") is not None else None
                side = str(r.get("side") or "").lower()
                vol = float(r.get("volume")) if r.get("volume") is not None else None
            except Exception:
                continue
            if vol is not None and vol >= 0:
                total_volume += vol
            if None in (k, iv, oi, exp) or k <= 0 or iv <= 0 or oi < 0 or spot <= 0:
                continue
            t = max((exp - now_ts) / sec_per_year, 1.0 / 365.0)
            d1 = (math.log(spot / k) + (risk_free + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
            d2 = d1 - iv * math.sqrt(t)
            pdf_d1 = _norm_pdf(d1)
            gamma = pdf_d1 / (spot * iv * math.sqrt(t))
            sign = 1.0 if side == "call" else -1.0
            gex = sign * gamma * oi * contract_mult * (spot**2) * 0.01
            gex_inputs.append((k, iv, oi, exp, sign))

            # Estimated vanna/charm exposures for risk-profile diagnostics.
            vanna = pdf_d1 * (-d2 / iv)
            vanna_exp = sign * vanna * oi * contract_mult * spot * 0.01

            charm = -pdf_d1 * (risk_free / (iv * math.sqrt(t)) - d2 / (2.0 * t))
            charm_exp = sign * charm * oi * contract_mult * spot * (1.0 / 365.0)

            total_vanna_exp += vanna_exp
            total_charm_exp += charm_exp
            if side == "call":
                vanna_call_exp += vanna_exp
                charm_call_exp += charm_exp
            else:
                vanna_put_exp += vanna_exp
                charm_put_exp += charm_exp
            total_abs_gex += abs(gex)
            net_by_strike[k] = net_by_strike.get(k, 0.0) + gex

            dte = (exp - now_ts) / (24.0 * 3600.0)
            if dte <= 1.0:
                bucket_net["zero_dte"] += gex
            elif dte <= 7.0:
                bucket_net["near_term_1w"] += gex
            else:
                bucket_net["long_term"] += gex

        strikes = sorted(net_by_strike.keys())

        # Spot-sweep zero-gamma root (institutional-style): solve GEX(S') = 0 over a spot grid.
        def _net_gex_at_spot(test_spot: float) -> float:
            if test_spot <= 0:
                return 0.0
            total = 0.0
            for k, iv, oi, exp, sign in gex_inputs:
                t = max((exp - now_ts) / sec_per_year, 1.0 / 365.0)
                d1 = (math.log(test_spot / k) + (risk_free + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
                gamma = _norm_pdf(d1) / (test_spot * iv * math.sqrt(t))
                total += sign * gamma * oi * contract_mult * (test_spot**2) * 0.01
            return total

        gamma_flip_price = None
        if spot > 0 and gex_inputs:
            grid_min = spot * 0.7
            grid_max = spot * 1.3
            steps = 121
            grid = [grid_min + (grid_max - grid_min) * i / (steps - 1) for i in range(steps)]
            vals = [_net_gex_at_spot(sv) for sv in grid]

            for i in range(1, len(grid)):
                x0, y0 = grid[i - 1], vals[i - 1]
                x1, y1 = grid[i], vals[i]
                if y0 == 0:
                    gamma_flip_price = x0
                    break
                if y1 == 0:
                    gamma_flip_price = x1
                    break
                if y0 * y1 < 0:
                    gamma_flip_price = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
                    break

        zero_gamma_distance_pct = (
            abs(spot - float(gamma_flip_price)) / spot * 100.0 if gamma_flip_price is not None and spot > 0 else None
        )

        call_wall = max(strikes, key=lambda x: net_by_strike.get(x, 0.0)) if strikes else None
        put_wall = min(strikes, key=lambda x: net_by_strike.get(x, 0.0)) if strikes else None
        max_gamma_strike = max(strikes, key=lambda x: abs(net_by_strike.get(x, 0.0))) if strikes else None

        total_abs_bucket = sum(abs(v) for v in bucket_net.values())

        # Gamma concentration index: top-5 absolute strike GEX share.
        abs_vals = sorted((abs(v) for v in net_by_strike.values()), reverse=True)
        top5_abs = sum(abs_vals[:5])
        all_abs = sum(abs_vals)
        concentration_index = (top5_abs / all_abs) if all_abs > 0 else None

        def _pct(v: float) -> float | None:
            return (abs(v) / total_abs_bucket * 100.0) if total_abs_bucket > 0 else None

        # Sample three strikes closest to spot.
        profile_data_sample: list[dict[str, Any]] = []
        if strikes and spot > 0:
            near = sorted(strikes, key=lambda x: abs(x - spot))[:3]
            for k in sorted(near):
                val = net_by_strike.get(k, 0.0)
                if val > 0:
                    bias = "call_heavy"
                elif val < 0:
                    bias = "put_heavy"
                else:
                    bias = "balanced"
                profile_data_sample.append({"strike": float(k), "net_gex": float(val), "side_bias": bias})

        # Approx OPEX-week flag: current week contains 3rd Friday.
        today = datetime.utcnow().date()
        first_day = today.replace(day=1)
        weekday_first = first_day.weekday()  # Mon=0
        first_friday = 1 + ((4 - weekday_first) % 7)
        third_friday = first_friday + 14
        opex_date = today.replace(day=third_friday)
        is_opex_week = (today - timedelta(days=today.weekday())) <= opex_date <= (today + timedelta(days=6 - today.weekday()))

        mean_iv = None
        iv_vals = [float(r.get("iv")) for r in rows if r.get("iv") is not None]
        if iv_vals:
            mean_iv = sum(iv_vals) / len(iv_vals)

        net_gex = float(summary.get("net_gex_est") or 0.0)
        call_gex_nominal = float(summary.get("call_gex_est") or 0.0)
        put_gex_nominal = float(summary.get("put_gex_est") or 0.0)
        put_share = (abs(put_gex_nominal) / (abs(call_gex_nominal) + abs(put_gex_nominal))) if (abs(call_gex_nominal) + abs(put_gex_nominal)) > 0 else None
        gamma_skew = (abs(put_gex_nominal) / abs(call_gex_nominal)) if abs(call_gex_nominal) > 0 else None
        gamma_purity = (net_gex / total_abs_gex) if total_abs_gex > 0 else None
        tail_risk_warning = bool((put_share is not None and put_share > 0.7) and (net_gex < 0.0))

        zero_dte_pct = _pct(bucket_net["zero_dte"])
        spot_near_flip = bool(
            gamma_flip_price is not None
            and spot > 0
            and (abs(spot - float(gamma_flip_price)) / spot) <= 0.01
        )

        vanna_regime = "positive_vanna" if total_vanna_exp >= 0 else "negative_vanna"
        charm_regime = "positive_charm" if total_charm_exp >= 0 else "negative_charm"
        flow_interpretation = {
            "vanna_regime": vanna_regime,
            "charm_regime": charm_regime,
            "tail_risk_warning": tail_risk_warning,
            "notes": [
                "negative_vanna often amplifies rebound sensitivity when IV compresses",
                "positive_charm can create passive buy-to-hedge drift into expiry",
                "negative_charm can create passive sell-to-hedge drift into expiry",
            ],
        }

        # Underlying dollar-volume proxy (20D avg) for GEX dominance checks.
        avg_underlying_dollar_volume_20d = None
        gex_to_underlying_dollar_volume_ratio = None
        try:
            end_dt = datetime.utcnow().date()
            start_dt = end_dt - timedelta(days=40)
            px_rows = self.get_prices(
                ticker=str(symbol).strip().upper(),
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d"),
                interval="1d",
                force=False,
            )
            if isinstance(px_rows, list) and px_rows:
                notional_series: list[float] = []
                for pr in px_rows:
                    close_v = pr.get("close")
                    vol_v = pr.get("volume")
                    try:
                        close_f = float(close_v)
                        vol_f = float(vol_v)
                    except (TypeError, ValueError):
                        continue
                    if close_f > 0 and vol_f >= 0:
                        notional_series.append(close_f * vol_f)
                if notional_series:
                    tail = notional_series[-20:]
                    avg_underlying_dollar_volume_20d = sum(tail) / len(tail)
                    if avg_underlying_dollar_volume_20d > 0:
                        gex_to_underlying_dollar_volume_ratio = net_gex / avg_underlying_dollar_volume_20d
        except Exception:
            avg_underlying_dollar_volume_20d = None
            gex_to_underlying_dollar_volume_ratio = None

        if avg_underlying_dollar_volume_20d is None:
            # Fallback for indices where primary provider price history is empty/intermittent.
            try:
                import yfinance as yf  # type: ignore

                yf_symbol = str(symbol).strip().upper()
                if yf_symbol in {".SPX", "SPX", "SPXW", "^SPX"}:
                    yf_symbol = "^GSPC"
                hist = yf.download(
                    yf_symbol,
                    period="3mo",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                )
                if hist is not None and not hist.empty:
                    closes = None
                    vols = None
                    if getattr(hist.columns, "nlevels", 1) > 1:
                        sym_key = yf_symbol
                        if ("Close", sym_key) in hist.columns and ("Volume", sym_key) in hist.columns:
                            closes = hist[("Close", sym_key)].astype(float)
                            vols = hist[("Volume", sym_key)].astype(float)
                    else:
                        if "Close" in hist.columns and "Volume" in hist.columns:
                            closes = hist["Close"].astype(float)
                            vols = hist["Volume"].astype(float)
                    if closes is not None and vols is not None:
                        notionals = (closes * vols).dropna().tolist()
                        if notionals:
                            tail = notionals[-20:]
                            avg_underlying_dollar_volume_20d = sum(tail) / len(tail)
                            if avg_underlying_dollar_volume_20d > 0:
                                gex_to_underlying_dollar_volume_ratio = net_gex / avg_underlying_dollar_volume_20d
            except Exception:
                pass

        out = {
            "metadata": {
                "symbol": str(symbol).strip().upper(),
                "spot_price": spot,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gamma_flip_price": float(gamma_flip_price) if gamma_flip_price is not None else None,
                "data_quality": {
                    "rows_total": rows_total,
                    "rows_usable": rows_usable,
                    "quality_score": quality,
                    "contract_multiplier": contract_mult,
                    "mean_iv": mean_iv,
                },
            },
            "aggregate_stats": {
                "net_gex_nominal": net_gex,
                "call_gex_nominal": call_gex_nominal,
                "put_gex_nominal": put_gex_nominal,
                "regime": summary.get("gamma_regime_est") or ("positive_gamma" if net_gex >= 0 else "negative_gamma"),
                "alt_sign_convention": float(summary.get("net_gex_alt_sign") or -net_gex),
                "gex_to_volume_ratio": (float(net_gex) / total_volume) if total_volume > 0 else None,
                "total_abs_gex": total_abs_gex,
                "gamma_purity": gamma_purity,
                "concentration_index": concentration_index,
                "gamma_skew": gamma_skew,
                "put_gex_share": put_share,
                "tail_risk_warning": tail_risk_warning,
                "avg_underlying_dollar_volume_20d": avg_underlying_dollar_volume_20d,
                "gex_to_underlying_dollar_volume_ratio": gex_to_underlying_dollar_volume_ratio,
            },
            "key_levels": {
                "gamma_flip_price": float(gamma_flip_price) if gamma_flip_price is not None else None,
                "zero_gamma_distance_pct": float(zero_gamma_distance_pct) if zero_gamma_distance_pct is not None else None,
                "call_wall": float(call_wall) if call_wall is not None else None,
                "put_wall": float(put_wall) if put_wall is not None else None,
                "max_gamma_strike": float(max_gamma_strike) if max_gamma_strike is not None else None,
            },
            "risk_profile": {
                "vanna_exposure*": float(total_vanna_exp),
                "charm_exposure*": float(total_charm_exp),
                "vanna_call_exposure*": float(vanna_call_exp),
                "vanna_put_exposure*": float(vanna_put_exp),
                "charm_call_exposure*": float(charm_call_exp),
                "charm_put_exposure*": float(charm_put_exp),
                "is_opex_week": bool(is_opex_week),
                "volatility_bias": "stabilizing" if net_gex >= 0 else "amplifying",
            },
            "term_structure": {
                "zero_dte": {"net_gex": float(bucket_net["zero_dte"]), "pct": zero_dte_pct},
                "near_term_1w": {"net_gex": float(bucket_net["near_term_1w"]), "pct": _pct(bucket_net["near_term_1w"])},
                "long_term": {"net_gex": float(bucket_net["long_term"]), "pct": _pct(bucket_net["long_term"])},
            },
            "profile_data_sample": profile_data_sample,
            "flow_interpretation": flow_interpretation,
            "regime_mapping": {
                "volatility_acceleration": {
                    "condition": net_gex < 0.0,
                    "trigger": "net_gex_nominal < 0",
                    "advice": "Avoid aggressive left-side dip buying; place protective stop below put_wall.",
                },
                "transition_zone": {
                    "condition": spot_near_flip,
                    "trigger": "abs(spot-gamma_flip_price)/spot <= 1%",
                    "advice": "Volatility regime may switch soon; reduce short-vol inventory / consider take-profit on short-vol trades.",
                },
                "speculative_noise": {
                    "condition": bool(zero_dte_pct is not None and zero_dte_pct > 50.0),
                    "trigger": "zero_dte_pct > 50%",
                    "advice": "Signal is mainly intraday; avoid using it as standalone swing horizon anchor.",
                },
            },
            "assumptions": {
                "method": "Black-Scholes gamma estimated from chain iv/oi",
                "risk_free_rate": risk_free,
                "risk_free_rate_source": meta.get("risk_free_rate_source"),
                "expiration": resolved_expiration,
            },
        }
        return out

    def get_futures_option_chain_analytics(
        self,
        symbol: str,
        expiration: str | None = None,
        option_type: str | None = None,
        strike: float | None = None,
        min_dte: int | None = None,
        max_dte: int | None = None,
        snapshot_mode: str = "prev_close",
        snapshot_date: str | None = None,
        provider: str = "auto",
        force: bool = False,
        extra_params: dict[str, Any] | None = None,
        underlying_price: float | None = None,
        risk_free_rate: float = 0.02,
        dividend_yield: float = 0.0,
        iv_history: list[float] | None = None,
        price_history: list[dict[str, Any]] | None = None,
        hv_lookback_days: int = 20,
    ) -> dict[str, Any]:
        routed_symbol = normalize_crypto_option_underlying(symbol)
        params = {
            "symbol": routed_symbol,
            "requested_symbol": symbol,
            "expiration": expiration,
            "type": option_type,
            "strike": strike,
            "min_dte": min_dte,
            "max_dte": max_dte,
            "snapshot_mode": snapshot_mode,
            "snapshot_date": snapshot_date,
            "provider": provider,
            "extra_params": extra_params or {},
            "underlying_price": underlying_price,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield,
            "iv_history_len": len(iv_history) if isinstance(iv_history, list) else 0,
            "price_history_len": len(price_history) if isinstance(price_history, list) else 0,
            "hv_lookback_days": hv_lookback_days,
            "analytics_version": "options_analytics_v1",
        }
        if not force:
            hit = self.cache.get("futures_option_chain_analytics", params, ttl_seconds=self.ttl_seconds)
            if isinstance(hit, dict):
                return hit

        payload = self.get_futures_option_chain(
            symbol=routed_symbol,
            expiration=expiration,
            option_type=option_type,
            strike=strike,
            min_dte=min_dte,
            max_dte=max_dte,
            snapshot_mode=snapshot_mode,
            snapshot_date=snapshot_date,
            provider=provider,
            force=force,
            extra_params=extra_params,
        )
        rows = payload.get("data", [])
        analytics = compute_chain_analytics(
            rows if isinstance(rows, list) else [],
            underlying_price=underlying_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            snapshot_date=snapshot_date or str(payload.get("meta", {}).get("snapshot_date") or ""),
            price_history=price_history if isinstance(price_history, list) else [],
            iv_history=iv_history if isinstance(iv_history, list) else None,
            hv_lookback_days=hv_lookback_days,
        )

        result = {
            "meta": {
                **(payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}),
                "analytics_version": "options_analytics_v1",
            },
            "data": rows if isinstance(rows, list) else [],
            "raw": payload.get("raw", {}) if isinstance(payload.get("raw"), dict) else {},
            "analytics": analytics,
        }
        self.cache.set("futures_option_chain_analytics", params, result)
        return result

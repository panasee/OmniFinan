"""Finnhub-backed provider implementation."""

from __future__ import annotations

from datetime import datetime

import requests
from pyomnix.omnix_logger import get_logger

from ...data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from ...unified_api import detect_market, normalize_ticker
from .akshare_provider import AkshareProvider
from .base import DataProvider
from .credentials import get_api_key

logger = get_logger("finnhub_provider")


class FinnhubProvider(DataProvider):
    def __init__(self, api_key: str | None = None, timeout: int = 20):
        self.api_key = api_key or get_api_key("finnhub")
        self.timeout = timeout
        self._fallback = AkshareProvider()

    def _require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("Finnhub API key missing. Configure OMNIX_PATH/finn_api.json")
        return self.api_key

    def _resolution(self, interval: str) -> str:
        mapping = {
            "1m": "1",
            "3m": "1",  # no 3m native; caller can resample
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "60m": "60",
            "1d": "D",
        }
        return mapping.get(interval, "D")

    @staticmethod
    def _first_metric(metric: dict, keys: list[str]) -> float | None:
        for key in keys:
            value = metric.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _is_access_denied(response: requests.Response) -> bool:
        if response.status_code not in {401, 403}:
            return False
        try:
            text = response.text or ""
        except Exception:
            text = ""
        return "don't have access to this resource" in text.lower()

    def get_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> list[Price]:
        key = self._require_key()
        symbol = normalize_ticker(ticker)
        market = detect_market(symbol)
        start_ts = int(pd_to_datetime(start_date, floor=True).timestamp())
        end_ts = int(pd_to_datetime(end_date, floor=False).timestamp())
        resolution = self._resolution(interval)
        # Finnhub expects US symbols like AAPL; skip unsupported markets for now.
        if market.name != "US":
            return self._fallback.get_prices(ticker, start_date, end_date, interval=interval)

        resp = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={
                "symbol": symbol,
                "resolution": resolution,
                "from": start_ts,
                "to": end_ts,
                "token": key,
            },
            timeout=self.timeout,
        )
        if self._is_access_denied(resp):
            logger.warning(
                "Finnhub stock/candle access denied for %s. Falling back to AkShare.",
                symbol,
            )
            return self._fallback.get_prices(ticker, start_date, end_date, interval=interval)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("s") != "ok":
            return []

        out: list[Price] = []
        ts_list = payload.get("t", [])
        opens = payload.get("o", [])
        highs = payload.get("h", [])
        lows = payload.get("l", [])
        closes = payload.get("c", [])
        vols = payload.get("v", [])

        for i, ts in enumerate(ts_list):
            dt = datetime.utcfromtimestamp(ts)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S") if resolution != "D" else dt.strftime("%Y-%m-%d")
            volume = int(vols[i]) if i < len(vols) else 0
            close = float(closes[i]) if i < len(closes) else 0.0
            out.append(
                Price(
                    open=float(opens[i]) if i < len(opens) else close,
                    high=float(highs[i]) if i < len(highs) else close,
                    low=float(lows[i]) if i < len(lows) else close,
                    close=close,
                    volume=max(volume, 0),
                    amount=float(volume) * close,
                    time=time_str,
                    market=market,
                )
            )
        return out

    def get_financial_metrics(
        self, ticker: str, end_date: str | None = None, period: str = "ttm", limit: int = 1
    ) -> list[FinancialMetrics]:
        key = self._require_key()
        symbol = normalize_ticker(ticker)
        market = detect_market(symbol)

        # Finnhub financial endpoints are focused on US symbols.
        if market.name != "US":
            return self._fallback.get_financial_metrics(ticker, end_date, period, limit)

        metric_resp = requests.get(
            "https://finnhub.io/api/v1/stock/metric",
            params={"symbol": symbol, "metric": "all", "token": key},
            timeout=self.timeout,
        )
        metric_resp.raise_for_status()
        metric_payload = metric_resp.json()
        metric = metric_payload.get("metric", {}) if isinstance(metric_payload, dict) else {}

        profile_resp = requests.get(
            "https://finnhub.io/api/v1/stock/profile2",
            params={"symbol": symbol, "token": key},
            timeout=self.timeout,
        )
        profile_resp.raise_for_status()
        profile_payload = profile_resp.json()
        profile = profile_payload if isinstance(profile_payload, dict) else {}

        report_period = end_date or datetime.utcnow().strftime("%Y-%m-%d")
        market_cap = self._first_metric(metric, ["marketCapitalization"])
        enterprise_value = self._first_metric(metric, ["enterpriseValue"])
        # Finnhub profile2 marketCapitalization is usually in millions.
        if market_cap is None:
            profile_cap = profile.get("marketCapitalization")
            try:
                market_cap = float(profile_cap) * 1_000_000 if profile_cap is not None else None
            except (TypeError, ValueError):
                market_cap = None
        else:
            market_cap = market_cap * 1_000_000

        row = FinancialMetrics(
            ticker=symbol,
            report_period=report_period,
            period=period,
            currency=str(profile.get("currency", "USD") or "USD"),
            market=market,
            market_cap=market_cap,
            price_to_earnings_ratio=self._first_metric(metric, ["peTTM", "peBasicExclExtraTTM", "peAnnual"]),
            price_to_book_ratio=self._first_metric(metric, ["pbQuarterly", "pb", "pbAnnual"]),
            price_to_sales_ratio=self._first_metric(metric, ["psTTM", "psAnnual"]),
            gross_margin=self._first_metric(metric, ["grossMarginTTM", "grossMarginAnnual"]),
            operating_margin=self._first_metric(metric, ["operatingMarginTTM", "operatingMarginAnnual"]),
            net_margin=self._first_metric(metric, ["netProfitMarginTTM", "netProfitMarginAnnual"]),
            return_on_equity=self._first_metric(metric, ["roeTTM", "roeRfy"]),
            return_on_assets=self._first_metric(metric, ["roaTTM", "roaRfy"]),
            current_ratio=self._first_metric(metric, ["currentRatioQuarterly", "currentRatioAnnual"]),
            quick_ratio=self._first_metric(metric, ["quickRatioQuarterly", "quickRatioAnnual"]),
            debt_to_equity=self._first_metric(
                metric, ["totalDebt/totalEquityQuarterly", "totalDebt/totalEquityAnnual"]
            ),
            revenue_growth=self._first_metric(metric, ["revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy"]),
            earnings_growth=self._first_metric(metric, ["epsGrowthTTMYoy", "epsGrowthQuarterlyYoy"]),
            earnings_per_share=self._first_metric(metric, ["epsBasicExclExtraItemsTTM", "epsAnnual"]),
            book_value_per_share=self._first_metric(
                metric, ["bookValuePerShareQuarterly", "bookValuePerShareAnnual"]
            ),
            free_cash_flow_per_share=self._first_metric(metric, ["cashFlowPerShareTTM", "cashFlowPerShareAnnual"]),
            enterprise_value=(enterprise_value * 1_000_000) if enterprise_value is not None else None,
        )

        return [row][: max(1, limit)]

    def search_line_items(self, ticker: str, period: str = "ttm", limit: int = 10) -> list[LineItem]:
        return self._fallback.search_line_items(ticker, period, limit)

    def get_company_news(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ) -> list[CompanyNews]:
        key = self._require_key()
        symbol = normalize_ticker(ticker)
        start = pd_to_datetime(start_date, floor=True).strftime("%Y-%m-%d")
        end = pd_to_datetime(end_date, floor=False).strftime("%Y-%m-%d")
        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": symbol, "from": start, "to": end, "token": key},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return []
        market = detect_market(symbol)
        out: list[CompanyNews] = []
        for item in payload[:limit]:
            try:
                ts = int(item.get("datetime", 0))
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
                out.append(
                    CompanyNews(
                        ticker=symbol,
                        title=str(item.get("headline", "")),
                        source=str(item.get("source", "")),
                        date=date_str,
                        url=str(item.get("url", "")),
                        market=market,
                        content=str(item.get("summary", "")) or None,
                    )
                )
            except Exception:
                continue
        return out

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        return self._fallback.get_insider_trades(ticker, end_date, start_date, limit)

    def get_market_cap(self, ticker: str, end_date: str | None = None) -> float | None:
        key = self._require_key()
        symbol = normalize_ticker(ticker)
        resp = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": key},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        # Finnhub quote does not contain market cap. Fallback to AkShare to keep behavior.
        _ = payload
        return self._fallback.get_market_cap(ticker, end_date)

    def get_macro_indicators(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        return self._fallback.get_macro_indicators(start_date, end_date)


def pd_to_datetime(value: str | None, floor: bool) -> datetime:
    if value:
        try:
            dt = datetime.strptime(value, "%Y-%m-%d")
            return dt.replace(hour=0, minute=0, second=0) if floor else dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            pass
    now = datetime.utcnow()
    if floor:
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    return now.replace(hour=23, minute=59, second=59, microsecond=0)

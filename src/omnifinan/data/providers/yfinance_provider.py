"""Yahoo Finance backed provider implementation."""

from __future__ import annotations

import pandas as pd

from ...data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from ...unified_api import detect_market, normalize_ticker
from ..symbols import is_crypto_ticker
from .akshare_provider import AkshareProvider
from .base import DataProvider


class YFinanceProvider(DataProvider):
    def __init__(self):
        self._fallback = AkshareProvider()

    def _import_yf(self):
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency optional
            raise RuntimeError("yfinance not installed. Run `pip install yfinance`.") from exc
        return yf

    def get_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> list[Price]:
        is_crypto = is_crypto_ticker(ticker)
        symbol = ticker.strip().upper().replace("/", "-") if is_crypto else normalize_ticker(ticker)
        market = detect_market(symbol)
        if not is_crypto and market.name != "US":
            return self._fallback.get_prices(ticker, start_date, end_date, interval=interval)

        yf = self._import_yf()
        yf_interval_map = {
            "1d": "1d",
            "1m": "1m",
            "3m": "1m",  # resample later
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "60m": "60m",
        }
        use_interval = yf_interval_map.get(interval, "1d")

        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=use_interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return []

        # yfinance may return MultiIndex columns like ("Close", "MSFT").
        if isinstance(df.columns, pd.MultiIndex):
            flattened = []
            for col in df.columns:
                if isinstance(col, tuple) and col:
                    flattened.append(str(col[0]))
                else:
                    flattened.append(str(col))
            df.columns = flattened

        df = df.reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "time"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "time"})
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        if "time" not in df.columns:
            return []

        if interval == "3m":
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"]).set_index("time")
            rs = df.resample("3min").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            df = rs.dropna(subset=["open", "high", "low", "close"]).reset_index()

        out: list[Price] = []
        intraday = interval != "1d"
        for _, row in df.iterrows():
            dt = pd.to_datetime(row["time"], errors="coerce")
            if pd.isna(dt):
                continue
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S") if intraday else dt.strftime("%Y-%m-%d")
            close = float(pd.to_numeric(row.get("close", 0.0), errors="coerce") or 0.0)
            volume = int(float(pd.to_numeric(row.get("volume", 0.0), errors="coerce") or 0.0))
            out.append(
                Price(
                    open=float(pd.to_numeric(row.get("open", close), errors="coerce") or close),
                    high=float(pd.to_numeric(row.get("high", close), errors="coerce") or close),
                    low=float(pd.to_numeric(row.get("low", close), errors="coerce") or close),
                    close=close,
                    volume=max(volume, 0),
                    amount=float(volume) * close,
                    time=time_str,
                    market=None if is_crypto else market,
                )
            )
        return out

    def get_financial_metrics(
        self, ticker: str, end_date: str | None = None, period: str = "ttm", limit: int = 1
    ) -> list[FinancialMetrics]:
        if is_crypto_ticker(ticker):
            return []
        return self._fallback.get_financial_metrics(ticker, end_date, period, limit)

    def search_line_items(self, ticker: str, period: str = "ttm", limit: int = 10) -> list[LineItem]:
        if is_crypto_ticker(ticker):
            return []
        return self._fallback.search_line_items(ticker, period, limit)

    def get_company_news(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ) -> list[CompanyNews]:
        if is_crypto_ticker(ticker):
            return []
        return self._fallback.get_company_news(ticker, start_date, end_date, limit)

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        if is_crypto_ticker(ticker):
            return []
        return self._fallback.get_insider_trades(ticker, end_date, start_date, limit)

    def get_market_cap(self, ticker: str, end_date: str | None = None) -> float | None:
        if is_crypto_ticker(ticker):
            return None
        return self._fallback.get_market_cap(ticker, end_date)

    def get_macro_indicators(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        return self._fallback.get_macro_indicators(start_date, end_date)

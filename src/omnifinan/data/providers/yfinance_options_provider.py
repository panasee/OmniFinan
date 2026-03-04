"""YFinance-backed provider for stock option chain data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd


class YFinanceOptionsProvider:
    def __init__(self):
        pass

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        s = str(symbol or "").strip().upper()
        aliases = {
            ".SPX": "^SPX",
        }
        return aliases.get(s, s)

    def _import_yf(self):
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("yfinance not installed. Run `pip install yfinance`.") from exc
        return yf

    @staticmethod
    def _previous_business_day_utc(today_utc: datetime) -> str:
        d = today_utc.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
        return d.strftime("%Y-%m-%d")

    @staticmethod
    def _to_epoch(value: Any) -> int | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return int(ts.timestamp())
        except Exception:
            return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _to_bool(value: Any) -> bool | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return bool(value)

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
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = extra_params
        yf = self._import_yf()
        normalized_symbol = self._normalize_symbol(symbol)
        ticker = yf.Ticker(normalized_symbol)

        expiries = list(getattr(ticker, "options", []) or [])
        if expiration:
            expiries = [d for d in expiries if d == expiration]

        target_snapshot_date = snapshot_date
        if snapshot_mode == "prev_close" and not target_snapshot_date:
            target_snapshot_date = self._previous_business_day_utc(datetime.now(timezone.utc))

        rows: list[dict[str, Any]] = []
        for exp in expiries:
            dte = None
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                dte = max((exp_dt.date() - datetime.now(timezone.utc).date()).days, 0)
            except Exception:
                pass

            if min_dte is not None and dte is not None and dte < min_dte:
                continue
            if max_dte is not None and dte is not None and dte > max_dte:
                continue

            chain = ticker.option_chain(exp)
            pairs = [("call", chain.calls), ("put", chain.puts)]
            if option_type in {"call", "put"}:
                pairs = [(option_type, chain.calls if option_type == "call" else chain.puts)]

            for side, df in pairs:
                if df is None or df.empty:
                    continue
                for _, r in df.iterrows():
                    row_strike = self._to_float(r.get("strike"))
                    if strike is not None and row_strike is not None and abs(row_strike - strike) > 1e-9:
                        continue
                    ts = self._to_epoch(r.get("lastTradeDate"))
                    if snapshot_mode == "prev_close" and target_snapshot_date and ts is not None:
                        snap_cutoff = pd.Timestamp(f"{target_snapshot_date}T23:59:59Z")
                        if pd.to_datetime(ts, unit="s", utc=True) > snap_cutoff:
                            continue
                    rows.append(
                        {
                            "optionSymbol": r.get("contractSymbol"),
                            "underlying": normalized_symbol,
                            "expiration": self._to_epoch(exp),
                            "side": side,
                            "strike": row_strike,
                            "dte": dte,
                            "updated": ts,
                            "bid": self._to_float(r.get("bid")),
                            "ask": self._to_float(r.get("ask")),
                            "mid": None,
                            "last": self._to_float(r.get("lastPrice")),
                            "openInterest": self._to_float(r.get("openInterest")),
                            "volume": self._to_float(r.get("volume")),
                            "inTheMoney": self._to_bool(r.get("inTheMoney")),
                            "iv": self._to_float(r.get("impliedVolatility")),
                            "delta": None,
                            "gamma": None,
                            "theta": None,
                            "vega": None,
                        }
                    )

        for row in rows:
            bid = row.get("bid")
            ask = row.get("ask")
            if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
                row["mid"] = (float(bid) + float(ask)) / 2.0

        return {
            "meta": {
                "source": "yfinance",
                "asset_kind": "stock_option",
                "symbol": normalized_symbol,
                "requested_symbol": str(symbol).strip().upper(),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_mode": snapshot_mode,
                "snapshot_date": target_snapshot_date,
                "filters": {
                    "expiration": expiration,
                    "type": option_type,
                    "strike": strike,
                    "min_dte": min_dte,
                    "max_dte": max_dte,
                },
            },
            "data": rows,
            "raw": {"expirations": expiries},
        }

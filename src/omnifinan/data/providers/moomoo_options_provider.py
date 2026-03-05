"""Moomoo-backed provider for stock option chain data."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd


class MoomooOptionsProvider:
    """Fetch stock option chains from moomoo OpenAPI.

    Notes:
    - Pull expiries first, then pull chain per-expiry to avoid the 30-day range limit.
    - Enrich dynamic fields (OI/volume/greeks) via market snapshot batches.
    """

    def __init__(self) -> None:
        self.default_host = os.getenv("MOOMOO_HOST", "172.22.128.1")
        self.default_port = int(os.getenv("MOOMOO_PORT", "12316"))
        self.snapshot_batch_size = int(os.getenv("MOOMOO_SNAPSHOT_BATCH", "200"))

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        s = str(symbol or "").strip().upper()
        aliases = {
            ".SPX": "US..SPX",
            "SPX": "US..SPX",
            "^SPX": "US..SPX",
            "SPXW": "US..SPX",
        }
        if s in aliases:
            return aliases[s]
        if s.startswith("US."):
            return s
        if s.startswith("US.."):
            return s
        return f"US.{s}"

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _to_epoch(value: Any) -> int | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return int(ts.timestamp())

    @staticmethod
    def _to_side(value: Any) -> str:
        raw = str(value or "").strip().upper()
        if raw == "CALL":
            return "call"
        if raw == "PUT":
            return "put"
        return raw.lower()

    def _get_connection(self, host: str, port: int):
        try:
            from moomoo import OpenQuoteContext  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("moomoo package not installed") from exc
        return OpenQuoteContext(host=host, port=port)

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
        from moomoo import OptionType, RET_OK  # type: ignore

        params = extra_params or {}
        host = str(params.get("host") or self.default_host)
        port = int(params.get("port") or self.default_port)
        normalized_symbol = self._normalize_symbol(symbol)

        ctx = self._get_connection(host=host, port=port)
        try:
            ret_exp, exp_df = ctx.get_option_expiration_date(code=normalized_symbol)
            if ret_exp != RET_OK or exp_df is None or exp_df.empty:
                return {
                    "meta": {
                        "source": "moomoo",
                        "asset_kind": "stock_option",
                        "symbol": normalized_symbol,
                        "requested_symbol": str(symbol).strip().upper(),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "error": f"get_option_expiration_date failed: ret={ret_exp}",
                    },
                    "data": [],
                    "raw": {"expirations": []},
                }

            expiries = sorted(exp_df["strike_time"].astype(str).tolist())
            if expiration:
                expiries = [d for d in expiries if d == expiration]

            chunks: list[pd.DataFrame] = []
            expiry_stats: list[dict[str, Any]] = []
            for exp in expiries:
                ret_chain = -1
                chain_df = None
                for _ in range(3):
                    ret_chain, chain_df = ctx.get_option_chain(
                        code=normalized_symbol,
                        start=exp,
                        end=exp,
                        option_type=OptionType.ALL,
                    )
                    if ret_chain == RET_OK and chain_df is not None:
                        break
                    time.sleep(0.15)
                rows = int(len(chain_df)) if ret_chain == RET_OK and chain_df is not None else 0
                expiry_stats.append({"expiry": exp, "ret": int(ret_chain), "rows": rows})
                if ret_chain == RET_OK and chain_df is not None and not chain_df.empty:
                    chunks.append(chain_df.copy())

            if not chunks:
                return {
                    "meta": {
                        "source": "moomoo",
                        "asset_kind": "stock_option",
                        "symbol": normalized_symbol,
                        "requested_symbol": str(symbol).strip().upper(),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "snapshot_mode": snapshot_mode,
                        "snapshot_date": snapshot_date,
                        "error": "no option-chain rows returned",
                    },
                    "data": [],
                    "raw": {"expirations": expiries, "expiry_stats": expiry_stats},
                }

            chain = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["code"])

            contract_codes = chain["code"].dropna().astype(str).tolist()
            snapshots: list[pd.DataFrame] = []
            for i in range(0, len(contract_codes), self.snapshot_batch_size):
                batch = contract_codes[i : i + self.snapshot_batch_size]
                ret_snap, snap_df = ctx.get_market_snapshot(batch)
                if ret_snap == RET_OK and snap_df is not None and not snap_df.empty:
                    snapshots.append(snap_df)

            if snapshots:
                snap = pd.concat(snapshots, ignore_index=True)
                keep = [
                    c
                    for c in [
                        "code",
                        "option_open_interest",
                        "volume",
                        "option_implied_volatility",
                        "option_delta",
                        "option_gamma",
                        "option_theta",
                        "option_vega",
                        "update_time",
                    ]
                    if c in snap.columns
                ]
                chain = chain.merge(snap[keep].drop_duplicates(subset=["code"]), on="code", how="left")

            now_utc = datetime.now(timezone.utc)
            rows: list[dict[str, Any]] = []
            for _, row in chain.iterrows():
                row_side = self._to_side(row.get("option_type"))
                if option_type in {"call", "put"} and row_side != option_type:
                    continue

                row_strike = self._to_float(row.get("strike_price"))
                if strike is not None and row_strike is not None and abs(row_strike - strike) > 1e-9:
                    continue

                exp_date = str(row.get("strike_time") or "")
                exp_dt = pd.to_datetime(exp_date, utc=True, errors="coerce")
                if pd.isna(exp_dt):
                    continue
                dte = max((exp_dt.date() - now_utc.date()).days, 0)
                if min_dte is not None and dte < min_dte:
                    continue
                if max_dte is not None and dte > max_dte:
                    continue

                bid = self._to_float(row.get("bid_price"))
                ask = self._to_float(row.get("ask_price"))
                mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None

                rows.append(
                    {
                        "optionSymbol": row.get("code"),
                        "underlying": normalized_symbol,
                        "expiration": self._to_epoch(exp_date),
                        "side": row_side,
                        "strike": row_strike,
                        "dte": dte,
                        "updated": self._to_epoch(row.get("update_time")),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "last": self._to_float(row.get("last_price")),
                        "openInterest": self._to_float(row.get("option_open_interest")),
                        "volume": self._to_float(row.get("volume")),
                        "inTheMoney": None,
                        "iv": self._to_float(row.get("option_implied_volatility")),
                        "delta": self._to_float(row.get("option_delta")),
                        "gamma": self._to_float(row.get("option_gamma")),
                        "theta": self._to_float(row.get("option_theta")),
                        "vega": self._to_float(row.get("option_vega")),
                    }
                )

            return {
                "meta": {
                    "source": "moomoo",
                    "asset_kind": "stock_option",
                    "symbol": normalized_symbol,
                    "requested_symbol": str(symbol).strip().upper(),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "snapshot_mode": snapshot_mode,
                    "snapshot_date": snapshot_date,
                    "filters": {
                        "expiration": expiration,
                        "type": option_type,
                        "strike": strike,
                        "min_dte": min_dte,
                        "max_dte": max_dte,
                    },
                },
                "data": rows,
                "raw": {"expirations": expiries, "expiry_stats": expiry_stats},
            }
        finally:
            ctx.close()

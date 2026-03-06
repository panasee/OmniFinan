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

    def _fetch_option_chain_once(
        self,
        *,
        ctx,
        symbol: str,
        expiry: str,
        option_cond_type: Any,
        index_option_type: Any,
    ):
        from moomoo import OptionType  # type: ignore

        return ctx.get_option_chain(
            code=symbol,
            index_option_type=index_option_type,
            start=expiry,
            end=expiry,
            option_type=OptionType.ALL,
            option_cond_type=option_cond_type,
        )

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
        from moomoo import IndexOptionType, OptionCondType, RET_OK  # type: ignore

        params = extra_params or {}
        host = str(params.get("host") or self.default_host)
        port = int(params.get("port") or self.default_port)
        normalized_symbol = self._normalize_symbol(symbol)

        # SPX: enforce NORMAL index-option type by default (empirically avoids ret=-1 on some expiries).
        index_option_type = IndexOptionType.NORMAL
        # Keep ALL as default; cond-type does not fix missing expiries when ret=-1.
        option_cond_type = OptionCondType.ALL

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

            # NOTE: Underlying spot/volume must be resolved via OmniFinan price interfaces,
            # not from the options provider. Keep underlying_price unset here.

            chunks: list[pd.DataFrame] = []
            expiry_stats: list[dict[str, Any]] = []
            # Empirical: ret=-1 is intermittent. Use retry + backoff + reconnect.
            reconnect_every = int(params.get("reconnect_every", 12))
            max_attempts = int(params.get("max_attempts", 6))
            base_sleep_s = float(params.get("base_sleep_s", 0.25))
            inter_expiry_sleep_s = float(params.get("inter_expiry_sleep_s", 0.2))

            for i, exp in enumerate(expiries):
                if reconnect_every > 0 and i > 0 and i % reconnect_every == 0:
                    try:
                        ctx.close()
                    except Exception:
                        pass
                    time.sleep(0.25)
                    ctx = self._get_connection(host=host, port=port)

                ret_chain = -1
                chain_df = None
                for attempt in range(max_attempts):
                    try:
                        ret_chain, chain_df = self._fetch_option_chain_once(
                            ctx=ctx,
                            symbol=normalized_symbol,
                            expiry=exp,
                            option_cond_type=option_cond_type,
                            index_option_type=index_option_type,
                        )
                    except Exception:
                        ret_chain, chain_df = -1, None

                    if ret_chain == RET_OK and chain_df is not None:
                        break

                    # exponential-ish backoff
                    time.sleep(base_sleep_s * (1.5 ** attempt))

                    # reconnect on later failures
                    if attempt in {2, 4}:
                        try:
                            ctx.close()
                        except Exception:
                            pass
                        time.sleep(0.25)
                        ctx = self._get_connection(host=host, port=port)

                rows = int(len(chain_df)) if ret_chain == RET_OK and chain_df is not None else 0
                expiry_stats.append({"expiry": exp, "ret": int(ret_chain), "rows": rows, "attempts": (attempt + 1)})
                if ret_chain == RET_OK and chain_df is not None and not chain_df.empty:
                    chunks.append(chain_df.copy())

                # Fixed pacing between expiries to reduce flow control / ret=-1 bursts.
                if inter_expiry_sleep_s > 0:
                    time.sleep(inter_expiry_sleep_s)

            # Second pass: rerun failed expiries with fresh sessions.
            # Empirical: this recovers many transient ret=-1 failures.
            failed_expiries = [s["expiry"] for s in expiry_stats if int(s.get("ret", -1)) != RET_OK]
            rerun_enabled = bool(params.get("rerun_failed_expiries", True))
            if rerun_enabled and failed_expiries:
                rerun_max_attempts = int(params.get("rerun_max_attempts", 10))
                rerun_base_sleep_s = float(params.get("rerun_base_sleep_s", 0.35))
                rerun_post_sleep_s = float(params.get("rerun_post_sleep_s", 0.4))

                for exp in failed_expiries:
                    # Fresh connection per expiry.
                    try:
                        ctx.close()
                    except Exception:
                        pass
                    time.sleep(0.25)
                    ctx = self._get_connection(host=host, port=port)

                    ret_chain = -1
                    chain_df = None
                    for attempt in range(rerun_max_attempts):
                        try:
                            ret_chain, chain_df = self._fetch_option_chain_once(
                                ctx=ctx,
                                symbol=normalized_symbol,
                                expiry=exp,
                                option_cond_type=option_cond_type,
                                index_option_type=index_option_type,
                            )
                        except Exception:
                            ret_chain, chain_df = -1, None

                        if ret_chain == RET_OK and chain_df is not None:
                            break
                        time.sleep(rerun_base_sleep_s * (1.7 ** attempt))

                    rows = int(len(chain_df)) if ret_chain == RET_OK and chain_df is not None else 0
                    expiry_stats.append(
                        {
                            "expiry": exp,
                            "ret": int(ret_chain),
                            "rows": rows,
                            "attempts": (attempt + 1),
                            "phase": "rerun_failed",
                        }
                    )
                    if ret_chain == RET_OK and chain_df is not None and not chain_df.empty:
                        chunks.append(chain_df.copy())

                    if rerun_post_sleep_s > 0:
                        time.sleep(rerun_post_sleep_s)

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

            def _snapshot_batches(code_list: list[str], batch_size: int) -> list[pd.DataFrame]:
                out_batches: list[pd.DataFrame] = []
                for j in range(0, len(code_list), batch_size):
                    batch = code_list[j : j + batch_size]
                    ret_snap, snap_df = ctx.get_market_snapshot(batch)
                    if ret_snap == RET_OK and snap_df is not None and not snap_df.empty:
                        out_batches.append(snap_df)
                return out_batches

            # Snapshot enrichment (dynamic fields). Empirical: can return partial coverage.
            snapshots: list[pd.DataFrame] = _snapshot_batches(contract_codes, self.snapshot_batch_size)

            # Retry snapshot for missing codes with smaller batches + backoff.
            snapshot_retry_rounds = int(params.get("snapshot_retry_rounds", 2))
            snapshot_retry_batch = int(params.get("snapshot_retry_batch", 50))
            snapshot_retry_sleep_s = float(params.get("snapshot_retry_sleep_s", 0.6))

            if snapshots and snapshot_retry_rounds > 0:
                got = pd.concat(snapshots, ignore_index=True)
                got_codes = set(got["code"].dropna().astype(str).tolist()) if "code" in got.columns else set()
                missing = [c for c in contract_codes if c not in got_codes]

                for r in range(snapshot_retry_rounds):
                    if not missing:
                        break
                    # Backoff between rounds.
                    time.sleep(snapshot_retry_sleep_s * (1.4 ** r))

                    # Refresh connection defensively.
                    try:
                        ctx.close()
                    except Exception:
                        pass
                    time.sleep(0.25)
                    ctx = self._get_connection(host=host, port=port)

                    new_batches = _snapshot_batches(missing, snapshot_retry_batch)
                    if not new_batches:
                        continue
                    snapshots.extend(new_batches)
                    got = pd.concat(snapshots, ignore_index=True)
                    got_codes = set(got["code"].dropna().astype(str).tolist()) if "code" in got.columns else set()
                    missing = [c for c in contract_codes if c not in got_codes]

            if snapshots:
                snap = pd.concat(snapshots, ignore_index=True)
                keep = [
                    c
                    for c in [
                        "code",
                        "bid_price",
                        "ask_price",
                        "last_price",
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
                last = self._to_float(row.get("last_price"))

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
                        "last": last,
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

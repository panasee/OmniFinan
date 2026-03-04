"""MarketData-backed provider for options chain data only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .credentials import get_api_key, load_provider_credentials


class MarketDataOptionsProvider:
    """Fetch stock/futures option chains from MarketData endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 20,
    ):
        cfg = load_provider_credentials()
        node = cfg.get("marketdata") if isinstance(cfg.get("marketdata"), dict) else {}
        env_base = node.get("base_url") if isinstance(node, dict) else None
        key = api_key or get_api_key("marketdata")
        self.api_key = key.strip() if isinstance(key, str) and key.strip() else None
        self.base_url = str(base_url or env_base or "https://api.marketdata.app/v1").rstrip("/")
        self.timeout = timeout

    def _require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("MarketData API key missing. Configure OMNIX_PATH/finn_api.json")
        return self.api_key

    @staticmethod
    def _coerce_rows(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if not isinstance(payload, dict):
            return []

        for key in ("data", "results", "chain", "options", "optionChain"):
            val = payload.get(key)
            if isinstance(val, list):
                return [row for row in val if isinstance(row, dict)]

        # Some vendors return columnar payloads: {field_a:[...], field_b:[...]}
        list_fields: dict[str, list[Any]] = {
            k: v for k, v in payload.items() if isinstance(k, str) and isinstance(v, list)
        }
        if not list_fields:
            return []
        row_count = max((len(v) for v in list_fields.values()), default=0)
        rows: list[dict[str, Any]] = []
        for idx in range(row_count):
            row: dict[str, Any] = {}
            for field, values in list_fields.items():
                row[field] = values[idx] if idx < len(values) else None
            rows.append(row)
        return rows

    def _request_chain(
        self,
        *,
        symbol: str,
        asset_kind: str,
        expiration: str | None,
        option_type: str | None,
        strike: float | None,
        min_dte: int | None,
        max_dte: int | None,
        snapshot_mode: str,
        snapshot_date: str | None,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        key = self._require_key()
        clean_symbol = str(symbol or "").strip().upper()
        if not clean_symbol:
            raise ValueError("symbol is required")

        endpoint = f"{self.base_url}/options/chain/{clean_symbol}/"
        params: dict[str, Any] = {"token": key}
        if expiration:
            params["expiration"] = expiration
        if option_type:
            params["type"] = option_type.lower()
        if strike is not None:
            params["strike"] = strike
        if min_dte is not None:
            params["min_dte"] = int(min_dte)
        if max_dte is not None:
            params["max_dte"] = int(max_dte)
        if snapshot_date:
            # Best-effort server-side date filter; unsupported providers may ignore.
            params["date"] = snapshot_date
        if extra_params:
            for k, v in extra_params.items():
                if v is not None:
                    params[str(k)] = v

        resp = requests.get(endpoint, params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload: Any = resp.json()
        rows = self._coerce_rows(payload)
        filtered_rows, effective_snapshot_date = self._apply_snapshot_filter(
            rows=rows,
            snapshot_mode=snapshot_mode,
            snapshot_date=snapshot_date,
        )
        return {
            "meta": {
                "source": "marketdata",
                "asset_kind": asset_kind,
                "symbol": clean_symbol,
                "endpoint": endpoint,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_mode": snapshot_mode,
                "snapshot_date": effective_snapshot_date,
                "filters": {
                    "expiration": expiration,
                    "type": option_type.lower() if isinstance(option_type, str) else None,
                    "strike": strike,
                    "min_dte": min_dte,
                    "max_dte": max_dte,
                },
            },
            "data": filtered_rows,
            "raw": payload if isinstance(payload, dict) else {"payload": payload},
        }

    @staticmethod
    def _previous_business_day_utc(today_utc: datetime) -> str:
        d = today_utc.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
        return d.strftime("%Y-%m-%d")

    def _apply_snapshot_filter(
        self,
        *,
        rows: list[dict[str, Any]],
        snapshot_mode: str,
        snapshot_date: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if snapshot_mode != "prev_close":
            return rows, snapshot_date

        target_date = snapshot_date or self._previous_business_day_utc(datetime.now(timezone.utc))
        cutoff = f"{target_date}T23:59:59+00:00"
        try:
            cutoff_dt = datetime.fromisoformat(cutoff)
        except Exception:
            return rows, target_date

        filtered: list[dict[str, Any]] = []
        for row in rows:
            updated = row.get("updated")
            if not isinstance(updated, (int, float)):
                continue
            ts = datetime.fromtimestamp(int(updated), tz=timezone.utc)
            if ts <= cutoff_dt:
                filtered.append(row)

        # Fallback to original rows when historical snapshot is unavailable.
        if not filtered:
            return rows, target_date
        return filtered, target_date

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
        return self._request_chain(
            symbol=symbol,
            asset_kind="stock_option",
            expiration=expiration,
            option_type=option_type,
            strike=strike,
            min_dte=min_dte,
            max_dte=max_dte,
            snapshot_mode=snapshot_mode,
            snapshot_date=snapshot_date,
            extra_params=extra_params,
        )

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
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = dict(extra_params or {})
        params.setdefault("asset", "futures")
        return self._request_chain(
            symbol=symbol,
            asset_kind="futures_option",
            expiration=expiration,
            option_type=option_type,
            strike=strike,
            min_dte=min_dte,
            max_dte=max_dte,
            snapshot_mode=snapshot_mode,
            snapshot_date=snapshot_date,
            extra_params=params,
        )

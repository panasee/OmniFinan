"""Option analytics helpers (non-LLM, deterministic)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _parse_option_type(row: dict[str, Any]) -> str | None:
    for key in ("side", "type", "option_type", "right"):
        raw = str(row.get(key, "")).strip().lower()
        if raw in {"call", "c"}:
            return "call"
        if raw in {"put", "p"}:
            return "put"
    symbol = str(row.get("optionSymbol", "")).upper()
    if symbol:
        if "C" in symbol[-15:]:
            return "call"
        if "P" in symbol[-15:]:
            return "put"
    return None


def _parse_snapshot_datetime(snapshot_date: str | None) -> datetime:
    if snapshot_date:
        try:
            return datetime.strptime(snapshot_date[:10], "%Y-%m-%d").replace(tzinfo=UTC)
        except Exception:
            pass
    return datetime.now(UTC)


def _parse_expiry_days(row: dict[str, Any], snapshot_dt: datetime) -> float | None:
    dte = _safe_float(row.get("dte"))
    if dte is not None and dte >= 0:
        return dte

    expiry_raw = row.get("expiration")
    if expiry_raw is None:
        return None

    expiry_dt: datetime | None = None
    if isinstance(expiry_raw, (int, float)):
        value = float(expiry_raw)
        if value > 1e12:
            value = value / 1000.0
        try:
            expiry_dt = datetime.fromtimestamp(value, tz=UTC)
        except Exception:
            expiry_dt = None
    elif isinstance(expiry_raw, str):
        text = expiry_raw.strip()
        if text:
            try:
                expiry_dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except Exception:
                for fmt in ("%Y-%m-%d", "%Y%m%d"):
                    try:
                        expiry_dt = datetime.strptime(text[:10], fmt).replace(tzinfo=UTC)
                        break
                    except Exception:
                        continue
    if expiry_dt is None:
        return None
    return max((expiry_dt - snapshot_dt).total_seconds() / 86400.0, 0.0)


def _price_from_row(row: dict[str, Any]) -> float | None:
    for key in ("mid", "mark", "last", "price"):
        v = _safe_float(row.get(key))
        if v is not None and v > 0:
            return v
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return None


def _extract_close_prices(price_history: list[dict[str, Any]] | None) -> list[float]:
    if not isinstance(price_history, list):
        return []
    closes: list[float] = []
    for row in price_history:
        if not isinstance(row, dict):
            continue
        c = _safe_float(row.get("close"))
        if c is not None and c > 0:
            closes.append(c)
    return closes


def _compute_realized_vol(price_history: list[dict[str, Any]] | None, lookback_days: int = 20) -> float | None:
    closes = _extract_close_prices(price_history)
    if len(closes) < max(lookback_days, 2):
        return None
    closes = closes[-(lookback_days + 1) :]
    if len(closes) < 2:
        return None
    rets: list[float] = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            continue
        rets.append(math.log(closes[i] / closes[i - 1]))
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    if var < 0:
        return None
    return math.sqrt(var) * math.sqrt(252.0)


def _percentile_rank(current: float, history: list[float]) -> float | None:
    clean = sorted(v for v in history if isinstance(v, (int, float)) and not math.isnan(float(v)))
    if not clean:
        return None
    count = sum(1 for v in clean if float(v) <= current)
    return 100.0 * count / len(clean)


def _top_levels(oi_map: dict[float, float], k: int = 3) -> list[dict[str, float]]:
    ranked = sorted(oi_map.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"strike": float(strike), "open_interest": float(oi)} for strike, oi in ranked]


def _compute_max_pain_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_expiry: dict[str, dict[str, dict[float, float]]] = {}
    global_strikes: set[float] = set()
    for row in rows:
        expiry = str(row.get("expiration", "unknown"))
        option_type = _parse_option_type(row)
        strike = _safe_float(row.get("strike"))
        oi = _safe_float(row.get("openInterest"))
        if option_type is None or strike is None or oi is None or oi <= 0:
            continue
        book = per_expiry.setdefault(expiry, {"call": {}, "put": {}})
        bucket = book["call"] if option_type == "call" else book["put"]
        bucket[strike] = bucket.get(strike, 0.0) + oi
        global_strikes.add(strike)

    def _pain_for_target(target: float, calls: dict[float, float], puts: dict[float, float]) -> float:
        pain = 0.0
        for strike, oi in calls.items():
            pain += max(0.0, target - strike) * oi
        for strike, oi in puts.items():
            pain += max(0.0, strike - target) * oi
        return pain

    by_expiry: list[dict[str, Any]] = []
    merged_calls: dict[float, float] = {}
    merged_puts: dict[float, float] = {}
    for expiry, book in per_expiry.items():
        calls = book["call"]
        puts = book["put"]
        if not calls and not puts:
            continue
        targets = sorted(set(calls.keys()) | set(puts.keys()))
        if not targets:
            continue
        pains = [{"strike": t, "pain": _pain_for_target(t, calls, puts)} for t in targets]
        best = min(pains, key=lambda x: x["pain"])
        by_expiry.append(
            {
                "expiration": expiry,
                "max_pain_strike": float(best["strike"]),
                "min_total_pain": float(best["pain"]),
            }
        )
        for k, v in calls.items():
            merged_calls[k] = merged_calls.get(k, 0.0) + v
        for k, v in puts.items():
            merged_puts[k] = merged_puts.get(k, 0.0) + v

    overall_targets = sorted(set(merged_calls.keys()) | set(merged_puts.keys()) | global_strikes)
    overall = None
    if overall_targets:
        pains = [{"strike": t, "pain": _pain_for_target(t, merged_calls, merged_puts)} for t in overall_targets]
        best = min(pains, key=lambda x: x["pain"])
        overall = {"max_pain_strike": float(best["strike"]), "min_total_pain": float(best["pain"])}
    return {"overall": overall, "by_expiry": sorted(by_expiry, key=lambda x: str(x["expiration"]))}


def _compute_oi_levels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    call_oi: dict[float, float] = {}
    put_oi: dict[float, float] = {}
    for row in rows:
        option_type = _parse_option_type(row)
        strike = _safe_float(row.get("strike"))
        oi = _safe_float(row.get("openInterest"))
        if option_type is None or strike is None or oi is None or oi <= 0:
            continue
        if option_type == "call":
            call_oi[strike] = call_oi.get(strike, 0.0) + oi
        else:
            put_oi[strike] = put_oi.get(strike, 0.0) + oi
    total_call = sum(call_oi.values())
    total_put = sum(put_oi.values())
    return {
        "call_walls": _top_levels(call_oi, k=3),
        "put_walls": _top_levels(put_oi, k=3),
        "primary_resistance": _top_levels(call_oi, k=1)[0] if call_oi else None,
        "primary_support": _top_levels(put_oi, k=1)[0] if put_oi else None,
        "put_call_oi_ratio": (total_put / total_call) if total_call > 0 else None,
    }


def bs_price(
    spot: float,
    strike: float,
    ttm: float,
    rate: float,
    vol: float,
    option_type: str,
    dividend: float = 0.0,
) -> float:
    if spot <= 0 or strike <= 0 or ttm <= 0 or vol <= 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        return intrinsic
    sqt = math.sqrt(ttm)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * ttm) / (vol * sqt)
    d2 = d1 - vol * sqt
    if option_type == "call":
        return spot * math.exp(-dividend * ttm) * _norm_cdf(d1) - strike * math.exp(-rate * ttm) * _norm_cdf(d2)
    return strike * math.exp(-rate * ttm) * _norm_cdf(-d2) - spot * math.exp(-dividend * ttm) * _norm_cdf(-d1)


def bs_delta(
    spot: float,
    strike: float,
    ttm: float,
    rate: float,
    vol: float,
    option_type: str,
    dividend: float = 0.0,
) -> float:
    if spot <= 0 or strike <= 0 or ttm <= 0 or vol <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * ttm) / (vol * math.sqrt(ttm))
    if option_type == "call":
        return math.exp(-dividend * ttm) * _norm_cdf(d1)
    return -math.exp(-dividend * ttm) * _norm_cdf(-d1)


def bs_vega(
    spot: float,
    strike: float,
    ttm: float,
    rate: float,
    vol: float,
    dividend: float = 0.0,
) -> float:
    if spot <= 0 or strike <= 0 or ttm <= 0 or vol <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * ttm) / (vol * math.sqrt(ttm))
    return spot * math.exp(-dividend * ttm) * math.sqrt(ttm) * _norm_pdf(d1)


def compute_implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    ttm: float,
    rate: float,
    option_type: str,
    dividend: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 120,
) -> float | None:
    if option_price <= 0 or spot <= 0 or strike <= 0 or ttm <= 0:
        return None
    low, high = 1e-4, 5.0
    price_low = bs_price(spot, strike, ttm, rate, low, option_type, dividend)
    price_high = bs_price(spot, strike, ttm, rate, high, option_type, dividend)
    if option_price < price_low - 1e-8 or option_price > price_high + 1e-8:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        p = bs_price(spot, strike, ttm, rate, mid, option_type, dividend)
        if abs(p - option_price) <= tol:
            return mid
        if p > option_price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def compute_greeks(
    spot: float,
    strike: float,
    ttm: float,
    rate: float,
    vol: float,
    option_type: str,
    dividend: float = 0.0,
) -> dict[str, float]:
    if spot <= 0 or strike <= 0 or ttm <= 0 or vol <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    sqt = math.sqrt(ttm)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * ttm) / (vol * sqt)
    d2 = d1 - vol * sqt
    pdf = _norm_pdf(d1)
    disc_r = math.exp(-rate * ttm)
    disc_q = math.exp(-dividend * ttm)

    gamma = disc_q * pdf / (spot * vol * sqt)
    vega = spot * disc_q * sqt * pdf
    if option_type == "call":
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -(spot * disc_q * pdf * vol) / (2.0 * sqt)
            - rate * strike * disc_r * _norm_cdf(d2)
            + dividend * spot * disc_q * _norm_cdf(d1)
        )
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = (
            -(spot * disc_q * pdf * vol) / (2.0 * sqt)
            + rate * strike * disc_r * _norm_cdf(-d2)
            - dividend * spot * disc_q * _norm_cdf(-d1)
        )
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def compute_chain_analytics(
    rows: list[dict[str, Any]],
    *,
    underlying_price: float | None = None,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    snapshot_date: str | None = None,
    price_history: list[dict[str, Any]] | None = None,
    iv_history: list[float] | None = None,
    hv_lookback_days: int = 20,
) -> dict[str, Any]:
    snapshot_dt = _parse_snapshot_datetime(snapshot_date)

    spot = _safe_float(underlying_price)
    if spot is None:
        for row in rows:
            spot = _safe_float(row.get("underlyingPrice"))
            if spot is not None and spot > 0:
                break
    if spot is None or spot <= 0:
        strikes = sorted(
            s for s in (_safe_float(row.get("strike")) for row in rows) if s is not None and s > 0
        )
        if strikes:
            spot = strikes[len(strikes) // 2]
    if spot is None or spot <= 0:
        return {
            "summary": {"option_count": len(rows), "underlying_price": None},
            "surface": [],
            "term_structure": [],
            "skew_by_expiry": [],
            "smile_by_expiry": [],
            "max_pain": {"overall": None, "by_expiry": []},
            "levels": {},
            "implied_vs_realized": {},
            "errors": ["Unable to infer underlying price from chain."],
        }

    enriched: list[dict[str, Any]] = []
    for row in rows:
        strike = _safe_float(row.get("strike"))
        option_type = _parse_option_type(row)
        dte_days = _parse_expiry_days(row, snapshot_dt)
        market_price = _price_from_row(row)
        if strike is None or strike <= 0 or option_type is None or dte_days is None:
            continue
        ttm = max(dte_days / 365.0, 1e-6)

        iv = _safe_float(row.get("iv"))
        if iv is None or iv <= 0:
            if market_price is not None:
                iv = compute_implied_volatility(
                    market_price,
                    spot=spot,
                    strike=strike,
                    ttm=ttm,
                    rate=risk_free_rate,
                    option_type=option_type,
                    dividend=dividend_yield,
                )
        if iv is None or iv <= 0:
            continue
        greeks = compute_greeks(
            spot=spot,
            strike=strike,
            ttm=ttm,
            rate=risk_free_rate,
            vol=iv,
            option_type=option_type,
            dividend=dividend_yield,
        )
        delta = _safe_float(row.get("delta"))
        if delta is None:
            delta = greeks["delta"]

        expiry_key = str(row.get("expiration", "unknown"))
        oi = _safe_float(row.get("openInterest")) or 0.0
        enriched.append(
            {
                "expiration": expiry_key,
                "dte": float(dte_days),
                "type": option_type,
                "strike": float(strike),
                "moneyness": float(strike / spot),
                "iv": float(iv),
                "delta": float(delta),
                "gamma": float(greeks["gamma"]),
                "theta": float(greeks["theta"]),
                "vega": float(greeks["vega"]),
                "open_interest": float(oi),
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in enriched:
        groups.setdefault(str(row["expiration"]), []).append(row)

    skew_by_expiry: list[dict[str, Any]] = []
    term_structure: list[dict[str, Any]] = []
    smile_by_expiry: list[dict[str, Any]] = []
    nearest_atm_iv: float | None = None
    nearest_dte: float | None = None
    for expiry, items in sorted(groups.items(), key=lambda kv: min(x["dte"] for x in kv[1])):
        calls = [x for x in items if x["type"] == "call"]
        puts = [x for x in items if x["type"] == "put"]
        atm = min(items, key=lambda x: abs(x["strike"] - spot))
        atm_iv = float(atm["iv"])

        call25 = min(calls, key=lambda x: abs(x["delta"] - 0.25))["iv"] if calls else None
        put25 = min(puts, key=lambda x: abs(x["delta"] + 0.25))["iv"] if puts else None
        rr25 = (call25 - put25) if call25 is not None and put25 is not None else None
        bf25 = (0.5 * (call25 + put25) - atm_iv) if call25 is not None and put25 is not None else None
        dte = float(min(x["dte"] for x in items))

        skew_by_expiry.append(
            {
                "expiration": expiry,
                "dte": dte,
                "atm_iv": atm_iv,
                "call_25d_iv": call25,
                "put_25d_iv": put25,
                "risk_reversal_25d": rr25,
                "butterfly_25d": bf25,
                "count": len(items),
            }
        )
        term_structure.append({"expiration": expiry, "dte": dte, "atm_iv": atm_iv, "count": len(items)})
        smile_points = sorted(
            [{"strike": x["strike"], "moneyness": x["moneyness"], "iv": x["iv"], "type": x["type"]} for x in items],
            key=lambda x: (x["strike"], x["type"]),
        )
        smile_by_expiry.append({"expiration": expiry, "dte": dte, "atm_iv": atm_iv, "points": smile_points})
        if nearest_dte is None or dte < nearest_dte:
            nearest_dte = dte
            nearest_atm_iv = atm_iv

    hv = _compute_realized_vol(price_history, lookback_days=hv_lookback_days)
    iv_hv = {
        "current_atm_iv": nearest_atm_iv,
        "historical_volatility": hv,
        "hv_lookback_days": int(hv_lookback_days),
        "iv_minus_hv": (nearest_atm_iv - hv) if nearest_atm_iv is not None and hv is not None else None,
        "iv_to_hv_ratio": (nearest_atm_iv / hv) if nearest_atm_iv is not None and hv and hv > 0 else None,
    }
    iv_percentile = None
    if nearest_atm_iv is not None and isinstance(iv_history, list):
        iv_percentile = _percentile_rank(nearest_atm_iv, iv_history)
    max_pain = _compute_max_pain_from_rows(rows)
    levels = _compute_oi_levels(rows)

    errors: list[str] = []
    if hv is None:
        errors.append("Insufficient price history for historical volatility.")
    if iv_percentile is None:
        errors.append("IV historical percentile unavailable (missing iv_history).")
    if max_pain.get("overall") is None:
        errors.append("Max pain unavailable (missing open interest).")
    if not levels.get("primary_support") or not levels.get("primary_resistance"):
        errors.append("Support/resistance unavailable (missing open interest by strike).")

    return {
        "summary": {
            "option_count": len(rows),
            "enriched_count": len(enriched),
            "underlying_price": float(spot),
            "median_iv": (
                sorted(x["iv"] for x in enriched)[len(enriched) // 2] if enriched else None
            ),
            "iv_historical_percentile": iv_percentile,
        },
        "surface": enriched,
        "term_structure": term_structure,
        "skew_by_expiry": skew_by_expiry,
        "smile_by_expiry": smile_by_expiry,
        "max_pain": max_pain,
        "levels": levels,
        "implied_vs_realized": iv_hv,
        "errors": errors,
    }

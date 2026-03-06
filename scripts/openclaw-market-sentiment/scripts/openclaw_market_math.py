"""Small math helpers for OpenClaw sentiment workflow demos.

Usage:
    python scripts/openclaw_market_math.py --mode demo
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime

import pandas as pd


def decayed_sentiment(
    sentiments: list[float],
    weights: list[float],
    ages_days: list[float],
    lambda_decay: float = 0.3,
) -> float:
    total = 0.0
    n = min(len(sentiments), len(weights), len(ages_days))
    for i in range(n):
        total += float(weights[i]) * float(sentiments[i]) * math.exp(-lambda_decay * float(ages_days[i]))
    return total


def centrality_weighted_decayed_sentiment(
    sentiments: list[float],
    weights: list[float],
    ages_days: list[float],
    centrality_multipliers: list[float],
    lambda_decay: float = 0.3,
    centrality_min: float = 0.5,
    centrality_max: float = 3.0,
) -> float:
    """KG-aware decayed sentiment with bounded centrality multiplier.

    Effective weight = weight_i * clip(centrality_i, centrality_min, centrality_max).
    """
    total = 0.0
    n = min(len(sentiments), len(weights), len(ages_days), len(centrality_multipliers))
    for i in range(n):
        c = min(max(float(centrality_multipliers[i]), centrality_min), centrality_max)
        w = float(weights[i]) * c
        total += w * float(sentiments[i]) * math.exp(-lambda_decay * float(ages_days[i]))
    return total


def event_multiplier_for_macro_event(event_tag: str | None) -> float:
    """Return event multiplier for lambda adaptation.

    Supported tags (case-insensitive):
    - nfp: 1.5
    - fomc: 1.6
    - cpi: 1.3
    - pce: 1.2
    - none/unknown: 1.0
    """
    tag = str(event_tag or "").strip().lower()
    table = {
        "nfp": 1.5,
        "fomc": 1.6,
        "cpi": 1.3,
        "pce": 1.2,
    }
    return table.get(tag, 1.0)


def adaptive_lambda(base_lambda: float, event_multiplier: float = 1.0) -> float:
    """Adaptive lambda for event windows (e.g., NFP/FOMC days)."""
    return max(0.01, float(base_lambda) * max(0.1, float(event_multiplier)))


def percentile_rank(current: float, history: list[float]) -> float | None:
    clean = sorted(float(x) for x in history if x is not None)
    if not clean:
        return None
    count = sum(1 for x in clean if x <= current)
    return 100.0 * count / len(clean)


def realized_vol_annualized(closes: list[float], periods_per_year: int = 252) -> float | None:
    if len(closes) < 3:
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
    return math.sqrt(max(var, 0.0)) * math.sqrt(float(periods_per_year))


def iv_hv_relation(
    current_iv: float,
    closes: list[float],
    *,
    iv_in_percent: bool = False,
    hv_periods_per_year: int = 252,
) -> dict:
    """IV-HV relation with explicit annualization alignment.

    If iv_in_percent=True, converts IV from percent to fraction.
    """
    iv = float(current_iv) / 100.0 if iv_in_percent else float(current_iv)
    hv = realized_vol_annualized(closes, periods_per_year=hv_periods_per_year)
    return {
        "current_iv": iv,
        "historical_volatility": hv,
        "iv_minus_hv": (iv - hv) if hv is not None else None,
        "iv_to_hv_ratio": (iv / hv) if hv not in (None, 0) else None,
    }


def vvix_vix_ratio(vvix: float | None, vix: float | None) -> float | None:
    if vvix in (None, 0) or vix in (None, 0):
        return None
    return float(vvix) / float(vix)


def gamma_regime(net_gex: float | None) -> str:
    if net_gex is None:
        return "unknown"
    return "positive_gamma" if net_gex >= 0 else "negative_gamma"


def pct_above_ma(closes_by_symbol: dict[str, list[float]], window: int = 200) -> float | None:
    if not closes_by_symbol:
        return None
    total = 0
    above = 0
    for _sym, series in closes_by_symbol.items():
        if not series or len(series) < window:
            continue
        ma = sum(series[-window:]) / window
        last = series[-1]
        total += 1
        if last > ma:
            above += 1
    if total == 0:
        return None
    return above / total


def term_structure_state(front: float | None, next_: float | None) -> str:
    if front is None or next_ is None:
        return "unknown"
    return "backwardation" if front > next_ else "contango"


def weighted_contributions(
    s_opt: float | None,
    s_news: float | None,
    s_breadth: float | None,
    w_opt: float = 0.45,
    w_news: float = 0.30,
    w_breadth: float = 0.25,
) -> dict[str, float | None]:
    return {
        "option_implied": (w_opt * float(s_opt)) if s_opt is not None else None,
        "news_diffusion": (w_news * float(s_news)) if s_news is not None else None,
        "breadth_flow": (w_breadth * float(s_breadth)) if s_breadth is not None else None,
    }


def top_driver(path_scores: dict[str, float | None]) -> str | None:
    valid = {k: abs(float(v)) for k, v in path_scores.items() if v is not None}
    if not valid:
        return None
    return max(valid, key=valid.get)


def data_cutoff_time_iso(cutoffs: list[str]) -> str | None:
    """Return the minimum cutoff timestamp in ISO format.

    Use earliest cutoff to avoid look-ahead bias across mixed feeds.
    """
    parsed = [pd.to_datetime(x, utc=True, errors="coerce") for x in cutoffs]
    parsed = [x for x in parsed if not pd.isna(x)]
    if not parsed:
        return None
    return min(parsed).strftime("%Y-%m-%dT%H:%M:%SZ")


def _demo() -> dict:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    base_lambda = 0.30
    lam = adaptive_lambda(base_lambda, event_multiplier=event_multiplier_for_macro_event("nfp"))
    sent = centrality_weighted_decayed_sentiment(
        sentiments=[-0.7, -0.4, 0.2, 0.1],
        weights=[1.0, 0.8, 0.5, 0.3],
        ages_days=[0.5, 1.0, 2.0, 3.0],
        centrality_multipliers=[2.0, 1.4, 0.8, 0.6],
        lambda_decay=lam,
    )
    iv_hist = [0.14, 0.16, 0.18, 0.20, 0.23, 0.27, 0.31]
    iv_now = 0.24
    iv_pct = percentile_rank(iv_now, iv_hist)
    ivhv = iv_hv_relation(iv_now, [100, 101, 100.5, 102, 103, 101.5, 102.2, 103.1, 102.7, 104.0])
    breadth = pct_above_ma(
        {
            "A": [100 + i * 0.1 for i in range(220)],
            "B": [100 - i * 0.05 for i in range(220)],
            "C": [50 + math.sin(i / 8.0) for i in range(220)],
        },
        window=200,
    )
    term = term_structure_state(front=21.4, next_=20.1)
    ratio = vvix_vix_ratio(vvix=92.0, vix=21.4)
    gamma = gamma_regime(net_gex=-1.2e9)
    contrib = weighted_contributions(s_opt=-0.42, s_news=-0.18, s_breadth=-0.09)
    driver = top_driver(contrib)
    cutoff = data_cutoff_time_iso([
        "2026-03-05T08:30:00Z",
        "2026-03-05T08:15:00Z",
        "2026-03-05T08:25:00Z",
    ])
    return {
        "as_of": now,
        "data_cutoff_time": cutoff,
        "news_diffusion_score": sent,
        "lambda_used": lam,
        "iv_percentile": iv_pct,
        "iv_hv": ivhv,
        "vvix_vix_ratio": ratio,
        "gamma_regime": gamma,
        "pct_above_ma200": breadth,
        "term_structure_state": term,
        "weighted_contributions": contrib,
        "top_driver": driver,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="demo", choices=["demo"])
    args = parser.parse_args()
    if args.mode == "demo":
        print(json.dumps(_demo(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

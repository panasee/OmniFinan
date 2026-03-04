"""Small math helpers for OpenClaw sentiment workflow demos.

Usage:
    python tests/openclaw_market_math.py --mode demo
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime


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


def percentile_rank(current: float, history: list[float]) -> float | None:
    clean = sorted(float(x) for x in history if x is not None)
    if not clean:
        return None
    count = sum(1 for x in clean if x <= current)
    return 100.0 * count / len(clean)


def realized_vol_annualized(closes: list[float]) -> float | None:
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
    return math.sqrt(max(var, 0.0)) * math.sqrt(252.0)


def iv_hv_relation(current_iv: float, closes: list[float]) -> dict:
    hv = realized_vol_annualized(closes)
    return {
        "current_iv": current_iv,
        "historical_volatility": hv,
        "iv_minus_hv": (current_iv - hv) if hv is not None else None,
        "iv_to_hv_ratio": (current_iv / hv) if hv not in (None, 0) else None,
    }


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


def _demo() -> dict:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sent = decayed_sentiment(
        sentiments=[-0.7, -0.4, 0.2, 0.1],
        weights=[1.0, 0.8, 0.5, 0.3],
        ages_days=[0.5, 1.0, 2.0, 3.0],
        lambda_decay=0.3,
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
    return {
        "as_of": now,
        "news_diffusion_score": sent,
        "iv_percentile": iv_pct,
        "iv_hv": ivhv,
        "pct_above_ma200": breadth,
        "term_structure_state": term,
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

"""Small technical helpers for OpenClaw stock analysis workflow demos.

Usage:
    python scripts/openclaw_stock_technicals.py --mode demo
"""

from __future__ import annotations

import argparse
import json


def simple_moving_average(closes: list[float], window: int) -> float | None:
    if window <= 0 or len(closes) < window:
        return None
    tail = [float(x) for x in closes[-window:]]
    return sum(tail) / window


def ema_series(closes: list[float], span: int) -> list[float]:
    if span <= 0 or not closes:
        return []
    alpha = 2.0 / (span + 1.0)
    out: list[float] = []
    ema_val = float(closes[0])
    out.append(ema_val)
    for price in closes[1:]:
        ema_val = alpha * float(price) + (1.0 - alpha) * ema_val
        out.append(ema_val)
    return out


def relative_strength_index(closes: list[float], period: int = 14) -> float | None:
    if period <= 0 or len(closes) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        change = float(closes[i]) - float(closes[i - 1])
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = ((period - 1) * avg_gain + gains[i]) / period
        avg_loss = ((period - 1) * avg_loss + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    closes: list[float],
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> dict[str, float | None]:
    if not closes or fast_span <= 0 or slow_span <= 0 or signal_span <= 0:
        return {"macd_line": None, "signal_line": None, "histogram": None}
    if len(closes) < max(fast_span, slow_span, signal_span):
        return {"macd_line": None, "signal_line": None, "histogram": None}

    fast = ema_series(closes, fast_span)
    slow = ema_series(closes, slow_span)
    macd_line_series = [f - s for f, s in zip(fast, slow)]
    signal_series = ema_series(macd_line_series, signal_span)
    if not macd_line_series or not signal_series:
        return {"macd_line": None, "signal_line": None, "histogram": None}

    macd_line = macd_line_series[-1]
    signal_line = signal_series[-1]
    histogram = macd_line - signal_line
    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
    }


def moving_average_pack(closes: list[float]) -> dict[str, float | None]:
    return {
        "ma20": simple_moving_average(closes, 20),
        "ma100": simple_moving_average(closes, 100),
        "ma200": simple_moving_average(closes, 200),
    }


def technical_snapshot(
    closes: list[float],
    *,
    rsi_period: int = 14,
    macd_fast_span: int = 12,
    macd_slow_span: int = 26,
    macd_signal_span: int = 9,
) -> dict:
    if not closes:
        return {
            "last_close": None,
            "moving_averages": moving_average_pack([]),
            "rsi": None,
            "macd": {"macd_line": None, "signal_line": None, "histogram": None},
        }
    return {
        "last_close": float(closes[-1]),
        "moving_averages": moving_average_pack(closes),
        "rsi": relative_strength_index(closes, period=rsi_period),
        "macd": macd(
            closes,
            fast_span=macd_fast_span,
            slow_span=macd_slow_span,
            signal_span=macd_signal_span,
        ),
    }


def _demo() -> dict:
    closes = [
        100.0 + i * 0.35 + ((-1) ** i) * 0.8
        for i in range(240)
    ]
    snap = technical_snapshot(closes)
    return {
        "last_close": snap["last_close"],
        "moving_averages": snap["moving_averages"],
        "rsi": snap["rsi"],
        "macd": snap["macd"],
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

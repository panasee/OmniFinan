import numpy as np
import pandas as pd

from omnifinan.analysis.factor_backtest import (
    build_cross_sectional_weights,
    perf_stats,
    run_daily_backtest,
)


def _mock_factor_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=40, freq="B")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    rows = []
    rng = np.random.default_rng(11)
    for d in dates:
        for i, s in enumerate(symbols):
            rows.append(
                {
                    "date": d,
                    "symbol": s,
                    "ret_1": float(rng.normal(0.0005, 0.01)),
                    "score": float(i + rng.normal(0, 0.2)),
                }
            )
    return pd.DataFrame(rows)


def test_build_cross_sectional_weights_has_targets():
    frame = _mock_factor_frame()
    w = build_cross_sectional_weights(
        frame,
        score_col="score",
        quantile=0.25,
        min_universe=8,
        long_short=True,
    )
    assert not w.empty
    assert {"date", "symbol", "target_w"}.issubset(w.columns)


def test_run_daily_backtest_outputs_core_columns():
    frame = _mock_factor_frame()
    w = build_cross_sectional_weights(
        frame,
        score_col="score",
        quantile=0.25,
        min_universe=8,
        long_short=True,
    )
    bt = run_daily_backtest(frame, w, cost_rate=0.001)
    assert not bt.empty
    assert {
        "gross_ret",
        "cost",
        "net_ret",
        "turnover",
        "equity",
        "drawdown",
        "bench_ret",
        "bench_equity",
    }.issubset(bt.columns)


def test_perf_stats_returns_numeric_summary():
    frame = _mock_factor_frame()
    w = build_cross_sectional_weights(frame, score_col="score", quantile=0.25, min_universe=8)
    bt = run_daily_backtest(frame, w)
    stats = perf_stats(bt["net_ret"], bt["equity"])
    assert {
        "total_return",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "win_rate",
    }.issubset(stats.keys())

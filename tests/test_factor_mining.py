import numpy as np
import pandas as pd

from omnifinan.analysis.factor_mining import (
    CustomFactorSpec,
    add_candidate_factors,
    apply_custom_factor,
    apply_custom_factors,
    evaluate_factors,
    zscore_by_date,
)


def _mock_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=90, freq="B")
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    rows = []
    for i, sym in enumerate(symbols):
        base = 50 + i * 5
        noise = rng.normal(0, 0.6, len(dates)).cumsum()
        close = base + np.linspace(0, 8, len(dates)) + noise
        high = close * (1 + np.abs(rng.normal(0.01, 0.005, len(dates))))
        low = close * (1 - np.abs(rng.normal(0.01, 0.005, len(dates))))
        volume = rng.integers(1_000_000, 3_000_000, len(dates))
        for d, c, h, l, v in zip(dates, close, high, low, volume):
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "close": float(c),
                    "high": float(h),
                    "low": float(l),
                    "volume": int(v),
                }
            )
    return pd.DataFrame(rows)


def test_factor_pipeline_outputs_expected_columns():
    panel = _mock_panel()
    out = add_candidate_factors(panel)
    required = {
        "ret_1",
        "ret_5",
        "ret_20",
        "mom_ma_5_20",
        "mom_ma_20_60",
        "volatility_20",
        "amplitude_1",
        "vol_ratio_20",
        "rev_5",
        "fwd_ret_5",
    }
    assert required.issubset(set(out.columns))


def test_zscore_by_date_adds_standardized_factor_columns():
    panel = _mock_panel()
    out = add_candidate_factors(panel)
    factored, z_cols = zscore_by_date(out, ["ret_5", "mom_ma_5_20"])

    assert z_cols == ["ret_5_z", "mom_ma_5_20_z"]
    assert all(col in factored.columns for col in z_cols)


def test_evaluate_factors_returns_sorted_report():
    panel = _mock_panel()
    out = add_candidate_factors(panel)
    factored, z_cols = zscore_by_date(out, ["ret_5", "mom_ma_5_20", "vol_ratio_20"])
    report = evaluate_factors(factored, z_cols, label_col="fwd_ret_5")

    assert list(report.columns) == [
        "factor",
        "ic_mean",
        "ic_std",
        "ic_ir",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_ic_ir",
        "obs_days",
    ]
    assert set(report["factor"]) == set(z_cols)
    assert report["rank_ic_mean"].is_monotonic_decreasing


def test_apply_custom_factor_supports_linear_regression_slope():
    panel = _mock_panel()

    def lr_slope_10(group: pd.DataFrame) -> pd.Series:
        x = np.arange(10, dtype=float)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum()

        def _slope(arr: np.ndarray) -> float:
            y = np.asarray(arr, dtype=float)
            y_mean = y.mean()
            return float(((x - x_mean) * (y - y_mean)).sum() / denom)

        return group["close"].rolling(10, min_periods=10).apply(_slope, raw=True)

    out = apply_custom_factor(panel, name="lr_slope_10", func=lr_slope_10)
    assert "lr_slope_10" in out.columns
    assert out["lr_slope_10"].notna().sum() > 0


def test_apply_custom_factors_supports_specs_and_mapping():
    panel = _mock_panel()

    specs = [
        CustomFactorSpec(
            name="ret_3_custom",
            func=lambda g: g["close"].pct_change(3),
        ),
        CustomFactorSpec(
            name="amp_1_custom",
            func=lambda g: (g["high"] - g["low"]) / g["close"],
        ),
    ]
    out = apply_custom_factors(panel, specs)
    assert {"ret_3_custom", "amp_1_custom"}.issubset(out.columns)

    out2 = apply_custom_factors(
        panel,
        {
            "ret_1_custom": lambda g: g["close"].pct_change(1),
        },
    )
    assert "ret_1_custom" in out2.columns


def test_apply_custom_factor_raises_on_bad_length():
    panel = _mock_panel()

    def bad_factor(_: pd.DataFrame) -> np.ndarray:
        return np.array([1.0, 2.0])  # wrong length

    try:
        apply_custom_factor(panel, name="bad_factor", func=bad_factor)
        assert False, "Expected ValueError for incompatible factor output length"
    except ValueError as e:
        assert "bad_factor" in str(e)

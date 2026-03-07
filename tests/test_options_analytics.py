from __future__ import annotations

from omnifinan.analysis.options import bs_price, compute_chain_analytics, compute_implied_volatility


def test_compute_implied_volatility_recovers_input_vol():
    spot = 100.0
    strike = 100.0
    ttm = 30.0 / 365.0
    rate = 0.02
    true_vol = 0.25
    option_type = "call"

    price = bs_price(spot, strike, ttm, rate, true_vol, option_type)
    solved = compute_implied_volatility(
        option_price=price,
        spot=spot,
        strike=strike,
        ttm=ttm,
        rate=rate,
        option_type=option_type,
    )

    assert solved is not None
    assert abs(solved - true_vol) < 1e-3


def test_compute_chain_analytics_normalizes_percentage_point_iv():
    rows = [
        {
            "optionSymbol": "TEST240119C00100000",
            "expiration": "2026-03-20",
            "side": "call",
            "strike": 100.0,
            "dte": 13,
            "mid": 5.0,
            "openInterest": 100.0,
            "iv": 87.5,
        },
        {
            "optionSymbol": "TEST240119P00320000",
            "expiration": "2026-03-20",
            "side": "put",
            "strike": 320.0,
            "dte": 13,
            "mid": 0.05,
            "openInterest": 25.0,
            "iv": 1.12,
        },
    ]

    out = compute_chain_analytics(rows, underlying_price=100.0, risk_free_rate=0.02)
    surface = out["surface"]
    assert len(surface) == 2

    by_strike = {row["strike"]: row for row in surface}
    assert abs(by_strike[100.0]["iv"] - 0.875) < 1e-9
    assert abs(by_strike[320.0]["iv"] - 0.0112) < 1e-9
    assert by_strike[320.0]["gamma"] < 1e-6

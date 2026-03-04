from __future__ import annotations

from omnifinan.analysis.options import bs_price, compute_implied_volatility


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


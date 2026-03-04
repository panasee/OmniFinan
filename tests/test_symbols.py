from __future__ import annotations

from omnifinan.data.symbols import (
    is_china_a_equity_ticker,
    is_crypto_ticker,
    is_hk_equity_ticker,
    is_non_option_equity_market_ticker,
    normalize_crypto_option_underlying,
)


def test_is_crypto_ticker_true_cases():
    assert is_crypto_ticker("BTC-USD")
    assert is_crypto_ticker("eth/usdt")
    assert is_crypto_ticker("SOLUSDT")


def test_is_crypto_ticker_false_cases():
    assert not is_crypto_ticker("AAPL")
    assert not is_crypto_ticker("BRK-B")
    assert not is_crypto_ticker("000001")


def test_equity_market_detection_for_a_hk():
    assert is_china_a_equity_ticker("600519")
    assert is_china_a_equity_ticker("000001.SZ")
    assert is_hk_equity_ticker("00700")
    assert is_hk_equity_ticker("0700.HK")
    assert is_non_option_equity_market_ticker("600519")
    assert is_non_option_equity_market_ticker("00700")
    assert not is_non_option_equity_market_ticker("AAPL")


def test_normalize_crypto_option_underlying():
    assert normalize_crypto_option_underlying("BTC-USDT") == "BTC"
    assert normalize_crypto_option_underlying("ETH-USD") == "ETH"
    assert normalize_crypto_option_underlying("BTCUSDT") == "BTC"
    assert normalize_crypto_option_underlying("ETH") == "ETH"
    assert normalize_crypto_option_underlying("AAPL") == "AAPL"

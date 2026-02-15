from __future__ import annotations

from omnifinan.data.symbols import is_crypto_ticker


def test_is_crypto_ticker_true_cases():
    assert is_crypto_ticker("BTC-USD")
    assert is_crypto_ticker("eth/usdt")
    assert is_crypto_ticker("SOLUSDT")


def test_is_crypto_ticker_false_cases():
    assert not is_crypto_ticker("AAPL")
    assert not is_crypto_ticker("BRK-B")
    assert not is_crypto_ticker("000001")

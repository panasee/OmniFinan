"""Symbol classification helpers."""

from __future__ import annotations


_CRYPTO_QUOTES = {
    "USD",
    "USDT",
    "USDC",
    "BUSD",
    "EUR",
    "GBP",
    "JPY",
    "BTC",
    "ETH",
    "BNB",
    "SOL",
}

_CRYPTO_SUFFIX_QUOTES = ("USDT", "USDC", "BUSD", "BTC", "ETH")


def is_crypto_ticker(ticker: str | None) -> bool:
    """Return True if ticker looks like a crypto pair/symbol."""
    if not ticker:
        return False
    symbol = ticker.strip().upper()
    if not symbol:
        return False

    for sep in ("-", "/"):
        if sep in symbol:
            left, right = symbol.split(sep, 1)
            if left and right and right in _CRYPTO_QUOTES:
                return True

    # Compact pair style such as BTCUSDT, ETHBTC.
    return any(symbol.endswith(suffix) and len(symbol) > len(suffix) for suffix in _CRYPTO_SUFFIX_QUOTES)

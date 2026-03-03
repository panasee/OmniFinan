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

# Well-known bare crypto symbols that should be recognized without a quote suffix.
# This list covers the most commonly traded coins. When the user types "BTC" alone,
# we treat it as a crypto ticker and default the pair to "BTC-USDT".
_BARE_CRYPTO_SYMBOLS = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "MATIC",
    "LINK", "UNI", "ATOM", "LTC", "ETC", "FIL", "APT", "ARB", "OP", "NEAR",
    "SHIB", "TRX", "BCH", "XLM", "ALGO", "VET", "ICP", "FTM", "SAND", "MANA",
    "AAVE", "GRT", "CRV", "MKR", "SNX", "COMP", "YFI", "SUSHI", "CAKE",
    "HBAR", "EOS", "THETA", "XTZ", "ZEC", "DASH", "NEO", "WAVES",
    "TON", "SUI", "SEI", "TIA", "JUP", "WIF", "PEPE", "BONK", "FLOKI",
    "RENDER", "INJ", "STX", "RUNE", "JASMY", "CFX", "ENS", "BLUR",
}


def is_crypto_ticker(ticker: str | None) -> bool:
    """Return True if ticker looks like a crypto pair/symbol.

    Recognises three styles:
    1. Pair with separator: ``BTC-USDT``, ``ETH/USD``
    2. Compact pair: ``BTCUSDT``, ``ETHBTC``
    3. Bare well-known symbol: ``BTC``, ``ETH``, ``SOL``
    """
    if not ticker:
        return False
    symbol = ticker.strip().upper()
    if not symbol:
        return False

    # 1. Pair with separator
    for sep in ("-", "/"):
        if sep in symbol:
            left, right = symbol.split(sep, 1)
            if left and right and right in _CRYPTO_QUOTES:
                return True

    # 2. Compact pair style such as BTCUSDT, ETHBTC.
    if any(symbol.endswith(suffix) and len(symbol) > len(suffix) for suffix in _CRYPTO_SUFFIX_QUOTES):
        return True

    # 3. Bare well-known symbol
    if symbol in _BARE_CRYPTO_SYMBOLS:
        return True

    return False


def normalize_crypto_ticker(ticker: str) -> str:
    """Normalize a crypto ticker to the canonical ``BASE-QUOTE`` format.

    Rules:
    - ``"BTC"``       → ``"BTC-USDT"``   (bare symbol defaults to USDT quote)
    - ``"BTC/USDT"``  → ``"BTC-USDT"``   (slash replaced with dash)
    - ``"BTC-USDT"``  → ``"BTC-USDT"``   (already canonical)
    - ``"BTCUSDT"``   → ``"BTC-USDT"``   (compact pair split)
    """
    symbol = ticker.strip().upper()

    # Already has a dash separator → canonical
    if "-" in symbol:
        return symbol

    # Slash separator → replace with dash
    if "/" in symbol:
        return symbol.replace("/", "-")

    # Compact pair: try to split at known quote suffix
    for suffix in _CRYPTO_SUFFIX_QUOTES:
        if symbol.endswith(suffix) and len(symbol) > len(suffix):
            base = symbol[: -len(suffix)]
            return f"{base}-{suffix}"

    # Bare symbol → default to USDT quote
    if symbol in _BARE_CRYPTO_SYMBOLS:
        return f"{symbol}-USDT"

    # Fallback: return as-is with USDT
    return f"{symbol}-USDT"

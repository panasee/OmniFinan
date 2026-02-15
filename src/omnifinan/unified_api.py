"""
Unified API module for accessing US, HK, and Chinese stock market data using akshare.

This module provides a unified interface for accessing stock market data from
US, Hong Kong, and Chinese (A-shares) markets using the akshare library.
It automatically detects the market based on the stock code and routes requests
to the appropriate akshare functions.

**Compatibility Note:** This version prioritizes maintaining the exact original
data model structures (Price, FinancialMetrics, LineItem, etc.) for backward
compatibility. Fields defined in the models but not available from akshare for
a specific market will be populated with `None`. This may result in less complete
data for certain markets (especially US/HK financial metrics) compared to specialized
APIs, but ensures the output structure remains consistent.
"""

import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Literal  # Added Any for flexibility with model fields

import akshare as ak
import pandas as pd
import requests
from pyomnix.consts import OMNIX_PATH

# Assuming omnix_logger and data_models are in the same package structure
# If not, adjust the import paths accordingly
from pyomnix.omnix_logger import get_logger

from .utils.holidays import filter_trading_days

# Import data models (assuming they are defined correctly AND reflect the original structure)
from .data_models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    MarketType,
    Price,
)

# 鑾峰彇褰撳墠妯″潡鐨勮矾寰?
# Setup logging
logger = get_logger("unified_api")

# --- Helper Functions ---


def load_from_file(symbol: str) -> Any:
    """Load data from a JSON file."""
    file_path = OMNIX_PATH / "financial" / "data" / f"{symbol}.json"
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def normalize_ticker(ticker: str) -> str:
    """
    Normalizes the ticker symbol for use with akshare functions.
    Removes common suffixes like .SH, .SZ, .HK, .O, .N etc.
    Keeps the core identifier.
    """
    # Remove common exchange suffixes
    ticker = re.sub(
        r"\.(US|SH|SZ|HK|O|N|L|DE|PA|AS|MI|VX|SW|CO|HE|ST|OL|IC|KS|KQ|TWO|TW|BK|CR|SA|TL|JK|NZ|AX|IS|TO|NE|BR|VI)$",
        "",
        ticker.upper(),
        flags=re.IGNORECASE,
    )
    # For HK stocks, ensure leading zeros for 5 digits if purely numeric
    if re.match(r"^\d{1,5}$", ticker):
        return ticker.zfill(5)
    # For US stocks or others, return as is after removing suffix
    return ticker


def detect_market(normalized_ticker: str) -> MarketType:
    """
    Detects the market type based on the normalized ticker format.
    Also distinguishes between Shanghai (SH) and Shenzhen (SZ) stocks within China market.

    Returns:
        MarketType: The market type (US, CHINA, HK)
    """
    # Hong Kong: 5 digits (already zero-padded by normalize_ticker)
    if re.fullmatch(r"^\d{5}$", normalized_ticker):
        # Basic check, might need refinement if clashes occur (e.g., US OTC)
        logger.debug(f"Detected Market: HK for {normalized_ticker}")
        return MarketType.HK

    if normalized_ticker == "SH000001":
        return MarketType.CHINA_SH

    # China A-shares: 6 digits
    if re.match(r"^\d{6}$", normalized_ticker):
        # Shanghai (SH) stocks start with 6 or 9
        if normalized_ticker.startswith(("5", "6", "9")):
            logger.debug(f"Detected Market: CHINA (SH) for {normalized_ticker}")
            return MarketType.CHINA_SH
        # Shenzhen (SZ) stocks start with 0, 3, or 2
        elif normalized_ticker.startswith(("0", "3", "2", "4", "8")):
            logger.debug(f"Detected Market: CHINA (SZ) for {normalized_ticker}")
            if normalized_ticker == "000001":
                logger.info(f"Use sh000001 for SSE Index")
            return MarketType.CHINA_SZ
        elif normalized_ticker.startswith(("4", "8")):
            logger.debug(f"Detected Market: CHINA (BJ) for {normalized_ticker}")
            return MarketType.CHINA_BJ

    # US stocks: 1-5 letters, possibly with .A/.B suffix
    if re.fullmatch(r"^[A-Z]{1,5}(\.(A|B))?$", normalized_ticker):
        logger.debug(f"Detected Market: US for {normalized_ticker}")
        return MarketType.US

    # Unknown market
    logger.warning(f"Cannot determine market for ticker: {normalized_ticker}")
    return MarketType.UNKNOWN


def _safe_float_convert(value, default=None) -> float | None:
    """Safely converts a value to float, returning default (None) if conversion fails."""
    if pd.isna(value) or value == "-":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int_convert(value, default=None) -> int | None:
    """Safely converts a value to int, returning default (None) if conversion fails."""
    if pd.isna(value) or value == "-":
        return default
    try:
        # Handle potential floats before converting to int
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _format_date(date_str: str | None, default_delta_days: int = 0) -> str:
    """Formats date string to YYYYMMDD, providing default if None."""
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
        except ValueError:
            logger.warning(f"Invalid date format: {date_str}. Using default.")
            # Fall through to default
    # Default logic (yesterday or based on delta)
    return (datetime.now() - timedelta(days=default_delta_days)).strftime("%Y%m%d")


def _format_output_date(date_input) -> str | None:
    """Converts date-like input to output string. Keeps time for intraday values."""
    if pd.isna(date_input):
        return None
    if isinstance(date_input, str):
        try:
            # Handle YYYYMMDD
            if len(date_input) == 8 and date_input.isdigit():
                return datetime.strptime(date_input, "%Y%m%d").strftime("%Y-%m-%d")
            parsed = pd.to_datetime(date_input)
            if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0:
                return parsed.strftime("%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            logger.debug(f"Could not parse date string: {date_input}")
            return date_input  # Return original string if parsing fails as a fallback? Or None? Let's return None for consistency.
            # return None
    elif isinstance(date_input, (date, pd.Timestamp)):
        if isinstance(date_input, pd.Timestamp):
            if date_input.hour == 0 and date_input.minute == 0 and date_input.second == 0:
                return date_input.strftime("%Y-%m-%d")
            return date_input.strftime("%Y-%m-%d %H:%M:%S")
        return date_input.strftime("%Y-%m-%d")
    logger.debug(f"Could not format date input (type {type(date_input)}): {date_input}")
    return None  # Fallback


def _get_value_from_row(row: pd.Series, key: str, converter: callable, default=None) -> Any:
    """Gets value from series, handles missing key, calls converter."""
    if key in row.index:
        return converter(row.get(key))
    return default


def _infer_date_and_value_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Infer date and numeric value columns for noisy macro datasets."""
    if df is None or df.empty:
        return None, None
    date_col = None
    value_col = None
    best_date_count = -1
    best_value_count = -1

    for col in df.columns:
        parsed_dates = pd.to_datetime(df[col], errors="coerce")
        date_count = int(parsed_dates.notna().sum())
        if date_count > best_date_count:
            best_date_count = date_count
            date_col = col

        parsed_numbers = pd.to_numeric(df[col], errors="coerce")
        value_count = int(parsed_numbers.notna().sum())
        if value_count > best_value_count:
            best_value_count = value_count
            value_col = col

    if best_date_count <= 0:
        date_col = None
    if best_value_count <= 0:
        value_col = None
    return date_col, value_col


def _normalize_macro_series(
    df: pd.DataFrame,
    *,
    series_name: str,
    source: str,
    start_date: str | None = None,
    end_date: str | None = None,
    date_col: str | None = None,
    value_col: str | None = None,
) -> dict[str, Any]:
    """Normalize raw macro dataframe into a stable date-value series payload."""
    if df is None or df.empty:
        return {
            "series": series_name,
            "source": source,
            "observations": [],
            "latest": None,
            "previous": None,
            "trend": "flat",
            "error": "empty dataframe",
        }

    local_date_col, local_value_col = _infer_date_and_value_columns(df)
    use_date_col = date_col or local_date_col
    use_value_col = value_col or local_value_col
    if use_date_col is None or use_value_col is None:
        return {
            "series": series_name,
            "source": source,
            "observations": [],
            "latest": None,
            "previous": None,
            "trend": "flat",
            "error": "could not infer date/value columns",
        }

    work = df[[use_date_col, use_value_col]].copy()
    work.columns = ["date", "value"]
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["date", "value"]).sort_values("date")

    if start_date:
        start_dt = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_dt):
            work = work[work["date"] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_dt):
            work = work[work["date"] <= end_dt]

    observations = [
        {"date": ts.strftime("%Y-%m-%d"), "value": float(val)}
        for ts, val in zip(work["date"], work["value"], strict=False)
    ]
    latest = observations[-1] if observations else None
    previous = observations[-2] if len(observations) > 1 else None
    trend = "flat"
    if latest and previous:
        if latest["value"] > previous["value"]:
            trend = "up"
        elif latest["value"] < previous["value"]:
            trend = "down"

    return {
        "series": series_name,
        "source": source,
        "observations": observations,
        "latest": latest,
        "previous": previous,
        "trend": trend,
        "error": None,
    }


def _fetch_sofr_series(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Fetch SOFR from New York Fed public endpoint."""
    start = start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    url = (
        "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
        f"?startDate={start}&endDate={end}&type=rate"
    )
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("refRates", [])
        df = pd.DataFrame(rows)
        if df.empty:
            return {
                "series": "sofr",
                "source": "nyfed",
                "observations": [],
                "latest": None,
                "previous": None,
                "trend": "flat",
                "error": "empty SOFR response",
            }
        return _normalize_macro_series(
            df,
            series_name="sofr",
            source="nyfed",
            start_date=start_date,
            end_date=end_date,
            date_col="effectiveDate",
            value_col="percentRate",
        )
    except Exception as exc:
        logger.warning("SOFR fetch failed: %s", exc)
        return {
            "series": "sofr",
            "source": "nyfed",
            "observations": [],
            "latest": None,
            "previous": None,
            "trend": "flat",
            "error": str(exc),
        }


def get_macro_indicators(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Fetch major macro indicators used in top-down financial analysis.

    Indicator coverage (when available from current AkShare version):
    - Policy/liquidity: Fed policy rate, PBOC policy rate, China LPR, China SHIBOR 3M, SOFR
    - Inflation: US CPI YoY, US core PCE, China CPI YoY, China PPI YoY
    - Growth/activity: US GDP, US ISM PMI, US retail sales, China GDP YoY, China PMI (manufacturing/non-manufacturing)
    - Labor: US unemployment, US non-farm payrolls, US initial jobless claims, China urban unemployment
    - External/credit: China M2 YoY, exports YoY, imports YoY, trade balance, FX reserves
    """
    results: dict[str, Any] = {"series": {}}
    fetch_specs = {
        # US policy + inflation + activity + labor
        "fed_policy_rate": {"source": "akshare", "fetcher_name": "macro_bank_usa_interest_rate"},
        "us_cpi_yoy": {"source": "akshare", "fetcher_name": "macro_usa_cpi_yoy"},
        "us_core_pce_price": {"source": "akshare", "fetcher_name": "macro_usa_core_pce_price"},
        "us_unemployment_rate": {"source": "akshare", "fetcher_name": "macro_usa_unemployment_rate"},
        "us_non_farm_payrolls": {"source": "akshare", "fetcher_name": "macro_usa_non_farm"},
        "us_initial_jobless_claims": {"source": "akshare", "fetcher_name": "macro_usa_initial_jobless"},
        "us_ism_pmi": {"source": "akshare", "fetcher_name": "macro_usa_ism_pmi"},
        "us_retail_sales": {"source": "akshare", "fetcher_name": "macro_usa_retail_sales"},
        "us_gdp_growth": {"source": "akshare", "fetcher_name": "macro_usa_gdp_monthly"},
        "us_industrial_production": {"source": "akshare", "fetcher_name": "macro_usa_industrial_production"},
        # China policy + inflation + activity + labor + external
        "pboc_policy_rate": {"source": "akshare", "fetcher_name": "macro_bank_china_interest_rate"},
        "china_lpr_1y": {
            "source": "akshare",
            "fetcher_name": "macro_china_lpr",
            "date_col": "TRADE_DATE",
            "value_col": "LPR1Y",
        },
        "china_shibor_3m": {"source": "akshare", "fetcher_name": "macro_china_shibor_all"},
        "china_cpi_yoy": {"source": "akshare", "fetcher_name": "macro_china_cpi_yearly"},
        "china_ppi_yoy": {"source": "akshare", "fetcher_name": "macro_china_ppi_yearly"},
        "china_gdp_yoy": {"source": "akshare", "fetcher_name": "macro_china_gdp_yearly"},
        "china_pmi_manufacturing": {"source": "akshare", "fetcher_name": "macro_china_pmi_yearly"},
        "china_pmi_non_manufacturing": {"source": "akshare", "fetcher_name": "macro_china_non_man_pmi"},
        "china_urban_unemployment": {"source": "akshare", "fetcher_name": "macro_china_urban_unemployment"},
        "china_m2_yoy": {"source": "akshare", "fetcher_name": "macro_china_m2_yearly"},
        "china_exports_yoy": {"source": "akshare", "fetcher_name": "macro_china_exports_yoy"},
        "china_imports_yoy": {"source": "akshare", "fetcher_name": "macro_china_imports_yoy"},
        "china_trade_balance": {"source": "akshare", "fetcher_name": "macro_china_trade_balance"},
        "china_fx_reserves": {"source": "akshare", "fetcher_name": "macro_china_fx_reserves_yearly"},
    }

    for key, spec in fetch_specs.items():
        source = str(spec.get("source", "akshare"))
        fetcher_name = str(spec.get("fetcher_name", ""))
        fetcher = getattr(ak, fetcher_name, None)
        if fetcher is None:
            results["series"][key] = {
                "series": key,
                "source": source,
                "observations": [],
                "latest": None,
                "previous": None,
                "trend": "flat",
                "error": f"akshare fetcher unavailable: {fetcher_name}",
            }
            continue

        try:
            df = fetcher()
            if key == "china_shibor_3m":
                date_col = next(
                    (
                        col
                        for col in df.columns
                        if str(col).upper() in {"DATE", "日期", "TRADE_DATE"}
                    ),
                    None,
                )
                value_col = next(
                    (
                        col
                        for col in df.columns
                        if "3M" in str(col).upper()
                    ),
                    None,
                )
                series_payload = _normalize_macro_series(
                    df,
                    series_name=key,
                    source=source,
                    start_date=start_date,
                    end_date=end_date,
                    date_col=date_col,
                    value_col=value_col,
                )
            else:
                series_payload = _normalize_macro_series(
                    df,
                    series_name=key,
                    source=source,
                    start_date=start_date,
                    end_date=end_date,
                    date_col=spec.get("date_col"),
                    value_col=spec.get("value_col"),
                )
        except Exception as exc:
            logger.warning("Macro fetch failed for %s via %s: %s", key, fetcher_name, exc)
            series_payload = {
                "series": key,
                "source": source,
                "observations": [],
                "latest": None,
                "previous": None,
                "trend": "flat",
                "error": str(exc),
            }
        results["series"][key] = series_payload

    results["series"]["sofr"] = _fetch_sofr_series(start_date=start_date, end_date=end_date)
    results["snapshot_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    results["latest"] = {
        key: val.get("latest", {}).get("value") if isinstance(val, dict) and val.get("latest") else None
        for key, val in results["series"].items()
    }
    return results


def _is_crypto_symbol(symbol: str) -> bool:
    """Heuristic crypto symbol detection for holiday filtering bypass."""
    upper = symbol.upper()
    if "/" in upper or "-" in upper:
        return True
    quote_suffixes = ("USDT", "USDC", "BUSD", "BTC", "ETH")
    return any(upper.endswith(suffix) for suffix in quote_suffixes)


def _holiday_filter_market_code(market: MarketType, normalized_ticker: str) -> str | None:
    """Map MarketType to holiday filter market code used by utils.holidays."""
    if market == MarketType.US:
        return "US"
    if market == MarketType.HK:
        return "HK"
    if market in [MarketType.CHINA_SZ, MarketType.CHINA_SH, MarketType.CHINA_BJ, MarketType.CHINA]:
        return "A"
    if _is_crypto_symbol(normalized_ticker):
        # Crypto trades 24/7, do not remove weekend/holiday rows.
        return "NA"
    return None


def _apply_price_date_and_holiday_filters(
    df: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    market: MarketType,
    normalized_ticker: str,
) -> pd.DataFrame:
    """Apply date-range clipping and market holiday filtering to fetched prices."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "date" not in df.columns:
        return df

    filtered = df.copy()
    filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
    filtered = filtered.dropna(subset=["date"])
    if filtered.empty:
        return filtered

    if start_date:
        start_dt = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_dt):
            filtered = filtered[filtered["date"] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_dt):
            filtered = filtered[filtered["date"] <= end_dt]

    market_code = _holiday_filter_market_code(market, normalized_ticker)
    if market_code:
        try:
            calendar_input = filtered.rename(columns={"date": "time"})
            calendar_filtered = filter_trading_days(calendar_input, market=market_code)
            filtered = (
                calendar_filtered.rename(columns={"time": "date"})
                if "time" in calendar_filtered.columns
                else calendar_filtered
            )
        except Exception as holiday_error:
            logger.warning(
                "Holiday filter skipped for %s (%s): %s",
                normalized_ticker,
                market.name,
                holiday_error,
            )
            if "date" not in filtered.columns and "time" in filtered.columns:
                filtered = filtered.rename(columns={"time": "date"})
    if "date" in filtered.columns:
        return filtered.sort_values("date").reset_index(drop=True)
    return filtered.reset_index(drop=True)


def _normalize_interval(interval: str) -> str:
    normalized = str(interval or "1d").lower().strip()
    allowed = {"1d", "1m", "3m", "5m", "15m", "30m", "60m"}
    if normalized not in allowed:
        logger.warning("Unsupported interval '%s', fallback to '1d'.", interval)
        return "1d"
    return normalized


def _format_minute_datetime(date_str: str | None, default_dt: datetime) -> str:
    if not date_str:
        return default_dt.strftime("%Y-%m-%d %H:%M:%S")
    parsed = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(parsed):
        return default_dt.strftime("%Y-%m-%d %H:%M:%S")
    if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0:
        return parsed.strftime("%Y-%m-%d 00:00:00")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _to_us_minute_symbol(symbol: str) -> str:
    # AkShare EastMoney US minute endpoint expects "105.MSFT" style symbols.
    if re.fullmatch(r"^\d+\.[A-Z\.]+$", symbol):
        return symbol
    return f"105.{symbol}"


def _standardize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    cols = list(work.columns)
    lower_map = {str(col).lower(): col for col in cols}

    rename_map: dict[str, str] = {}
    if "date" in lower_map:
        rename_map[lower_map["date"]] = "date"
    elif "time" in lower_map:
        rename_map[lower_map["time"]] = "date"
    elif "datetime" in lower_map:
        rename_map[lower_map["datetime"]] = "date"

    if "open" in lower_map:
        rename_map[lower_map["open"]] = "open"
    if "close" in lower_map:
        rename_map[lower_map["close"]] = "close"
    if "high" in lower_map:
        rename_map[lower_map["high"]] = "high"
    if "low" in lower_map:
        rename_map[lower_map["low"]] = "low"
    if "volume" in lower_map:
        rename_map[lower_map["volume"]] = "volume"
    if "amount" in lower_map:
        rename_map[lower_map["amount"]] = "amount"

    work = work.rename(columns=rename_map)

    # Fallback for minute datasets where columns may be non-English:
    # [date, open, close, high, low, volume, amount, ...]
    col_names = list(work.columns)
    if "date" not in work.columns and len(col_names) >= 1:
        work = work.rename(columns={col_names[0]: "date"})
        col_names = list(work.columns)
    if "open" not in work.columns and len(col_names) >= 2:
        work = work.rename(columns={col_names[1]: "open"})
        col_names = list(work.columns)
    if "close" not in work.columns and len(col_names) >= 3:
        work = work.rename(columns={col_names[2]: "close"})
        col_names = list(work.columns)
    if "high" not in work.columns and len(col_names) >= 4:
        work = work.rename(columns={col_names[3]: "high"})
        col_names = list(work.columns)
    if "low" not in work.columns and len(col_names) >= 5:
        work = work.rename(columns={col_names[4]: "low"})
        col_names = list(work.columns)
    if "volume" not in work.columns and len(col_names) >= 6:
        work = work.rename(columns={col_names[5]: "volume"})
        col_names = list(work.columns)
    if "amount" not in work.columns and len(col_names) >= 7:
        work = work.rename(columns={col_names[6]: "amount"})

    return work


def _resample_intraday_prices(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")
    if work.empty:
        return pd.DataFrame()

    for numeric_col in ("open", "high", "low", "close", "volume", "amount"):
        if numeric_col in work.columns:
            work[numeric_col] = pd.to_numeric(work[numeric_col], errors="coerce")

    agg_map: dict[str, str] = {}
    if "open" in work.columns:
        agg_map["open"] = "first"
    if "high" in work.columns:
        agg_map["high"] = "max"
    if "low" in work.columns:
        agg_map["low"] = "min"
    if "close" in work.columns:
        agg_map["close"] = "last"
    if "volume" in work.columns:
        agg_map["volume"] = "sum"
    if "amount" in work.columns:
        agg_map["amount"] = "sum"

    if not agg_map:
        return work

    rs = work.set_index("date").resample(rule).agg(agg_map).dropna(subset=["open", "high", "low", "close"])
    rs = rs.reset_index()
    rs["date"] = rs["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return rs


def _finalize_price_volume_and_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and auto-complete `volume` (鎴愪氦閲? and `amount` (鎴愪氦棰?."""
    if df is None or df.empty:
        return df

    work = df.copy()
    for col in ("close", "volume", "amount"):
        if col not in work.columns:
            work[col] = pd.NA
        work[col] = pd.to_numeric(work[col], errors="coerce")

    close = work["close"]
    volume = work["volume"]
    amount = work["amount"]

    # If amount is missing but volume is present, estimate鎴愪氦棰?by close * volume.
    amount_fill_mask = amount.isna() & volume.notna() & close.notna()
    work.loc[amount_fill_mask, "amount"] = volume[amount_fill_mask] * close[amount_fill_mask]

    # If volume is missing but amount is present, infer鎴愪氦閲?by amount / close.
    volume_fill_mask = volume.isna() & amount.notna() & close.notna() & (close != 0)
    work.loc[volume_fill_mask, "volume"] = amount[volume_fill_mask] / close[volume_fill_mask]

    # Price model expects int volume; keep 0 when still missing after completion.
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0)
    work["volume"] = work["volume"].clip(lower=0).round().astype("int64")

    # amount remains nullable float.
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    return work


# --- Unified API Functions ---


def get_prices(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: Literal["1d", "1m", "3m", "5m", "15m", "30m", "60m"] = "1d",
    adjustment: Literal["qfq", "hfq", ""] = "",
    api: Literal["sina", "em"] = "sina",
    provider: Literal["akshare", "finnhub", "yfinance"] = "akshare",
) -> list[Price]:
    """
    Fetch price data compatible with the original Price model structure.

    Args:
        ticker: The ticker symbol (e.g., "AAPL", "00700", "600519")
        start_date: Start date in "YYYY-MM-DD" format. Defaults to 1 year ago.
        end_date: End date in "YYYY-MM-DD" format. Defaults to yesterday.
        interval: K-line interval. Supports "1d", "1m", "3m", "5m", "15m", "30m", "60m".
        adjustment: Price adjustment type.
        api: Data source for daily bars, "sina" or "em".

    Returns:
        List[Price]: A list of price data objects matching the original structure.
    """
    if provider != "akshare":
        from .data.providers.factory import create_data_provider

        external_provider = create_data_provider(provider)
        return external_provider.get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

    normalized_ticker = normalize_ticker(ticker)
    market = detect_market(normalized_ticker)
    interval = _normalize_interval(interval)
    intraday = interval != "1d"

    if normalized_ticker == "SH000001":
        normalized_ticker = "000001"

    end_date_ak = _format_date(end_date, default_delta_days=0)
    if start_date:
        start_date_ak = _format_date(start_date)
    else:
        end_dt = datetime.strptime(end_date_ak, "%Y%m%d")
        start_dt = end_dt - timedelta(days=365)
        start_date_ak = start_dt.strftime("%Y%m%d")

    logger.info(
        "Getting price history for %s (%s), interval=%s, from %s to %s",
        normalized_ticker,
        market.name,
        interval,
        start_date_ak,
        end_date_ak,
    )

    prices_list: list[Price] = []
    try:
        df = pd.DataFrame()

        if intraday:
            minute_period = "1" if interval == "3m" else interval.replace("m", "")
            minute_start = _format_minute_datetime(
                start_date,
                default_dt=datetime.now() - timedelta(days=7),
            )
            minute_end = _format_minute_datetime(
                end_date,
                default_dt=datetime.now(),
            )

            if market == MarketType.US:
                # AkShare US minute endpoint usually provides the most recent few sessions.
                df = ak.stock_us_hist_min_em(
                    symbol=_to_us_minute_symbol(normalized_ticker),
                    start_date=minute_start,
                    end_date=minute_end,
                )
            elif market == MarketType.HK:
                try:
                    df = ak.stock_hk_hist_min_em(
                        symbol=normalized_ticker,
                        period=minute_period,
                        start_date=minute_start,
                        end_date=minute_end,
                        adjust=adjustment,
                    )
                except TypeError:
                    df = ak.stock_hk_hist_min_em(
                        symbol=normalized_ticker,
                        period=minute_period,
                        start_date=minute_start,
                        end_date=minute_end,
                    )
            elif market in [MarketType.CHINA_SZ, MarketType.CHINA_SH, MarketType.CHINA_BJ]:
                try:
                    df = ak.stock_zh_a_hist_min_em(
                        symbol=normalized_ticker,
                        period=minute_period,
                        start_date=minute_start,
                        end_date=minute_end,
                        adjust=adjustment,
                    )
                except TypeError:
                    df = ak.stock_zh_a_hist_min_em(
                        symbol=normalized_ticker,
                        period=minute_period,
                        start_date=minute_start,
                        end_date=minute_end,
                    )
            else:
                logger.warning("Intraday not supported for market: %s", market.name)
                return []

            df = _standardize_price_columns(df)
            resample_rule_map = {
                "3m": "3min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "60m": "60min",
            }
            # US endpoint returns 1-minute bars; resample for requested higher intervals.
            # Also resample 3-minute bars for all markets because there is no direct 3m endpoint.
            if interval in resample_rule_map and (market == MarketType.US or interval == "3m"):
                df = _resample_intraday_prices(df, rule=resample_rule_map[interval])

        elif api == "sina":
            if market == MarketType.US:
                df = ak.stock_us_daily(symbol=normalized_ticker, adjust=adjustment)
            elif market == MarketType.HK:
                df = ak.stock_hk_daily(symbol=normalized_ticker, adjust=adjustment)
            elif market in [MarketType.CHINA_SZ, MarketType.CHINA_SH, MarketType.CHINA_BJ]:
                df = ak.stock_zh_a_daily(
                    symbol=f"{market.value}{normalized_ticker}",
                    start_date=start_date_ak,
                    end_date=end_date_ak,
                    adjust=adjustment,
                )
            df = _standardize_price_columns(df)

        elif api == "em":
            if market == MarketType.US:
                df = ak.stock_us_hist(
                    symbol=normalized_ticker,
                    period="daily",
                    start_date=start_date_ak,
                    end_date=end_date_ak,
                    adjust=adjustment,
                )
            elif market == MarketType.HK:
                df = ak.stock_hk_hist(
                    symbol=normalized_ticker,
                    period="daily",
                    start_date=start_date_ak,
                    end_date=end_date_ak,
                    adjust=adjustment,
                )
            else:
                df = ak.stock_zh_a_hist(
                    symbol=normalized_ticker,
                    period="daily",
                    start_date=start_date_ak,
                    end_date=end_date_ak,
                    adjust=adjustment,
                )
            df = _standardize_price_columns(df)

        if df is None or df.empty:
            logger.warning("No price data found for %s in the specified range.", normalized_ticker)
            return []

        filter_start = start_date
        filter_end = end_date
        if intraday:
            if filter_start and len(str(filter_start)) <= 10:
                filter_start = f"{filter_start} 00:00:00"
            if filter_end and len(str(filter_end)) <= 10:
                filter_end = f"{filter_end} 23:59:59"

        df = _apply_price_date_and_holiday_filters(
            df,
            start_date=filter_start,
            end_date=filter_end,
            market=market,
            normalized_ticker=normalized_ticker,
        )
        if df.empty:
            logger.warning("No price data left after filtering for %s", normalized_ticker)
            return []

        if "amount" not in df.columns:
            df["amount"] = None
        if "amplitude" not in df.columns:
            df["amplitude"] = None
        if "pct_change" not in df.columns:
            df["pct_change"] = df["close"].pct_change() * 100
        if "change_amount" not in df.columns:
            df["change_amount"] = df["close"].diff()
        if "turnover" not in df.columns:
            df["turnover"] = None

        df = _finalize_price_volume_and_amount(df)

        for _, row in df.iterrows():
            prices_list.append(
                Price(
                    open=_get_value_from_row(row, "open", _safe_float_convert),
                    high=_get_value_from_row(row, "high", _safe_float_convert),
                    low=_get_value_from_row(row, "low", _safe_float_convert),
                    close=_get_value_from_row(row, "close", _safe_float_convert),
                    volume=_get_value_from_row(row, "volume", _safe_int_convert),
                    time=_get_value_from_row(row, "date", _format_output_date),
                    amount=_get_value_from_row(row, "amount", _safe_float_convert),
                    amplitude=_get_value_from_row(row, "amplitude", _safe_float_convert),
                    pct_change=_get_value_from_row(row, "pct_change", _safe_float_convert),
                    change_amount=_get_value_from_row(row, "change_amount", _safe_float_convert),
                    turnover=_get_value_from_row(row, "turnover", _safe_float_convert),
                    market=market,
                )
            )

        logger.info("Successfully fetched %s price records for %s", len(prices_list), normalized_ticker)

    except Exception as e:
        logger.error(f"Error getting price history for {normalized_ticker}: {e}", exc_info=True)
        return []

    return prices_list


def get_financial_metrics(
    ticker: str,
    end_date: str | None = None,
    period: str = "ttm",  # Keep 'ttm' as original requested period type
    limit: int = 1,  # Fetch only the latest available snapshot consistent with original behavior
    *,
    manual_input: dict[str, Any] | None = None,
) -> list[FinancialMetrics]:
    """
    Fetch available financial metrics compatible with the original FinancialMetrics model.
    Many detailed metrics (ROE, Margins, Growth, etc.) will be None for US/HK markets
    as they are not readily available via simple akshare functions.

    Args:
        ticker: The ticker symbol.
        end_date: Contextual date, sets 'report_period' if provided.
        period: Period type (e.g., "ttm"). Primarily affects 'period' field in output.
        limit: Max number of results (typically 1 for latest snapshot).
        manual_input: Optional manual input for ticker info that can not be fetched (will skip fetching process)
            {
                "ticker": normalized_ticker,
                "report_period": end_date if end_date else datetime.now().strftime("%Y-%m-%d"),
                "period": period,
                "currency": "USD"
                if market == MarketType.US
                else "HKD"
                if market == MarketType.HK
                else "CNY",
                "market": market,
                "market_cap": None,
                "price_to_earnings_ratio": None,
                "price_to_book_ratio": None,
                "price_to_sales_ratio": None,
                "return_on_equity": None,
                "net_margin": None,
                "operating_margin": None,
                "revenue_growth": None,
                "earnings_growth": None,
                "book_value_growth": None,
                "current_ratio": None,
                "debt_to_equity": None,  # Note: CN indicator is Debt/Assets
                "free_cash_flow_per_share": None,  # Requires complex calculation
                "earnings_per_share": None,
            }


    Returns:
        List[FinancialMetrics]: List containing FinancialMetrics object(s) matching original structure.
    """
    #TODO
    logger.warning(f"sockets still need to be supplemented")

    if manual_input is not None:
        metric_data = manual_input

    else:
        normalized_ticker = normalize_ticker(ticker)
        market = detect_market(normalized_ticker)
        logger.info(f"Getting financial metrics for {normalized_ticker} ({market.name})...")

        # Initialize all fields from the original model with None or defaults
        metric_data = {
            "ticker": normalized_ticker,
            "report_period": end_date if end_date else datetime.now().strftime("%Y-%m-%d"),
            "period": period,
            "currency": "USD"
            if market == MarketType.US
            else "HKD"
            if market == MarketType.HK
            else "CNY",
            "market": market,
            "market_cap": None,
            "price_to_earnings_ratio": None,
            "price_to_book_ratio": None,
            "price_to_sales_ratio": None,
            "return_on_equity": None,
            "net_margin": None,
            "operating_margin": None,
            "revenue_growth": None,
            "earnings_growth": None,
            "book_value_growth": None,
            "current_ratio": None,
            "debt_to_equity": None,  # Note: CN indicator is Debt/Assets
            "free_cash_flow_per_share": None,  # Requires complex calculation
            "earnings_per_share": None,
        }

        try:
            # --- Fetch Base Data ---
            quote_data = pd.Series(dtype=object)
            indicator_data = pd.Series(dtype=object)
            calculated_market_cap = None

            if market == MarketType.US:
                q_df = ak.stock_individual_basic_info_us_xq(symbol=normalized_ticker)
                if not q_df.empty:
                    quote_data = q_df.iloc[0]
                else:
                    logger.warning(f"Could not fetch US quote data for {normalized_ticker}")
            elif market == MarketType.HK:
                q_df = ak.stock_individual_basic_info_hk_xq(symbol=normalized_ticker)
                if not q_df.empty:
                    quote_data = q_df.iloc[0]
                else:
                    logger.warning(f"Could not fetch HK quote data for {normalized_ticker}")
            else:  # MarketType.CHINA
                # Calculate Market Cap
                latest_price = None
                prices = get_prices(normalized_ticker, end_date=datetime.now().strftime("%Y-%m-%d"))
                if prices:
                    latest_price = prices[-1].close

                total_shares = None
                try:
                    info_df = ak.stock_individual_info_em(symbol=normalized_ticker, timeout=20)
                    total_shares_series = info_df[
                        info_df["item"].astype(str).str.contains("总股本|股本", regex=True, na=False)
                    ]["value"]
                    if not total_shares_series.empty:
                        value_str = total_shares_series.iloc[0]
                        num_part_list = re.findall(r"[\d\.]+", value_str)
                        if num_part_list:
                            num_part = num_part_list[0]
                            multiplier = 1e8 if "亿" in value_str else 1e4 if "万" in value_str else 1
                            total_shares = float(num_part) * multiplier
                except Exception as e_info:
                    logger.warning(
                        f"Failed getting CN total shares for {normalized_ticker}: {e_info}"
                    )

                if latest_price is not None and total_shares is not None:
                    calculated_market_cap = latest_price * total_shares

                # Fetch Indicators
                try:
                    current_year = datetime.now().year
                    ind_df = ak.stock_financial_analysis_indicator(
                        symbol=normalized_ticker, start_year=str(current_year - 1)
                    )
                    if not ind_df.empty:
                        ind_df["鏃ユ湡"] = pd.to_datetime(ind_df["鏃ユ湡"])
                        indicator_data = ind_df.sort_values("鏃ユ湡", ascending=False).iloc[0]
                except Exception as e_ind:
                    logger.warning(
                        f"Failed fetching CN financial indicators for {normalized_ticker}: {e_ind}"
                    )

            # --- Populate metric_data using fetched info ---

            # Market Cap
            if market == MarketType.US:
                metric_data["market_cap"] = _safe_float_convert(quote_data.get("market_capital"))
            elif market == MarketType.HK:
                metric_data["market_cap"] = _safe_float_convert(quote_data.get("market_value"))
            else:  # CN
                metric_data["market_cap"] = calculated_market_cap  # Already calculated

            # PE Ratio
            if market == MarketType.US or market == MarketType.HK:
                metric_data["price_to_earnings_ratio"] = _safe_float_convert(
                    quote_data.get("pe_ratio")
                )
            elif not indicator_data.empty:  # CN from indicators
                metric_data["price_to_earnings_ratio"] = _safe_float_convert(
                    indicator_data.get("甯傜泩鐜嘝E(TTM)")
                )

            # PB Ratio (Only available from CN indicators in this setup)
            if (
                market in [MarketType.CHINA_SZ, MarketType.CHINA_SH, MarketType.CHINA_BJ]
                and not indicator_data.empty
            ):
                metric_data["price_to_book_ratio"] = _safe_float_convert(
                    indicator_data.get("甯傚噣鐜嘝B(MRQ)")
                )

            # Update report_period if CN indicators provided a date
            if (
                market in [MarketType.CHINA_SZ, MarketType.CHINA_SH, MarketType.CHINA_BJ]
                and not indicator_data.empty
            ):
                indicator_date = _format_output_date(indicator_data.get("鏃ユ湡"))
                if indicator_date:
                    metric_data["report_period"] = indicator_date

            # CN Specific Metrics from Indicators
            if (
                market in [MarketType.CHINA_SZ, MarketType.CHINA_SH, MarketType.CHINA_BJ]
                and not indicator_data.empty
            ):

                def _parse_indicator(key, is_percent=False, default=None):
                    val = indicator_data.get(key)
                    if pd.isna(val) or val == "-":
                        return default
                    try:
                        num_val = float(val)
                        return num_val / 100.0 if is_percent else num_val
                    except ValueError:
                        return default

                metric_data["return_on_equity"] = _parse_indicator(
                    "鍑€璧勪骇鏀剁泭鐜嘡OE(鍔犳潈)", is_percent=True
                )
                metric_data["net_margin"] = _parse_indicator("閿€鍞噣鍒╃巼(%)", is_percent=True)
                metric_data["operating_margin"] = _parse_indicator("钀ヤ笟鍒╂鼎鐜?%)", is_percent=True)
                metric_data["revenue_growth"] = _parse_indicator(
                    "涓昏惀涓氬姟鏀跺叆澧為暱鐜?%)", is_percent=True
                )
                metric_data["earnings_growth"] = _parse_indicator(
                    "鍑€鍒╂鼎澧為暱鐜?%)", is_percent=True
                )
                metric_data["current_ratio"] = _parse_indicator("娴佸姩姣旂巼")
                metric_data["debt_to_equity"] = _parse_indicator(
                    "璧勪骇璐熷€虹巼(%)", is_percent=True
                )  # Note: D/A ratio
                metric_data["earnings_per_share"] = _parse_indicator("鍩烘湰姣忚偂鏀剁泭(鍏?")
                # book_value_growth, price_to_sales_ratio, free_cash_flow_per_share remain None

            # Log if detailed metrics are missing for US/HK
            if market == MarketType.US or market == MarketType.HK:
                logger.warning(
                    f"Detailed financial metrics (ROE, Margins, Growth, etc.) are largely unavailable via akshare for {market.name} and are set to None. Use manually input instead"
                )
        except Exception as e:
            logger.error(
                f"Error getting financial metrics for {normalized_ticker}: {e}",
                exc_info=True,
            )

    metrics = FinancialMetrics(**metric_data)
    # Keep return type stable for all callers and support historical `limit` usage.
    # Current data source yields one snapshot, so we return a single-item list.
    return [metrics]


def search_line_items(
    ticker: str,
    period: str = "ttm",  # Original argument
    limit: int = 2,  # Fetch latest and previous period if possible
) -> list[LineItem]:
    """
    Fetch standard financial statement line items compatible with the original LineItem model.
    Calculates specific fields like working_capital, FCF from base statement data.
    Returns None for fields if underlying data is unavailable.

    Args:
        ticker: The ticker symbol.
        line_items: Ignored. Standard items matching LineItem model are calculated.
        end_date: Contextual date.
        period: Period type from original request (e.g., "ttm", "latest"). Affects output 'period' field.
        limit: Max number of periods to return (e.g., 1 for latest, 2 for latest+previous).

    Returns:
        List[LineItem]: List of LineItem objects matching original structure.
    """
    #TODO
    logger.warning(f"sockets still need to be supplemented")

    normalized_ticker = normalize_ticker(ticker)
    market = detect_market(normalized_ticker)
    logger.info(f"Getting financial statement data for {normalized_ticker} ({market.name})")

    results = []
    fetched_data = {}  # Store dataframes: {'balance': df, 'income': df, 'cashflow': df}
    report_dates = set()
    date_col_name = "报告日"  # Default for CN/HK sources

    try:
        # --- Fetch Raw Statement Data ---
        if market in [MarketType.CHINA_SZ, MarketType.CHINA_SH, MarketType.CHINA_BJ]:
            try:
                fetched_data["balance"] = ak.stock_financial_report_sina(
                    stock=f"{market.value}{normalized_ticker}", symbol="资产负债表"
                )
            except Exception as e:
                logger.warning(f"Failed CN Balance Sheet fetch: {e}")
            try:
                fetched_data["income"] = ak.stock_financial_report_sina(
                    stock=f"{market.value}{normalized_ticker}", symbol="利润表"
                )
            except Exception as e:
                logger.warning(f"Failed CN Income Statement fetch: {e}")
            try:
                fetched_data["cashflow"] = ak.stock_financial_report_sina(
                    stock=f"{market.value}{normalized_ticker}", symbol="现金流量表"
                )
            except Exception as e:
                logger.warning(f"Failed CN Cash Flow fetch: {e}")

        elif market == MarketType.HK:
            report_map = {
                "资产负债表": "balance",
                "利润表": "income",
                "现金流量表": "cashflow",
            }
            for report_name_cn, key in report_map.items():
                try:
                    fetched_data[key] = ak.stock_financial_hk_report_em(
                        stock=normalized_ticker, symbol=report_name_cn
                    )
                except Exception as e:
                    logger.warning(f"Failed HK {report_name_cn} fetch: {e}")

        elif market == MarketType.US:
            logger.warning(
                "US financial statement fetching via akshare is limited/unverified and likely won't populate LineItem fields."
            )
            # Placeholder - unlikely to succeed in populating LineItem fields as desired
            # Need validated method for ak.stock_us_fundamental or similar

        # --- Process Fetched Data ---
        # Identify common report dates (use income statement as primary source of dates)
        if "income" in fetched_data and not fetched_data["income"].empty:
            # Try common date column names
            possible_date_cols = [
                "报告日",
                "REPORT_DATE",
                "EndDate",
                "date",
            ]  # Add more if needed
            for col in possible_date_cols:
                if col in fetched_data["income"].columns:
                    date_col_name = col
                    report_dates.update(fetched_data["income"][date_col_name].astype(str).tolist())
                    logger.info(f"Using date column '{date_col_name}' for report periods.")
                    break
            if not report_dates:
                logger.warning("Could not identify a suitable date column in the income statement.")

        if not report_dates:
            logger.warning(
                f"No report dates found for {normalized_ticker}. Cannot extract line items."
            )
            return []

        sorted_dates = sorted(list(report_dates), reverse=True)

        processed_count = 0
        for report_date_str in sorted_dates:
            if processed_count >= limit:
                break

            balance_row = income_row = cashflow_row = pd.Series(dtype=object)

            # Find rows matching the date in each available statement DataFrame
            for stmt_key, df in fetched_data.items():
                if df is not None and not df.empty and date_col_name in df.columns:
                    # Ensure consistent date string comparison
                    try:
                        mask = df[date_col_name].astype(str) == report_date_str
                        if mask.sum() == 0:
                            mask = (
                                pd.to_datetime(df[date_col_name]).dt.strftime("%Y-%m-%d")
                                == report_date_str
                            )
                        if "STD_ITEM_NAME" in df.columns:
                            tmp = df[mask][["STD_ITEM_NAME", "AMOUNT"]].transpose()
                            row_series = tmp.rename(columns=tmp.iloc[0]).drop(index="STD_ITEM_NAME")
                        else:
                            row_series = df[mask]
                        if not row_series.empty:
                            if stmt_key == "balance":
                                balance_row = row_series.iloc[0]
                            elif stmt_key == "income":
                                income_row = row_series.iloc[0]
                            elif stmt_key == "cashflow":
                                cashflow_row = row_series.iloc[0]
                    except Exception as e_match:
                        logger.warning(
                            f"Error matching date {report_date_str} in {stmt_key}: {e_match}"
                        )

            # --- Extract & Calculate Required LineItem Fields ---
            # Column names below are examples based on CN reports - **VERIFY/ADJUST for HK/US sources**
            item_data = {}
            item_data["ticker"] = normalized_ticker
            item_data["report_period"] = _format_output_date(report_date_str)
            # Use original requested period string or sequence if multiple periods
            item_data["period"] = f"{period}_{processed_count}" if limit > 1 else period
            item_data["currency"] = (
                "USD" if market == MarketType.US else "HKD" if market == MarketType.HK else "CNY"
            )

            # Base items (handle potential missing rows with multiple candidate keys)
            def pick_first(row: pd.Series, keys: list[str]) -> float | None:
                for k in keys:
                    val = _get_value_from_row(row, k, _safe_float_convert)
                    if val is not None:
                        return val
                return None

            ni = pick_first(income_row, ["净利润", "归属于母公司股东的净利润", "net_income"])
            rev = pick_first(income_row, ["营业总收入", "营业收入", "operating_revenue", "revenue"])
            op = pick_first(income_row, ["营业利润", "经营利润", "operating_profit"])

            ca = pick_first(balance_row, ["流动资产合计", "current_assets"])
            cl = pick_first(balance_row, ["流动负债合计", "current_liabilities"])

            cfo = pick_first(cashflow_row, ["经营活动产生的现金流量净额", "经营业务现金净额", "operating_cash_flow"])
            dep_amort = pick_first(
                cashflow_row,
                [
                    "固定资产折旧、油气资产折耗、生产性生物资产折旧",
                    "减值及拨备",
                    "depreciation_and_amortization",
                ],
            )
            capex_paid = pick_first(
                cashflow_row,
                [
                    "购建固定资产、无形资产和其他长期资产所支付的现金",
                    "购建无形资产及其他资产",
                    "capital_expenditure",
                ],
            )

            item_data["net_income"] = ni
            item_data["operating_revenue"] = rev
            item_data["operating_profit"] = op
            item_data["depreciation_and_amortization"] = dep_amort

            # Calculated items
            item_data["working_capital"] = (ca - cl) if ca is not None and cl is not None else None
            item_data["capital_expenditure"] = abs(capex_paid) if capex_paid is not None else None
            item_data["free_cash_flow"] = (
                (cfo - item_data["capital_expenditure"])
                if cfo is not None and item_data["capital_expenditure"] is not None
                else None
            )

            try:
                results.append(LineItem(**item_data))  # Pass the dict containing potential Nones
                processed_count += 1
            except Exception as model_error:
                logger.error(
                    f"Error creating LineItem model for {normalized_ticker} date {report_date_str}: {model_error}"
                )

    except Exception as e:
        logger.error(
            f"Error getting financial statements for {normalized_ticker}: {e}",
            exc_info=True,
        )

    # Return based on the requested period logic (simplistic: just return what was found up to limit)
    # The period field inside LineItem indicates sequence if limit > 1
    return results


# TODO: no insider trades info now
def get_insider_trades(
    ticker: str,
    end_date: str | None = None,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """
    Fetch insider trades compatible with the original InsiderTrade model.
    Only implemented for US market via akshare. Returns empty list for CN/HK.

    Args:
        ticker: The ticker symbol.
        end_date: End date filter ("YYYY-MM-DD").
        start_date: Start date filter ("YYYY-MM-DD").
        limit: Maximum number of results.

    Returns:
        List[InsiderTrade]: List of insider trade objects matching original structure.
    """
    normalized_ticker = normalize_ticker(ticker)
    market = detect_market(normalized_ticker)
    trades = []

    if market != MarketType.US:
        logger.info(
            f"Insider trade data is not currently available via akshare for market: {market.name}"
        )
        return []

    logger.info(f"Getting US insider trades for {normalized_ticker}...")
    try:
        # **VERIFY** column names from ak.stock_us_insider_trade output
        df = ak.stock_us_insider_trade(symbol=normalized_ticker)

        if df is None or df.empty:
            logger.warning(f"No insider trade data found for {normalized_ticker}")
            return []

        # --- Manual Date Filtering & Limit ---
        date_col = "浜ゆ槗鏃ユ湡"  # Verify name
        if date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col])
                df = df.sort_values(by=date_col, ascending=False)
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    df = df[df[date_col].dt.date >= start_dt.date()]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    df = df[df[date_col].dt.date <= end_dt.date()]
            except Exception as date_e:
                logger.warning(f"Could not perform date filtering on insider trades: {date_e}")
        else:
            logger.warning(f"Date column '{date_col}' not found for insider trades.")

        df = df.head(limit)

        # Convert DataFrame rows to InsiderTrade objects
        for _, row in df.iterrows():
            filing_date = _get_value_from_row(row, date_col, _format_output_date)
            trade_data = InsiderTrade(
                ticker=normalized_ticker,
                issuer=None,
                name=_get_value_from_row(row, "内部人名称", str, default=None),
                title=_get_value_from_row(row, "职务", str, default=None),
                is_board_director=None,
                transaction_date=filing_date,
                transaction_shares=_get_value_from_row(row, "交易股数", _safe_float_convert),
                transaction_price_per_share=_get_value_from_row(row, "交易价格", _safe_float_convert),
                transaction_value=None,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=_get_value_from_row(row, "持有股数", _safe_float_convert),
                security_title=None,
                filing_date=filing_date or datetime.now().strftime("%Y-%m-%d"),
            )
            trades.append(trade_data)

        logger.info(
            f"Successfully fetched {len(trades)} insider trade records for {normalized_ticker}"
        )

    except Exception as e:
        logger.error(f"Error getting insider trades for {normalized_ticker}: {e}", exc_info=True)
        return []

    return trades


def get_stock_news(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 10,
) -> list[CompanyNews]:
    """Fetch and normalize company news for a symbol."""
    market = detect_market(symbol)
    limit = max(1, min(limit, 100))

    try:
        news_df = ak.stock_news_em(symbol=symbol)
    except Exception as e:
        logger.error("Error fetching news for %s: %s", symbol, e, exc_info=True)
        return []

    if news_df is None or news_df.empty:
        return []

    # Normalize common column names from AkShare.
    col_map = {}
    for col in news_df.columns:
        col_str = str(col)
        if "标题" in col_str:
            col_map[col] = "title"
        elif "发布时间" in col_str or "时间" in col_str:
            col_map[col] = "date"
        elif "来源" in col_str:
            col_map[col] = "source"
        elif "链接" in col_str:
            col_map[col] = "url"
        elif "关键词" in col_str:
            col_map[col] = "keyword"
        elif "内容" in col_str:
            col_map[col] = "content"

    df = news_df.rename(columns=col_map)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date", ascending=False)

    if start_date and "date" in df.columns:
        start_dt = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_dt):
            df = df[df["date"] >= start_dt]
    if end_date and "date" in df.columns:
        end_dt = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_dt):
            df = df[df["date"] <= end_dt]

    rows = df.head(limit).to_dict(orient="records")
    result: list[CompanyNews] = []
    for row in rows:
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        date_val = row.get("date")
        if isinstance(date_val, pd.Timestamp):
            date_str = date_val.strftime("%Y-%m-%d %H:%M:%S")
        else:
            date_str = str(date_val) if date_val is not None else ""

        result.append(
            CompanyNews(
                ticker=symbol,
                title=title,
                source=str(row.get("source", "")).strip(),
                date=date_str,
                url=str(row.get("url", "")).strip(),
                market=market,
                publish_time=date_str,
                content=str(row.get("content", "")).strip() or None,
                keyword=str(row.get("keyword", "")).strip() or None,
            )
        )

    return result


def get_market_cap(
    ticker: str,
    end_date: str | None = None,
    manual_input: dict[str, Any] | None = None,
) -> float | None:
    """
    Fetch the latest market cap compatible with original expectation (float or potentially None).

    Args:
        ticker: The ticker symbol.
        end_date: Kept for interface consistency, ignored otherwise.

    Returns:
        Optional[float]: The market cap value, or None if unavailable.
    """
    # This function's logic from the previous refactoring already returns Optional[float]
    # and uses the specific quote functions or calculation, which is suitable.
    # No major changes needed here, just ensure it matches the previous logic.
    if manual_input is not None:
        return manual_input.get("market_cap", None)

    # TODO: auto-retrieval currently unavailable
    normalized_ticker = normalize_ticker(ticker)
    market = detect_market(normalized_ticker)
    logger.info(f"Fetching latest market cap for {normalized_ticker} ({market.name})...")

    market_cap = None
    try:
        if market == MarketType.US:
            quote_df = ak.stock_quote_us_sina(symbol=normalized_ticker)
            if not quote_df.empty:
                market_cap = _safe_float_convert(quote_df.iloc[0].get("market_capital"))
            else:
                logger.warning(f"Could not fetch US quote data for market cap: {normalized_ticker}")

        elif market == MarketType.HK:
            quote_df = ak.stock_quote_hk_sina(symbol=normalized_ticker)
            if not quote_df.empty:
                market_cap = _safe_float_convert(quote_df.iloc[0].get("market_value"))
            else:
                logger.warning(f"Could not fetch HK quote data for market cap: {normalized_ticker}")

        else:  # MarketType.CHINA
            latest_price = None
            prices_list = get_prices(normalized_ticker, end_date=end_date)
            if prices_list:
                latest_price = prices_list[-1].close
            else:
                logger.warning(
                    f"Could not get latest price for CN {normalized_ticker} for market cap calc."
                )

            total_shares = None
            try:
                info_df = ak.stock_individual_info_em(symbol=normalized_ticker)
                total_shares_series = info_df[
                    info_df["item"].astype(str).str.contains("总股本|股本", regex=True, na=False)
                ]["value"]
                if not total_shares_series.empty:
                    value_str = total_shares_series.iloc[0]
                    num_part_list = re.findall(r"[\d\.]+", value_str)
                    if num_part_list:
                        num_part = num_part_list[0]
                        multiplier = 1e8 if "亿" in value_str else 1e4 if "万" in value_str else 1
                        total_shares = float(num_part) * multiplier
                    else:
                        logger.warning(f"Could not parse number from total shares value: {value_str}")
                else:
                    logger.warning(f"Could not find total shares in info for CN {normalized_ticker}")
            except Exception as e_info:
                logger.warning(f"Failed to get total shares for CN {normalized_ticker}: {e_info}")

            if (
                latest_price is not None
                and total_shares is not None
                and latest_price > 0
                and total_shares > 0
            ):
                market_cap = latest_price * total_shares
            else:
                logger.warning(
                    f"Could not calculate market cap for CN {normalized_ticker}. Price: {latest_price}, Shares: {total_shares}"
                )

        if market_cap is not None:
            logger.info(f"Market cap for {normalized_ticker}: {market_cap}")
        else:
            logger.warning(f"Market cap for {normalized_ticker} is unavailable.")
        return market_cap

    except Exception as e:
        logger.error(f"Error getting market cap for {normalized_ticker}: {e}", exc_info=True)
        return None


def get_company_news(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 10,
) -> list[CompanyNews]:
    """Backward-compatible alias for get_stock_news."""
    return get_stock_news(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """
    Convert a list of Price objects (adhering to original structure with potential Nones)
    to a pandas DataFrame, setting DatetimeIndex.

    Args:
        prices: List of Price objects.

    Returns:
        pd.DataFrame: DataFrame containing price data with Date index.
    """
    if not prices:
        return pd.DataFrame()

    # Convert list of Pydantic models (or similar objects) to list of dicts
    # Assuming Price objects have a way to be converted to dict (e.g., Pydantic's model_dump)
    data = []
    for price in prices:
        if hasattr(price, "model_dump"):
            data.append(price.model_dump())
        elif hasattr(price, "__dict__"):
            data.append(vars(price))
        else:
            logger.error("Cannot convert Price object to dict. Skipping.")
            continue

    df = pd.DataFrame(data)

    # Set index - use 'time' field as defined in Price model
    date_col = "time"
    if date_col not in df.columns:
        logger.error(f"Date column '{date_col}' not found in Price data for DataFrame conversion.")
        return df  # Return df without index

    try:
        # Ensure the date column is suitable for conversion before setting index
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])  # Drop rows where date conversion failed
        if not df.empty:
            df = df.set_index(pd.DatetimeIndex(df[date_col]))
            # Optionally drop the original column if it's now the index name or redundant
            if df.index.name == date_col:
                df = df.drop(columns=[date_col])
        else:
            logger.warning("DataFrame empty after dropping rows with invalid dates.")
            return pd.DataFrame()  # Return empty df
    except Exception as e:
        logger.error(f"Failed to set DatetimeIndex during DataFrame conversion: {e}")
        # Fallback: return df without index if conversion/setting fails

    # Ensure numeric columns (based on original Price model)
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change_amount",
        "turnover",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by index (date)
    if isinstance(df.index, pd.DatetimeIndex):
        df.sort_index(inplace=True)

    return df


def get_price_df(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: Literal["1d", "1m", "3m", "5m", "15m", "30m", "60m"] = "1d",
    provider: Literal["akshare", "finnhub", "yfinance"] = "akshare",
) -> pd.DataFrame:
    """
    Fetch price data for the given ticker and convert directly to DataFrame,
    maintaining compatibility with the original data structure expectations.

    Args:
        ticker: The ticker symbol.
        start_date: Start date in "YYYY-MM-DD" format. Defaults to 1 year ago.
        end_date: End date in "YYYY-MM-DD" format. Defaults to yesterday.

    Returns:
        pd.DataFrame: DataFrame containing price data with Date index.
    """
    prices = get_prices(
        ticker,
        start_date,
        end_date,
        interval=interval,
        provider=provider,
    )
    return prices_to_df(prices)


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(logging.INFO)  # Ensure module logger level is also INFO

    # Test US Stock
    print("\n--- Testing US Stock (AAPL) ---")
    aapl_ticker = "AAPL"
    try:
        aapl_prices = get_prices(aapl_ticker, start_date="2024-01-01", end_date="2024-04-01")
        if aapl_prices:
            print(f"AAPL Prices (last 2): {aapl_prices[-2:]}")
            aapl_df = prices_to_df(aapl_prices)
            print(f"AAPL Price DataFrame tail:\n{aapl_df.tail(2)}")
        else:
            print("No AAPL price data found")
    except Exception as e:
        print(f"Error getting AAPL prices: {e}")

    try:
        aapl_metrics = get_financial_metrics(aapl_ticker)
        print(f"AAPL Metrics: {aapl_metrics}")
    except Exception as e:
        print(f"Error getting AAPL metrics: {e}")

    try:
        aapl_cap = get_market_cap(aapl_ticker)
        print(f"AAPL Market Cap: {aapl_cap}")
    except Exception as e:
        print(f"Error getting AAPL market cap: {e}")

    try:
        aapl_news = get_company_news(aapl_ticker, limit=1)
        print(f"AAPL News (limit 1): {aapl_news}")
    except Exception as e:
        print(f"Error getting AAPL news: {e}")

    try:
        aapl_insider = get_insider_trades(aapl_ticker, limit=1)
        print(f"AAPL Insider Trades (limit 1): {aapl_insider}")
    except Exception as e:
        print(f"Error getting AAPL insider trades: {e}")

    try:
        aapl_items = search_line_items(
            aapl_ticker,
        )  # Items ignored
        print(f"AAPL Line Items (latest): {aapl_items}")
    except Exception as e:
        print(f"Error getting AAPL line items: {e}")

    # Test HK Stock
    print("\n--- Testing HK Stock (00700) ---")
    tencent_ticker = "00700"  # Tencent
    try:
        tencent_prices = get_prices(tencent_ticker, start_date="2024-01-01", end_date="2024-04-01")
        if tencent_prices:
            print(f"Tencent Prices (last 2): {tencent_prices[-2:]}")
            tencent_df = get_price_df(
                tencent_ticker, start_date="2024-01-01", end_date="2024-04-01"
            )
            print(f"Tencent Price DataFrame tail:\n{tencent_df.tail(2)}")
        else:
            print("No Tencent price data found")
    except Exception as e:
        print(f"Error getting Tencent prices: {e}")

    try:
        tencent_metrics = get_financial_metrics(tencent_ticker)
        print(f"Tencent Metrics: {tencent_metrics}")
    except Exception as e:
        print(f"Error getting Tencent metrics: {e}")

    try:
        tencent_cap = get_market_cap(tencent_ticker)
        print(f"Tencent Market Cap: {tencent_cap}")
    except Exception as e:
        print(f"Error getting Tencent market cap: {e}")

    try:
        tencent_news = get_company_news(tencent_ticker, limit=1)
        print(f"Tencent News (limit 1): {tencent_news}")
    except Exception as e:
        print(f"Error getting Tencent news: {e}")

    try:
        tencent_items = search_line_items(tencent_ticker, limit=1)
        print(f"Tencent Line Items (latest): {tencent_items}")
    except Exception as e:
        print(f"Error getting Tencent line items: {e}")

    # Test CN Stock
    print("\n--- Testing CN Stock (600519) ---")
    moutai_ticker = "600519"  # Kweichow Moutai
    try:
        moutai_prices = get_prices(moutai_ticker, start_date="2024-01-01", end_date="2024-04-01")
        if moutai_prices:
            print(f"Moutai Prices (last 2): {moutai_prices[-2:]}")
            moutai_df = get_price_df(
                moutai_ticker, start_date="2024-01-01", end_date="2024-04-01"
            )
            print(f"Moutai Price DataFrame tail:\n{moutai_df.tail(2)}")
        else:
            print("No Moutai price data found")
    except Exception as e:
        print(f"Error getting Moutai prices: {e}")

    try:
        moutai_metrics = get_financial_metrics(moutai_ticker)
        print(f"Moutai Metrics: {moutai_metrics}")
    except Exception as e:
        print(f"Error getting Moutai metrics: {e}")

    try:
        moutai_cap = get_market_cap(moutai_ticker)
        print(f"Moutai Market Cap: {moutai_cap}")
    except Exception as e:
        print(f"Error getting Moutai market cap: {e}")

    try:
        moutai_news = get_company_news(moutai_ticker, limit=1)
        print(f"Moutai News (limit 1): {moutai_news}")
    except Exception as e:
        print(f"Error getting Moutai news: {e}")

    try:
        moutai_items = search_line_items(moutai_ticker, limit=2)  # Get latest 2 periods
        print(f"Moutai Line Items (latest 2): {moutai_items}")
    except Exception as e:
        print(f"Error getting Moutai line items: {e}")




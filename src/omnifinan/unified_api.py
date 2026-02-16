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
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from io import StringIO
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

    sample = df.head(300)
    for col in sample.columns:
        s = sample[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            date_count = int(s.notna().sum())
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed_dates = pd.to_datetime(s, errors="coerce")
            date_count = int(parsed_dates.notna().sum())
        if date_count > best_date_count:
            best_date_count = date_count
            date_col = col

        if pd.api.types.is_numeric_dtype(s):
            value_count = int(s.notna().sum())
        else:
            parsed_numbers = pd.to_numeric(s, errors="coerce")
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

    local_date_col = None
    local_value_col = None
    if date_col is None or value_col is None:
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
    work["date"] = _coerce_macro_date_series(work["date"])
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["date", "value"]).sort_values("date")
    full_work = work.copy()

    if start_date:
        start_dt = pd.to_datetime(start_date, errors="coerce")
        if pd.notna(start_dt):
            work = work[work["date"] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_dt):
            work = work[work["date"] <= end_dt]

    if work.empty and not full_work.empty:
        # Keep data continuity for low-frequency series even when caller window is narrow.
        work = full_work.tail(24).copy()

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

    payload = {
        "series": series_name,
        "source": source,
        "observations": observations,
        "latest": latest,
        "previous": previous,
        "trend": trend,
        "error": None,
    }
    if observations and (start_date or end_date) and full_work.shape[0] > work.shape[0]:
        payload["note"] = "window_filtered_or_fallback_latest_used"
    return payload


MACRO_SOURCE_POLICY_VERSION = "fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank"


def _coerce_macro_date_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    raw = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # 1) YYYYMMDD
    m = raw.str.fullmatch(r"\d{8}")
    if m.any():
        out.loc[m] = pd.to_datetime(raw.loc[m], format="%Y%m%d", errors="coerce")

    # 2) YYYYMM
    m = raw.str.fullmatch(r"\d{6}")
    if m.any():
        out.loc[m] = pd.to_datetime(raw.loc[m] + "01", format="%Y%m%d", errors="coerce")

    # 3) YYYY-MM / YYYY/MM
    m = raw.str.fullmatch(r"\d{4}[-/]\d{1,2}")
    if m.any():
        norm = raw.loc[m].str.replace("/", "-", regex=False) + "-01"
        out.loc[m] = pd.to_datetime(norm, format="%Y-%m-%d", errors="coerce")

    # 4) YYYY年MM月份 / YYYY年MM月
    m = raw.str.fullmatch(r"\d{4}年\d{1,2}(月份|月)")
    if m.any():
        norm = (
            raw.loc[m]
            .str.replace("年份", "年", regex=False)
            .str.replace("月份", "月", regex=False)
            .str.replace("年", "-", regex=False)
            .str.replace("月", "-01", regex=False)
        )
        out.loc[m] = pd.to_datetime(norm, format="%Y-%m-%d", errors="coerce")

    # 5) YYYY年M-N月 -> use N month
    m = raw.str.fullmatch(r"\d{4}年\d{1,2}-\d{1,2}月")
    if m.any():
        extracted = raw.loc[m].str.extract(r"(?P<y>\d{4})年(?P<m1>\d{1,2})-(?P<m2>\d{1,2})月")
        norm = extracted["y"] + "-" + extracted["m2"].str.zfill(2) + "-01"
        out.loc[m] = pd.to_datetime(norm, format="%Y-%m-%d", errors="coerce")

    # 6) fallback parser without warning spam
    rest = out.isna()
    if rest.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            out.loc[rest] = pd.to_datetime(raw.loc[rest], errors="coerce")
    return out


def _macro_error_payload(series_name: str, source: str, error: str) -> dict[str, Any]:
    return {
        "series": series_name,
        "source": source,
        "observations": [],
        "latest": None,
        "previous": None,
        "trend": "flat",
        "error": error,
    }


def _get_fred_api_key() -> str | None:
    env_key = os.getenv("FRED_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        from .data.providers.credentials import get_api_key, load_provider_credentials

        cfg_key = get_api_key("fred")
        if cfg_key:
            return cfg_key

        payload = load_provider_credentials()
        if isinstance(payload, dict):
            for key_name in ("fred_api_key", "FRED_API_KEY"):
                raw = payload.get(key_name)
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
            node = payload.get("fred")
            if isinstance(node, dict):
                raw = node.get("key")
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
    except Exception:
        return None
    return None


def _fred_fetch_series(
    *,
    series_name: str,
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    transform: str | None = None,
) -> dict[str, Any]:
    source = f"fred:{series_id}"
    params: dict[str, Any] = {"series_id": series_id, "file_type": "json"}
    api_key = _get_fred_api_key()
    if api_key:
        params["api_key"] = api_key
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date
    def _finalize(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return _macro_error_payload(series_name, source, "empty FRED response")
        if transform == "yoy":
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            df["value"] = (df["value"] / df["value"].shift(12) - 1.0) * 100.0
            df = df.dropna(subset=["value"])
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return _normalize_macro_series(
            df,
            series_name=series_name,
            source=source,
            start_date=start_date,
            end_date=end_date,
            date_col="date",
            value_col="value",
        )

    # Primary: FRED JSON API
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("observations", [])
        clean_rows: list[dict[str, Any]] = []
        for row in rows:
            raw_value = row.get("value")
            if raw_value in (None, ".", ""):
                continue
            try:
                clean_rows.append({"date": str(row.get("date")), "value": float(raw_value)})
            except (TypeError, ValueError):
                continue
        return _finalize(pd.DataFrame(clean_rows))
    except Exception as exc:
        logger.warning("FRED JSON fetch failed for %s(%s): %s", series_name, series_id, exc)

    # Fallback: official FRED graph CSV endpoint (works in environments where JSON API rejects requests).
    try:
        csv_params: dict[str, Any] = {"id": series_id}
        if start_date:
            csv_params["cosd"] = start_date
        if end_date:
            csv_params["coed"] = end_date
        resp_csv = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params=csv_params,
            timeout=20,
        )
        resp_csv.raise_for_status()
        df = pd.read_csv(
            StringIO(resp_csv.text),
            dtype=str,
        )
        if df.empty or "DATE" not in df.columns:
            return _macro_error_payload(series_name, source, "unexpected FRED CSV response")
        value_col = next((c for c in df.columns if c != "DATE"), None)
        if value_col is None:
            return _macro_error_payload(series_name, source, "FRED CSV missing value column")
        df = df.rename(columns={"DATE": "date", value_col: "value"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        return _finalize(df[["date", "value"]].copy())
    except Exception as exc:
        logger.warning("FRED CSV fallback failed for %s(%s): %s", series_name, series_id, exc)
        return _macro_error_payload(series_name, source, str(exc))


def _world_bank_fetch_series(
    *,
    series_name: str,
    indicator: str,
    country: str = "USA",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    source = f"world_bank:{country}:{indicator}"
    try:
        resp = requests.get(
            f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}",
            params={"format": "json", "per_page": 20000},
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return _macro_error_payload(series_name, source, "unexpected World Bank response")
        rows = payload[1]
        parsed: list[dict[str, Any]] = []
        for row in rows:
            year = row.get("date")
            value = row.get("value")
            if year in (None, "") or value in (None, ""):
                continue
            try:
                parsed.append({"date": f"{int(year):04d}-12-31", "value": float(value)})
            except (TypeError, ValueError):
                continue
        df = pd.DataFrame(parsed)
        if df.empty:
            return _macro_error_payload(series_name, source, "empty World Bank response")
        return _normalize_macro_series(
            df,
            series_name=series_name,
            source=source,
            start_date=start_date,
            end_date=end_date,
            date_col="date",
            value_col="value",
        )
    except Exception as exc:
        logger.warning("World Bank fetch failed for %s(%s): %s", series_name, indicator, exc)
        return _macro_error_payload(series_name, source, str(exc))


def _imf_fetch_series(
    *,
    series_name: str,
    indicator: str,
    country: str = "USA",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    source = f"imf_datamapper:{indicator}:{country}"
    try:
        resp = requests.get(
            f"https://www.imf.org/external/datamapper/api/v1/{indicator}/{country}",
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        values = payload.get("values", {})
        indicator_map = values.get(indicator, {})
        country_map = indicator_map.get(country, {})
        parsed: list[dict[str, Any]] = []
        for year, value in country_map.items():
            if value in (None, ""):
                continue
            try:
                parsed.append({"date": f"{int(year):04d}-12-31", "value": float(value)})
            except (TypeError, ValueError):
                continue
        df = pd.DataFrame(parsed)
        if df.empty:
            return _macro_error_payload(series_name, source, "empty IMF response")
        return _normalize_macro_series(
            df,
            series_name=series_name,
            source=source,
            start_date=start_date,
            end_date=end_date,
            date_col="date",
            value_col="value",
        )
    except Exception as exc:
        logger.warning("IMF fetch failed for %s(%s): %s", series_name, indicator, exc)
        return _macro_error_payload(series_name, source, str(exc))


def _akshare_fetch_series(
    *,
    series_name: str,
    fetcher_name: str,
    source: str,
    start_date: str | None = None,
    end_date: str | None = None,
    date_col: str | None = None,
    value_col: str | None = None,
) -> dict[str, Any]:
    fetcher = getattr(ak, fetcher_name, None)
    if fetcher is None:
        return _macro_error_payload(series_name, source, f"akshare fetcher unavailable: {fetcher_name}")
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fetcher)
            df = fut.result(timeout=300)
        if series_name == "china_urban_unemployment" and {"item", "value", "date"}.issubset(set(df.columns)):
            mask = df["item"].astype(str).str.contains("全国城镇调查失业率", na=False)
            if mask.any():
                df = df.loc[mask, ["date", "value"]].copy()
        use_date_col = date_col
        use_value_col = value_col
        if series_name == "china_shibor_3m":
            use_date_col = next(
                (col for col in df.columns if str(col).upper() in {"DATE", "日期", "TRADE_DATE"}),
                None,
            )
            use_value_col = next((col for col in df.columns if "3M" in str(col).upper()), None)
        return _normalize_macro_series(
            df,
            series_name=series_name,
            source=source,
            start_date=start_date,
            end_date=end_date,
            date_col=use_date_col,
            value_col=use_value_col,
        )
    except FuturesTimeoutError:
        logger.warning("AkShare macro fetch timed out for %s via %s", series_name, fetcher_name)
        return _macro_error_payload(series_name, source, "timeout")
    except Exception as exc:
        logger.warning("AkShare macro fetch failed for %s via %s: %s", series_name, fetcher_name, exc)
        return _macro_error_payload(series_name, source, str(exc))


def _payload_has_observations(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    obs = payload.get("observations", [])
    return isinstance(obs, list) and len(obs) > 0 and payload.get("error") is None


def _first_non_empty_payload(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for payload in candidates:
        if _payload_has_observations(payload):
            return payload
    return candidates[0] if candidates else _macro_error_payload("unknown", "unknown", "no payload candidates")


def _mas_fetch_sg_10y_yield(
    *,
    series_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    source = "mas:sgs_original_maturity_10y"
    try:
        resp = requests.get(
            "https://eservices.mas.gov.sg/statistics/fdanet/BondOriginalMaturities.aspx?type=NX",
            timeout=30,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            return _macro_error_payload(series_name, source, "empty MAS table")
        df = tables[0]
        # Flatten two-level headers (Yield/Price columns per issue).
        if isinstance(df.columns, pd.MultiIndex):
            flat_cols = []
            for lv0, lv1 in df.columns:
                a = str(lv0).strip()
                b = str(lv1).strip()
                flat_cols.append(f"{a}|{b}" if b and b.lower() != "nan" else a)
            df.columns = flat_cols
        else:
            df.columns = [str(c).strip() for c in df.columns]

        date_col = next((c for c in df.columns if "Issue Code" in c), None)
        if date_col is None:
            date_col = df.columns[0]
        yield_cols = [c for c in df.columns if str(c).strip().endswith("|Yield") or str(c).strip() == "Yield"]
        if not yield_cols:
            yield_cols = [c for c in df.columns if "Yield" in str(c)]
        if not yield_cols:
            return _macro_error_payload(series_name, source, "MAS 10Y page missing yield columns")

        work = pd.DataFrame()
        work["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        ymat = df[yield_cols].apply(pd.to_numeric, errors="coerce")
        work["value"] = ymat.mean(axis=1, skipna=True)
        work = work.dropna(subset=["date", "value"]).sort_values("date")
        if work.empty:
            return _macro_error_payload(series_name, source, "MAS 10Y page parsed empty")
        work["date"] = work["date"].dt.strftime("%Y-%m-%d")
        return _normalize_macro_series(
            work[["date", "value"]],
            series_name=series_name,
            source=source,
            start_date=start_date,
            end_date=end_date,
            date_col="date",
            value_col="value",
        )
    except Exception as exc:
        logger.warning("MAS SG 10Y fetch failed for %s: %s", series_name, exc)
        return _macro_error_payload(series_name, source, str(exc))


def _parse_ism_month_label(label: str, *, index_hint: int) -> str | None:
    raw = str(label).strip()
    if not raw:
        return None
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d")
    # Fallback for labels like "January"/"Jan" without explicit year.
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    token = raw[:3].lower()
    month = months.get(token)
    if month is None:
        return None
    # Infer year by walking backwards from current month for 12M history tables.
    today = date.today()
    year = today.year
    if month > today.month:
        year -= 1
    # index_hint handles same-month labels in older rows.
    year -= index_hint // 12
    return f"{year:04d}-{month:02d}-01"


def _extract_ism_table_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df is None or df.empty:
        return rows

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    lc_cols = [c.lower() for c in work.columns]

    # Long format: has a month/date column and a PMI column.
    pmi_col = next((c for c in work.columns if "pmi" in c.lower()), None)
    date_col = next((c for c in work.columns if any(k in c.lower() for k in ("month", "date"))), None)
    if pmi_col and date_col:
        for i, row in work.iterrows():
            raw_value = pd.to_numeric(row.get(pmi_col), errors="coerce")
            if pd.isna(raw_value):
                continue
            d = _parse_ism_month_label(str(row.get(date_col)), index_hint=int(i))
            if d is None:
                continue
            rows.append({"date": d, "value": float(raw_value)})
        return rows

    # Wide format: first column is row label, one row named PMI and month columns in header.
    if work.shape[1] >= 3 and not any("pmi" in c for c in lc_cols):
        label_col = work.columns[0]
        pmi_row = None
        for _, row in work.iterrows():
            label = str(row.get(label_col, "")).lower()
            if "pmi" in label:
                pmi_row = row
                break
        if pmi_row is not None:
            month_cols = work.columns[1:]
            for i, col in enumerate(month_cols):
                raw_value = pd.to_numeric(pmi_row.get(col), errors="coerce")
                if pd.isna(raw_value):
                    continue
                d = _parse_ism_month_label(str(col), index_hint=i)
                if d is None:
                    continue
                rows.append({"date": d, "value": float(raw_value)})
            return rows
    return rows


def _ism_fetch_series(
    *,
    series_name: str,
    report_type: Literal["manufacturing", "services"],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    source = f"ism_public:{report_type}"
    month_names = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    today = date.today()
    idx = today.month - 1
    prev_idx = (idx - 1) % 12
    base = (
        "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-pmi-reports/pmi/"
        if report_type == "manufacturing"
        else "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-pmi-reports/services/"
    )
    urls = [f"{base}{month_names[idx]}/", f"{base}{month_names[prev_idx]}/"]
    try:
        tables = []
        last_exc: Exception | None = None
        for url in urls:
            try:
                resp = requests.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    timeout=25,
                )
                resp.raise_for_status()
                tables = pd.read_html(StringIO(resp.text))
                if tables:
                    break
            except Exception as exc:
                last_exc = exc
                continue
        if not tables:
            if last_exc:
                return _macro_error_payload(series_name, source, str(last_exc))
            return _macro_error_payload(series_name, source, "ISM table not found")
        parsed_rows: list[dict[str, Any]] = []
        for table in tables:
            parsed_rows.extend(_extract_ism_table_rows(table))
        if not parsed_rows:
            return _macro_error_payload(series_name, source, "ISM PMI rows not parsed")
        df = pd.DataFrame(parsed_rows).drop_duplicates(subset=["date"]).sort_values("date")
        return _normalize_macro_series(
            df,
            series_name=series_name,
            source=source,
            start_date=start_date,
            end_date=end_date,
            date_col="date",
            value_col="value",
        )
    except Exception as exc:
        logger.warning("ISM fetch failed for %s(%s): %s", series_name, report_type, exc)
        return _macro_error_payload(series_name, source, str(exc))


def _load_us_pmi_local_series(
    *,
    series_name: str,
    bucket: Literal["service", "manufacturing"],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    source = f"local:us_pmi_json:{bucket}"
    p = OMNIX_PATH / "omnifinan" / "datasets" / "macro_indicators_history" / "us_pmi.json"
    if not p.exists():
        return _macro_error_payload(series_name, source, f"missing file: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows = payload.get(bucket, []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return _macro_error_payload(series_name, source, f"invalid bucket type: {bucket}")
        parsed: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            d = row.get("date")
            v = row.get("value")
            if d in (None, "") or v in (None, ""):
                continue
            try:
                parsed.append({"date": str(d), "value": float(v)})
            except (TypeError, ValueError):
                continue
        if not parsed:
            return _macro_error_payload(series_name, source, f"empty or invalid rows for {bucket}")
        df = pd.DataFrame(parsed).drop_duplicates(subset=["date"]).sort_values("date")
        return _normalize_macro_series(
            df,
            series_name=series_name,
            source=source,
            start_date=start_date,
            end_date=end_date,
            date_col="date",
            value_col="value",
        )
    except Exception as exc:
        return _macro_error_payload(series_name, source, str(exc))


def _load_us_pmi_series_with_fallback(
    *,
    series_name: str,
    ak_fetcher: str,
    local_bucket: Literal["service", "manufacturing"],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    primary = _akshare_fetch_series(
        series_name=series_name,
        fetcher_name=ak_fetcher,
        source=f"akshare:us_macro:{ak_fetcher}",
        start_date=start_date,
        end_date=end_date,
        date_col="日期",
        value_col="今值",
    )
    if _payload_has_observations(primary):
        return primary
    fallback = _load_us_pmi_local_series(
        series_name=series_name,
        bucket=local_bucket,
        start_date=start_date,
        end_date=end_date,
    )
    if _payload_has_observations(fallback):
        return fallback
    return primary


def _payload_latest_date(payload: dict[str, Any] | None) -> date | None:
    if not isinstance(payload, dict):
        return None
    latest = payload.get("latest")
    if isinstance(latest, dict):
        d = latest.get("date")
        if isinstance(d, str):
            for fmt in ("%Y-%m-%d", "%Y%m%d"):
                try:
                    return datetime.strptime(d[:10], fmt).date()
                except Exception:
                    continue
    return None


def _derived_growth_from_level_payload(
    base_payload: dict[str, Any],
    *,
    series_name: str,
    periods: int,
    source: str,
    annualize: bool = False,
) -> dict[str, Any]:
    obs = base_payload.get("observations", []) if isinstance(base_payload, dict) else []
    if not isinstance(obs, list) or not obs:
        return _macro_error_payload(series_name, source, "base observations unavailable")
    rows = []
    for row in obs:
        if not isinstance(row, dict):
            continue
        d = row.get("date")
        v = row.get("value")
        if d is None or v is None:
            continue
        rows.append({"date": d, "value": v})
    df = pd.DataFrame(rows)
    if df.empty:
        return _macro_error_payload(series_name, source, "base observations unavailable")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    pct = df["value"].pct_change(periods=periods)
    if annualize:
        pct = ((1.0 + pct) ** 4) - 1.0
    df["value"] = pct * 100.0
    df = df.dropna(subset=["value"])
    if df.empty:
        return _macro_error_payload(series_name, source, "insufficient history for growth calculation")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return _normalize_macro_series(
        df[["date", "value"]],
        series_name=series_name,
        source=source,
        date_col="date",
        value_col="value",
    )


def _clone_series_payload(base_payload: dict[str, Any], *, series_name: str, source: str) -> dict[str, Any]:
    if not isinstance(base_payload, dict):
        return _macro_error_payload(series_name, source, "base payload unavailable")
    cloned = dict(base_payload)
    cloned["series"] = series_name
    cloned["source"] = source
    return cloned


def get_macro_indicators(
    start_date: str | None = None,
    end_date: str | None = None,
    include_series: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch macro indicators with fixed source policy.

    Fixed sources:
    - China: AkShare wrappers for official China sources (NBS/PBOC/SAFE, etc.)
    - International: FRED / IMF / World Bank only
    """
    results: dict[str, Any] = {"series": {}}
    include = set(include_series) if include_series else None

    def _want(key: str) -> bool:
        return include is None or key in include

    # International: FRED / IMF / World Bank (fixed)
    if _want("fed_policy_rate"):
        results["series"]["fed_policy_rate"] = _fred_fetch_series(
            series_name="fed_policy_rate",
            series_id="FEDFUNDS",
            start_date=start_date,
            end_date=end_date,
        )
    if _want("sofr"):
        results["series"]["sofr"] = _fred_fetch_series(
            series_name="sofr",
            series_id="SOFR",
            start_date=start_date,
            end_date=end_date,
        )
    if _want("us_cpi_yoy"):
        results["series"]["us_cpi_yoy"] = _fred_fetch_series(
            series_name="us_cpi_yoy",
            series_id="CPIAUCSL",
            start_date=start_date,
            end_date=end_date,
            transform="yoy",
        )
    if _want("us_core_pce_price"):
        results["series"]["us_core_pce_price"] = _fred_fetch_series(
            series_name="us_core_pce_price",
            series_id="PCEPILFE",
            start_date=start_date,
            end_date=end_date,
            transform="yoy",
        )
    if _want("us_unemployment_rate"):
        results["series"]["us_unemployment_rate"] = _fred_fetch_series(
            series_name="us_unemployment_rate",
            series_id="UNRATE",
            start_date=start_date,
            end_date=end_date,
        )
    if _want("us_non_farm_payrolls"):
        results["series"]["us_non_farm_payrolls"] = _fred_fetch_series(
            series_name="us_non_farm_payrolls",
            series_id="PAYEMS",
            start_date=start_date,
            end_date=end_date,
        )
    if _want("us_initial_jobless_claims"):
        results["series"]["us_initial_jobless_claims"] = _fred_fetch_series(
            series_name="us_initial_jobless_claims",
            series_id="ICSA",
            start_date=start_date,
            end_date=end_date,
        )
    if _want("us_retail_sales"):
        results["series"]["us_retail_sales"] = _fred_fetch_series(
            series_name="us_retail_sales",
            series_id="RSAFS",
            start_date=start_date,
            end_date=end_date,
        )
    if _want("us_consumer_confidence_cb"):
        results["series"]["us_consumer_confidence_cb"] = _akshare_fetch_series(
            series_name="us_consumer_confidence_cb",
            fetcher_name="macro_usa_cb_consumer_confidence",
            source="akshare:us_macro:macro_usa_cb_consumer_confidence",
            start_date=start_date,
            end_date=end_date,
            date_col="日期",
            value_col="今值",
        )
    if _want("us_consumer_sentiment_michigan"):
        results["series"]["us_consumer_sentiment_michigan"] = _akshare_fetch_series(
            series_name="us_consumer_sentiment_michigan",
            fetcher_name="macro_usa_michigan_consumer_sentiment",
            source="akshare:us_macro:macro_usa_michigan_consumer_sentiment",
            start_date=start_date,
            end_date=end_date,
            date_col="日期",
            value_col="今值",
        )
    if _want("us_industrial_production"):
        results["series"]["us_industrial_production"] = _fred_fetch_series(
            series_name="us_industrial_production",
            series_id="INDPRO",
            start_date=start_date,
            end_date=end_date,
        )
    need_gdp_level = any(
        _want(k)
        for k in ("us_real_gdp_latest_quarter", "us_real_gdp_qoq_annualized", "us_real_gdp_yoy", "us_gdp_growth")
    )
    us_real_gdp_level = None
    if need_gdp_level:
        us_real_gdp_level = _fred_fetch_series(
            series_name="us_real_gdp_latest_quarter",
            series_id="GDPC1",
            start_date=start_date,
            end_date=end_date,
        )
    if _want("us_real_gdp_latest_quarter") and us_real_gdp_level is not None:
        results["series"]["us_real_gdp_latest_quarter"] = us_real_gdp_level
    if _want("us_real_gdp_qoq_annualized") and us_real_gdp_level is not None:
        results["series"]["us_real_gdp_qoq_annualized"] = _derived_growth_from_level_payload(
            us_real_gdp_level,
            series_name="us_real_gdp_qoq_annualized",
            periods=1,
            source="derived:fred:GDPC1:qoq_annualized",
            annualize=True,
        )
    if _want("us_real_gdp_yoy") and us_real_gdp_level is not None:
        results["series"]["us_real_gdp_yoy"] = _derived_growth_from_level_payload(
            us_real_gdp_level,
            series_name="us_real_gdp_yoy",
            periods=4,
            source="derived:fred:GDPC1:yoy",
        )
    if _want("us_core_cpi_yoy"):
        results["series"]["us_core_cpi_yoy"] = _fred_fetch_series(
        series_name="us_core_cpi_yoy",
        series_id="CPILFESL",
        start_date=start_date,
        end_date=end_date,
        transform="yoy",
    )
    if _want("us_breakeven_10y"):
        results["series"]["us_breakeven_10y"] = _fred_fetch_series(
        series_name="us_breakeven_10y",
        series_id="T10YIE",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_m2"):
        results["series"]["us_m2"] = _fred_fetch_series(
        series_name="us_m2",
        series_id="M2SL",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_central_bank_total_assets"):
        results["series"]["us_central_bank_total_assets"] = _fred_fetch_series(
        series_name="us_central_bank_total_assets",
        series_id="WALCL",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_real_rate_10y"):
        results["series"]["us_real_rate_10y"] = _fred_fetch_series(
        series_name="us_real_rate_10y",
        series_id="DFII10",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_treasury_2y"):
        results["series"]["us_treasury_2y"] = _fred_fetch_series(
        series_name="us_treasury_2y",
        series_id="DGS2",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_treasury_10y"):
        results["series"]["us_treasury_10y"] = _fred_fetch_series(
        series_name="us_treasury_10y",
        series_id="DGS10",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_term_spread_10y_2y"):
        results["series"]["us_term_spread_10y_2y"] = _fred_fetch_series(
        series_name="us_term_spread_10y_2y",
        series_id="T10Y2Y",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_term_spread_10y2y"):
        if "us_term_spread_10y_2y" in results["series"]:
            results["series"]["us_term_spread_10y2y"] = _clone_series_payload(
                results["series"]["us_term_spread_10y_2y"],
                series_name="us_term_spread_10y2y",
                source="alias:fred:T10Y2Y",
            )
        else:
            results["series"]["us_term_spread_10y2y"] = _fred_fetch_series(
                series_name="us_term_spread_10y2y",
                series_id="T10Y2Y",
                start_date=start_date,
                end_date=end_date,
            )
    if _want("us_corporate_bbb_oas"):
        results["series"]["us_corporate_bbb_oas"] = _fred_fetch_series(
        series_name="us_corporate_bbb_oas",
        series_id="BAMLC0A4CBBB",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_business_loan_delinquency_rate"):
        results["series"]["us_business_loan_delinquency_rate"] = _fred_fetch_series(
        series_name="us_business_loan_delinquency_rate",
        series_id="DRBLACBS",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_bank_loan_growth_yoy"):
        us_bank_loan_level = _fred_fetch_series(
            series_name="us_bank_loans_level",
            series_id="BUSLOANS",
            start_date=start_date,
            end_date=end_date,
        )
        results["series"]["us_bank_loan_growth_yoy"] = _derived_growth_from_level_payload(
            us_bank_loan_level,
            series_name="us_bank_loan_growth_yoy",
            periods=12,
            source="derived:fred:BUSLOANS:yoy",
        )
    if _want("us_equity_sp500"):
        results["series"]["us_equity_sp500"] = _fred_fetch_series(
        series_name="us_equity_sp500",
        series_id="SP500",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_vix"):
        results["series"]["us_vix"] = _fred_fetch_series(
        series_name="us_vix",
        series_id="VIXCLS",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("us_dollar_index_broad"):
        results["series"]["us_dollar_index_broad"] = _fred_fetch_series(
        series_name="us_dollar_index_broad",
        series_id="DTWEXBGS",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("commodity_wti_crude"):
        results["series"]["commodity_wti_crude"] = _fred_fetch_series(
        series_name="commodity_wti_crude",
        series_id="DCOILWTICO",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("commodity_copper"):
        results["series"]["commodity_copper"] = _fred_fetch_series(
        series_name="commodity_copper",
        series_id="PCOPPUSDM",
        start_date=start_date,
        end_date=end_date,
    )
    # Singapore focus (single-source per metric, mostly World Bank + selected FRED FX).
    if _want("sg_gdp_growth"):
        results["series"]["sg_gdp_growth"] = _world_bank_fetch_series(
        series_name="sg_gdp_growth",
        indicator="NY.GDP.MKTP.KD.ZG",
        country="SGP",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("sg_gdp_yoy"):
        if "sg_gdp_growth" in results["series"]:
            results["series"]["sg_gdp_yoy"] = _clone_series_payload(
                results["series"]["sg_gdp_growth"],
                series_name="sg_gdp_yoy",
                source="alias:world_bank:SGP:NY.GDP.MKTP.KD.ZG",
            )
        else:
            results["series"]["sg_gdp_yoy"] = _world_bank_fetch_series(
                series_name="sg_gdp_yoy",
                indicator="NY.GDP.MKTP.KD.ZG",
                country="SGP",
                start_date=start_date,
                end_date=end_date,
            )
    if _want("sg_inflation_cpi"):
        results["series"]["sg_inflation_cpi"] = _world_bank_fetch_series(
        series_name="sg_inflation_cpi",
        indicator="FP.CPI.TOTL.ZG",
        country="SGP",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("sg_cpi_yoy"):
        if "sg_inflation_cpi" in results["series"]:
            results["series"]["sg_cpi_yoy"] = _clone_series_payload(
                results["series"]["sg_inflation_cpi"],
                series_name="sg_cpi_yoy",
                source="alias:world_bank:SGP:FP.CPI.TOTL.ZG",
            )
        else:
            results["series"]["sg_cpi_yoy"] = _world_bank_fetch_series(
                series_name="sg_cpi_yoy",
                indicator="FP.CPI.TOTL.ZG",
                country="SGP",
                start_date=start_date,
                end_date=end_date,
            )
    if _want("sg_industrial_production_yoy"):
        results["series"]["sg_industrial_production_yoy"] = _macro_error_payload(
            "sg_industrial_production_yoy",
            "fixed_sources_unavailable",
            "unavailable in current fixed providers (AkShare/FRED/IMF/World Bank) for Singapore industrial production YoY",
        )
    if _want("sg_unemployment_rate"):
        results["series"]["sg_unemployment_rate"] = _world_bank_fetch_series(
        series_name="sg_unemployment_rate",
        indicator="SL.UEM.TOTL.ZS",
        country="SGP",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("sg_exports_growth"):
        results["series"]["sg_exports_growth"] = _world_bank_fetch_series(
        series_name="sg_exports_growth",
        indicator="NE.EXP.GNFS.KD.ZG",
        country="SGP",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("sg_imports_growth"):
        results["series"]["sg_imports_growth"] = _world_bank_fetch_series(
        series_name="sg_imports_growth",
        indicator="NE.IMP.GNFS.KD.ZG",
        country="SGP",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("sg_current_account_gdp"):
        results["series"]["sg_current_account_gdp"] = _world_bank_fetch_series(
        series_name="sg_current_account_gdp",
        indicator="BN.CAB.XOKA.GD.ZS",
        country="SGP",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("sg_real_interest_rate"):
        sg_real_interest_candidates = [
            _world_bank_fetch_series(
                series_name="sg_real_interest_rate",
                indicator="FR.INR.RINR",
                country="SGP",
                start_date=start_date,
                end_date=end_date,
            ),
            _world_bank_fetch_series(
                series_name="sg_real_interest_rate",
                indicator="FR.INR.LNDP",
                country="SGP",
                start_date=start_date,
                end_date=end_date,
            ),
        ]
        results["series"]["sg_real_interest_rate"] = _first_non_empty_payload(sg_real_interest_candidates)
    if _want("sg_broad_money_growth"):
        results["series"]["sg_broad_money_growth"] = _world_bank_fetch_series(
        series_name="sg_broad_money_growth",
        indicator="FM.LBL.BMNY.ZG",
        country="SGP",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("sg_usd_fx"):
        results["series"]["sg_usd_fx"] = _fred_fetch_series(
        series_name="sg_usd_fx",
        series_id="DEXSIUS",
        start_date=start_date,
        end_date=end_date,
    )
    # Singapore policy-rate proxy fixed to World Bank interest-rate series (best available stable public endpoint).
    if _want("sg_policy_rate"):
        sg_policy_candidates = [
            _world_bank_fetch_series(
                series_name="sg_policy_rate",
                indicator="FR.INR.DPST",
                country="SGP",
                start_date=start_date,
                end_date=end_date,
            ),
            _world_bank_fetch_series(
                series_name="sg_policy_rate",
                indicator="FR.INR.LEND",
                country="SGP",
                start_date=start_date,
                end_date=end_date,
            ),
        ]
        results["series"]["sg_policy_rate"] = _first_non_empty_payload(sg_policy_candidates)
    # Singapore 10Y yield fixed to MAS SGS original-maturity 10Y page.
    if _want("sg_government_bond_10y"):
        results["series"]["sg_government_bond_10y"] = _mas_fetch_sg_10y_yield(
        series_name="sg_government_bond_10y",
        start_date=start_date,
        end_date=end_date,
    )
    # Japan/Europe cross-impact liquidity (only high-impact overlap with CN/US/SG).
    if _want("jp_short_rate_3m"):
        results["series"]["jp_short_rate_3m"] = _fred_fetch_series(
        series_name="jp_short_rate_3m",
        series_id="IR3TIB01JPM156N",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("jp_government_bond_10y"):
        results["series"]["jp_government_bond_10y"] = _fred_fetch_series(
        series_name="jp_government_bond_10y",
        series_id="IRLTLT01JPM156N",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("jp_usd_fx"):
        results["series"]["jp_usd_fx"] = _fred_fetch_series(
        series_name="jp_usd_fx",
        series_id="DEXJPUS",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("eu_short_rate_3m"):
        results["series"]["eu_short_rate_3m"] = _fred_fetch_series(
        series_name="eu_short_rate_3m",
        series_id="IR3TIB01EZM156N",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("eu_government_bond_10y"):
        results["series"]["eu_government_bond_10y"] = _fred_fetch_series(
        series_name="eu_government_bond_10y",
        series_id="IRLTLT01EZM156N",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("eu_usd_fx"):
        results["series"]["eu_usd_fx"] = _fred_fetch_series(
        series_name="eu_usd_fx",
        series_id="DEXUSEU",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("eu_pmi_manufacturing"):
        results["series"]["eu_pmi_manufacturing"] = _akshare_fetch_series(
            series_name="eu_pmi_manufacturing",
            fetcher_name="macro_euro_manufacturing_pmi",
            source="akshare:eu_macro:macro_euro_manufacturing_pmi",
            start_date=start_date,
            end_date=end_date,
            date_col="日期",
            value_col="今值",
        )
    if _want("jp_policy_rate"):
        results["series"]["jp_policy_rate"] = _akshare_fetch_series(
            series_name="jp_policy_rate",
            fetcher_name="macro_japan_bank_rate",
            source="akshare:jp_macro:macro_japan_bank_rate",
            start_date=start_date,
            end_date=end_date,
            date_col="时间",
            value_col="现值",
        )
    # Backward-compatible alias: use the same canonical GDP YoY calculation source.
    if _want("us_gdp_growth"):
        if "us_real_gdp_yoy" not in results["series"] and us_real_gdp_level is not None:
            results["series"]["us_real_gdp_yoy"] = _derived_growth_from_level_payload(
                us_real_gdp_level,
                series_name="us_real_gdp_yoy",
                periods=4,
                source="derived:fred:GDPC1:yoy",
            )
        if "us_real_gdp_yoy" in results["series"]:
            results["series"]["us_gdp_growth"] = _clone_series_payload(
                results["series"]["us_real_gdp_yoy"],
                series_name="us_gdp_growth",
                source="alias:derived:fred:GDPC1:yoy",
            )
    if _want("us_real_interest_rate"):
        if "us_real_rate_10y" in results["series"]:
            results["series"]["us_real_interest_rate"] = _clone_series_payload(
                results["series"]["us_real_rate_10y"],
                series_name="us_real_interest_rate",
                source="alias:fred:REAINTRATREARAT10Y",
            )
        else:
            results["series"]["us_real_interest_rate"] = _fred_fetch_series(
                series_name="us_real_interest_rate",
                series_id="REAINTRATREARAT10Y",
                start_date=start_date,
                end_date=end_date,
            )
    # US PMI: AkShare official macro endpoints first; local manual file as fallback.
    if _want("us_pmi_manufacturing"):
        results["series"]["us_pmi_manufacturing"] = _load_us_pmi_series_with_fallback(
            series_name="us_pmi_manufacturing",
            ak_fetcher="macro_usa_ism_pmi",
            local_bucket="manufacturing",
            start_date=start_date,
            end_date=end_date,
        )
    if _want("us_pmi_services"):
        results["series"]["us_pmi_services"] = _load_us_pmi_series_with_fallback(
            series_name="us_pmi_services",
            ak_fetcher="macro_usa_ism_non_pmi",
            local_bucket="service",
            start_date=start_date,
            end_date=end_date,
        )
    # Global aggregates use World Bank WLD, which is stable for world-level totals/rates.
    if _want("world_gdp_growth"):
        results["series"]["world_gdp_growth"] = _world_bank_fetch_series(
        series_name="world_gdp_growth",
        indicator="NY.GDP.MKTP.KD.ZG",
        country="WLD",
        start_date=start_date,
        end_date=end_date,
    )
    if _want("world_inflation"):
        results["series"]["world_inflation"] = _world_bank_fetch_series(
        series_name="world_inflation",
        indicator="FP.CPI.TOTL.ZG",
        country="WLD",
        start_date=start_date,
        end_date=end_date,
    )
    # Reserve IMF fetches for Singapore/world extensions to avoid redundant US duplicates here.

    # China: AkShare wrappers for official China data sources (fixed)
    china_specs = [
        {
            "key": "pboc_policy_rate",
            "fetcher_name": "macro_bank_china_interest_rate",
            "source": "akshare:china_official:pboc",
            "date_col": "日期",
            "value_col": "今值",
        },
        {
            "key": "china_lpr_1y",
            "fetcher_name": "macro_china_lpr",
            "source": "akshare:china_official:pboc",
            "date_col": "TRADE_DATE",
            "value_col": "LPR1Y",
        },
        {"key": "china_shibor_3m", "fetcher_name": "macro_china_shibor_all", "source": "akshare:china_official:cfets"},
        {"key": "china_cpi_yoy", "fetcher_name": "macro_china_cpi_yearly", "source": "akshare:china_official:nbs"},
        {
            "key": "china_cpi_mom",
            "fetcher_name": "macro_china_cpi_monthly",
            "source": "akshare:china_official:nbs",
            "date_col": "日期",
            "value_col": "今值",
        },
        {"key": "china_ppi_yoy", "fetcher_name": "macro_china_ppi_yearly", "source": "akshare:china_official:nbs"},
        {"key": "china_gdp_yoy", "fetcher_name": "macro_china_gdp_yearly", "source": "akshare:china_official:nbs"},
        {"key": "china_pmi_manufacturing", "fetcher_name": "macro_china_pmi_yearly", "source": "akshare:china_official:nbs"},
        {
            "key": "china_caixin_pmi_manufacturing",
            "fetcher_name": "macro_china_cx_pmi_yearly",
            "source": "akshare:china_official:caixin",
            "date_col": "日期",
            "value_col": "今值",
        },
        {
            "key": "china_caixin_pmi_services",
            "fetcher_name": "macro_china_cx_services_pmi_yearly",
            "source": "akshare:china_official:caixin",
            "date_col": "日期",
            "value_col": "今值",
        },
        {"key": "china_pmi_non_manufacturing", "fetcher_name": "macro_china_non_man_pmi", "source": "akshare:china_official:nbs"},
        {
            "key": "china_urban_unemployment",
            "fetcher_name": "macro_china_urban_unemployment",
            "source": "akshare:china_official:nbs",
            "date_col": "date",
            "value_col": "value",
        },
        {"key": "china_m2_yoy", "fetcher_name": "macro_china_m2_yearly", "source": "akshare:china_official:pboc"},
        {
            "key": "china_social_financing",
            "fetcher_name": "macro_china_shrzgm",
            "source": "akshare:china_official:pboc",
            "date_col": "月份",
            "value_col": "社会融资规模增量",
        },
        {
            "key": "china_bank_financing",
            "fetcher_name": "macro_china_bank_financing",
            "source": "akshare:china_official:pboc",
            "date_col": "日期",
            "value_col": "最新值",
        },
        {"key": "china_central_bank_balance_sheet", "fetcher_name": "macro_china_central_bank_balance", "source": "akshare:china_official:pboc"},
        {"key": "china_bank_loan_growth", "fetcher_name": "macro_rmb_loan", "source": "akshare:china_official:pboc"},
        {"key": "china_real_estate_financing", "fetcher_name": "macro_china_real_estate", "source": "akshare:china_official:nbs"},
        {
            "key": "china_fixed_asset_investment_yoy",
            "fetcher_name": "macro_china_gdzctz",
            "source": "akshare:china_official:nbs",
            "date_col": "月份",
            "value_col": "同比增长",
        },
        {
            "key": "china_retail_sales_yoy",
            "fetcher_name": "macro_china_consumer_goods_retail",
            "source": "akshare:china_official:nbs",
            "date_col": "月份",
            "value_col": "同比增长",
        },
        {"key": "china_industrial_production_yoy", "fetcher_name": "macro_china_industrial_production_yoy", "source": "akshare:china_official:nbs"},
        {"key": "china_exports_yoy", "fetcher_name": "macro_china_exports_yoy", "source": "akshare:china_official:customs"},
        {"key": "china_imports_yoy", "fetcher_name": "macro_china_imports_yoy", "source": "akshare:china_official:customs"},
        {"key": "china_trade_balance", "fetcher_name": "macro_china_trade_balance", "source": "akshare:china_official:customs"},
        {"key": "china_fx_reserves", "fetcher_name": "macro_china_fx_reserves_yearly", "source": "akshare:china_official:safe"},
    ]
    for spec in china_specs:
        if not _want(str(spec["key"])):
            continue
        results["series"][spec["key"]] = _akshare_fetch_series(
            series_name=str(spec["key"]),
            fetcher_name=str(spec["fetcher_name"]),
            source=str(spec["source"]),
            start_date=start_date,
            end_date=end_date,
            date_col=spec.get("date_col"),
            value_col=spec.get("value_col"),
        )

    # PBOC policy rate fallback: if benchmark-rate series is stale, use LPR 1Y as operational policy proxy.
    if _want("pboc_policy_rate") and _want("china_lpr_1y"):
        pb = results["series"].get("pboc_policy_rate")
        lpr = results["series"].get("china_lpr_1y")
        pb_dt = _payload_latest_date(pb)
        lpr_dt = _payload_latest_date(lpr)
        if lpr_dt and (pb_dt is None or lpr_dt > pb_dt):
            results["series"]["pboc_policy_rate"] = _clone_series_payload(
                lpr if isinstance(lpr, dict) else _macro_error_payload("china_lpr_1y", "akshare:china_official:pboc", "lpr unavailable"),
                series_name="pboc_policy_rate",
                source="alias:akshare:china_official:pboc:lpr1y",
            )

    results["snapshot_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    results["source_policy"] = {
        "version": MACRO_SOURCE_POLICY_VERSION,
        "china": ["akshare:china_official"],
        "international": ["fred", "imf_datamapper", "world_bank", "akshare:us_macro_ism"],
        "manual_overrides": ["local:us_pmi_json"],
    }
    results["latest"] = {
        key: val.get("latest", {}).get("value") if isinstance(val, dict) and val.get("latest") else None
        for key, val in results["series"].items()
    }
    return results


MACRO_DIMENSION_MAP: dict[str, str] = {
    # Growth
    "us_real_gdp_latest_quarter": "growth",
    "us_real_gdp_qoq_annualized": "growth",
    "us_real_gdp_yoy": "growth",
    "us_gdp_growth": "growth",
    "us_industrial_production": "growth",
    "us_retail_sales": "growth",
    "us_consumer_confidence_cb": "growth",
    "us_consumer_sentiment_michigan": "growth",
    "us_non_farm_payrolls": "growth",
    "china_gdp_yoy": "growth",
    "china_pmi_manufacturing": "growth",
    "china_caixin_pmi_manufacturing": "growth",
    "china_caixin_pmi_services": "growth",
    "china_pmi_non_manufacturing": "growth",
    "china_industrial_production_yoy": "growth",
    "china_retail_sales_yoy": "growth",
    "china_fixed_asset_investment_yoy": "growth",
    "china_exports_yoy": "growth",
    "china_imports_yoy": "growth",
    "sg_gdp_growth": "growth",
    "sg_gdp_yoy": "growth",
    "sg_industrial_production_yoy": "growth",
    "sg_exports_growth": "growth",
    "sg_imports_growth": "growth",
    # Inflation
    "us_cpi_yoy": "inflation",
    "us_core_cpi_yoy": "inflation",
    "us_core_pce_price": "inflation",
    "us_breakeven_10y": "inflation",
    "china_cpi_yoy": "inflation",
    "china_cpi_mom": "inflation",
    "china_ppi_yoy": "inflation",
    "sg_inflation_cpi": "inflation",
    "sg_cpi_yoy": "inflation",
    "world_inflation": "inflation",
    # Liquidity
    "fed_policy_rate": "liquidity",
    "pboc_policy_rate": "liquidity",
    "sofr": "liquidity",
    "us_m2": "liquidity",
    "china_m2_yoy": "liquidity",
    "sg_broad_money_growth": "liquidity",
    "us_central_bank_total_assets": "liquidity",
    "china_central_bank_balance_sheet": "liquidity",
    "us_real_rate_10y": "liquidity",
    "us_real_interest_rate": "liquidity",
    "sg_real_interest_rate": "liquidity",
    "us_treasury_2y": "liquidity",
    "us_treasury_10y": "liquidity",
    "us_term_spread_10y_2y": "liquidity",
    "us_term_spread_10y2y": "liquidity",
    "china_shibor_3m": "liquidity",
    "china_lpr_1y": "liquidity",
    "sg_policy_rate": "liquidity",
    "sg_government_bond_10y": "liquidity",
    "jp_short_rate_3m": "liquidity",
    "jp_government_bond_10y": "liquidity",
    "jp_policy_rate": "liquidity",
    "eu_short_rate_3m": "liquidity",
    "eu_government_bond_10y": "liquidity",
    # Credit
    "us_corporate_bbb_oas": "credit",
    "us_business_loan_delinquency_rate": "credit",
    "us_bank_loan_growth_yoy": "credit",
    "china_social_financing": "credit",
    "china_bank_financing": "credit",
    "china_bank_loan_growth": "credit",
    "china_real_estate_financing": "credit",
    # Market
    "us_equity_sp500": "market_feedback",
    "us_vix": "market_feedback",
    "us_dollar_index_broad": "market_feedback",
    "commodity_wti_crude": "market_feedback",
    "commodity_copper": "market_feedback",
    "sg_usd_fx": "market_feedback",
    "jp_usd_fx": "market_feedback",
    "eu_usd_fx": "market_feedback",
    "eu_pmi_manufacturing": "growth",
}


def _macro_country_from_key(key: str) -> str:
    if key.startswith("us_") or key in {"fed_policy_rate", "sofr"}:
        return "US"
    if key.startswith("china_") or key == "pboc_policy_rate":
        return "CN"
    if key.startswith("sg_"):
        return "SG"
    if key.startswith("jp_"):
        return "JP"
    if key.startswith("eu_"):
        return "EU"
    if key.startswith("world_") or key.startswith("commodity_"):
        return "GLOBAL"
    return "GLOBAL"


def _macro_frequency(dates: list[datetime]) -> str:
    if len(dates) < 2:
        return "unknown"
    diffs = sorted(
        (
            (dates[i] - dates[i - 1]).days
            for i in range(1, len(dates))
            if (dates[i] - dates[i - 1]).days > 0
        )
    )
    if not diffs:
        return "unknown"
    median = diffs[len(diffs) // 2]
    if median <= 10:
        return "daily"
    if median <= 45:
        return "monthly"
    if median <= 120:
        return "quarterly"
    return "annual"


def _macro_steps_for_frequency(freq: str) -> dict[str, int]:
    if freq == "daily":
        return {"mom": 21, "qoq": 63, "yoy": 252}
    if freq == "monthly":
        return {"mom": 1, "qoq": 3, "yoy": 12}
    if freq == "quarterly":
        return {"mom": 1, "qoq": 1, "yoy": 4}
    if freq == "annual":
        return {"mom": 1, "qoq": 1, "yoy": 1}
    return {}


def _macro_change(values: list[float], steps: int) -> dict[str, float] | None:
    if steps <= 0 or len(values) <= steps:
        return None
    curr = values[-1]
    prev = values[-1 - steps]
    delta = curr - prev
    pct = None
    if prev != 0:
        pct = delta / abs(prev)
    out: dict[str, float] = {"current": curr, "previous": prev, "delta": delta}
    if pct is not None:
        out["delta_pct"] = pct
    return out


def _macro_trend(values: list[float], lookback: int = 3) -> str:
    if len(values) < lookback + 1:
        return "insufficient"
    start = values[-(lookback + 1)]
    end = values[-1]
    base = max(abs(start), 1e-9)
    rel = (end - start) / base
    if rel > 0.01:
        return "up"
    if rel < -0.01:
        return "down"
    return "flat"


def structure_macro_indicators(macro_payload: dict[str, Any]) -> dict[str, Any]:
    """Transform raw macro payload into LLM/analysis friendly structured format."""
    series = macro_payload.get("series", {}) if isinstance(macro_payload, dict) else {}
    cards: dict[str, dict[str, Any]] = {}
    chart_long: list[dict[str, Any]] = []
    dimension_buckets: dict[str, list[str]] = {
        "growth": [],
        "inflation": [],
        "liquidity": [],
        "credit": [],
        "market_feedback": [],
        "other": [],
    }

    for key, payload in series.items():
        if not isinstance(payload, dict):
            continue
        observations = payload.get("observations", [])
        parsed: list[tuple[datetime, float]] = []
        if isinstance(observations, list):
            for row in observations:
                if not isinstance(row, dict):
                    continue
                d = row.get("date")
                v = row.get("value")
                if not isinstance(d, str) or not isinstance(v, int | float):
                    continue
                try:
                    parsed.append((datetime.strptime(d[:10], "%Y-%m-%d"), float(v)))
                except ValueError:
                    continue
        parsed.sort(key=lambda x: x[0])
        dates = [d for d, _ in parsed]
        values = [v for _, v in parsed]
        freq = _macro_frequency(dates)
        steps = _macro_steps_for_frequency(freq)
        latest_value = values[-1] if values else None
        latest_date = dates[-1].strftime("%Y-%m-%d") if dates else None
        mom = _macro_change(values, steps.get("mom", 0)) if steps else None
        qoq = _macro_change(values, steps.get("qoq", 0)) if steps else None
        yoy = _macro_change(values, steps.get("yoy", 0)) if steps else None
        trend_short = _macro_trend(values, lookback=3)
        trend_medium = _macro_trend(values, lookback=6)
        volatility = None
        if len(values) >= 6:
            volatility = float(pd.Series(values[-24:]).pct_change().dropna().std())

        dimension = MACRO_DIMENSION_MAP.get(key, "other")
        country = _macro_country_from_key(key)
        card = {
            "key": key,
            "dimension": dimension,
            "country": country,
            "source": payload.get("source"),
            "frequency": freq,
            "latest_value": latest_value,
            "latest_date": latest_date,
            "trend_short": trend_short,
            "trend_medium": trend_medium,
            "mom": mom,
            "qoq": qoq,
            "yoy": yoy,
            "volatility": volatility,
            "obs_count": len(values),
            "error": payload.get("error"),
        }
        cards[key] = card
        dimension_buckets.setdefault(dimension, []).append(key)

        for d, v in parsed:
            chart_long.append(
                {
                    "key": key,
                    "date": d.strftime("%Y-%m-%d"),
                    "value": v,
                    "dimension": dimension,
                    "country": country,
                    "source": payload.get("source"),
                }
            )

    coverage = {
        "total_metrics": len(cards),
        "ok_metrics": sum(1 for c in cards.values() if c.get("error") is None and c.get("obs_count", 0) > 0),
        "error_metrics": sum(1 for c in cards.values() if c.get("error")),
        "empty_metrics": sum(1 for c in cards.values() if c.get("obs_count", 0) == 0),
    }

    return {
        "meta": {
            "snapshot_at": macro_payload.get("snapshot_at") if isinstance(macro_payload, dict) else None,
            "source_policy": macro_payload.get("source_policy") if isinstance(macro_payload, dict) else None,
            "coverage": coverage,
        },
        "dimensions": {
            dim: [cards[k] for k in keys]
            for dim, keys in dimension_buckets.items()
        },
        "metrics": cards,
        "chart_data": {
            "long": chart_long,
        },
    }


def get_macro_indicators_structured(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    raw = get_macro_indicators(start_date=start_date, end_date=end_date)
    return structure_macro_indicators(raw)


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

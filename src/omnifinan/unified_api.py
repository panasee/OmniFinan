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

from pyomnix.consts import OMNIX_PATH

# Import data models (assuming they are defined correctly AND reflect the original structure)
from pyomnix.financial.data_models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    MarketType,
    Price,
)

# Assuming omnix_logger and data_models are in the same package structure
# If not, adjust the import paths accordingly
from pyomnix.omnix_logger import get_logger

# 获取当前模块的路径

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
        r"\.(SH|SZ|HK|O|N|L|DE|PA|AS|MI|VX|SW|CO|HE|ST|OL|IC|KS|KQ|TWO|TW|BK|CR|SA|TL|JK|NZ|AX|IS|TO|NE|BR|VI)$",
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

    # China A-shares: 6 digits
    if re.match(r"^\d{6}$", normalized_ticker):
        # Shanghai (SH) stocks start with 6 or 9
        if normalized_ticker.startswith(("6", "9")):
            logger.debug(f"Detected Market: CHINA (SH) for {normalized_ticker}")
            return MarketType.CHINA_SH
        # Shenzhen (SZ) stocks start with 0, 3, or 2
        elif normalized_ticker.startswith(("0", "3", "2", "4", "8")):
            logger.debug(f"Detected Market: CHINA (SZ) for {normalized_ticker}")
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
    """Converts various date inputs to 'YYYY-MM-DD' string, returning None on failure."""
    if pd.isna(date_input):
        return None
    if isinstance(date_input, str):
        try:
            # Handle YYYYMMDD
            if len(date_input) == 8 and date_input.isdigit():
                return datetime.strptime(date_input, "%Y%m%d").strftime("%Y-%m-%d")
            # Handle other common formats parsable by pandas
            return pd.to_datetime(date_input).strftime("%Y-%m-%d")
        except Exception:
            logger.debug(f"Could not parse date string: {date_input}")
            return date_input  # Return original string if parsing fails as a fallback? Or None? Let's return None for consistency.
            # return None
    elif isinstance(date_input, (date, pd.Timestamp)):
        return date_input.strftime("%Y-%m-%d")
    logger.debug(f"Could not format date input (type {type(date_input)}): {date_input}")
    return None  # Fallback


def _get_value_from_row(
    row: pd.Series, key: str, converter: callable, default=None
) -> Any:
    """Gets value from series, handles missing key, calls converter."""
    if key in row.index:
        return converter(row.get(key))
    return default


# --- Unified API Functions ---


def get_prices(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    adjustment: Literal["qfq", "hfq", ""] = "",
    api: Literal["sina", "em"] = "sina",
) -> list[Price]:
    """
    Fetch price data compatible with the original Price model structure.
    Includes fields like amount, amplitude, turnover, setting them to None if
    unavailable for the market (e.g., US).

    Args:
        ticker: The ticker symbol (e.g., "AAPL", "00700", "600519")
        start_date: Start date in "YYYY-MM-DD" format. Defaults to 1 year ago.
        end_date: End date in "YYYY-MM-DD" format. Defaults to yesterday.
        adjustment: Price adjustment type. "qfq" for 前复权, "hfq" for 后复权, "" for no adjustment.
        api: Data source, "sina" for Sina, "em" for East Money.

    Returns:
        List[Price]: A list of price data objects matching the original structure.
    """
    normalized_ticker = normalize_ticker(ticker)
    market = detect_market(normalized_ticker)

    end_date_ak = _format_date(end_date, default_delta_days=0)
    if start_date:
        start_date_ak = _format_date(start_date)
    else:
        end_dt = datetime.strptime(end_date_ak, "%Y%m%d")
        start_dt = end_dt - timedelta(days=365)
        start_date_ak = start_dt.strftime("%Y%m%d")

    logger.info(
        f"Getting price history for {normalized_ticker} ({market.name}) from {start_date_ak} to {end_date_ak}"
    )

    prices_list = []
    try:
        df = pd.DataFrame()
        if api == "sina":
            if market == MarketType.US:
                # For US stocks, use stock_us_daily function which is more reliable
                df = ak.stock_us_daily(
                    symbol=normalized_ticker,
                    adjust=adjustment,
                )

            elif market == MarketType.HK:
                # For HK stocks
                df = ak.stock_hk_daily(
                    symbol=normalized_ticker,
                    adjust=adjustment,
                )
            elif market in [
                MarketType.CHINA_SZ,
                MarketType.CHINA_SH,
                MarketType.CHINA_BJ,
            ]:
                df = ak.stock_zh_a_daily(
                    symbol=f"{market.value}{normalized_ticker}",
                    start_date=start_date_ak,
                    end_date=end_date_ak,
                    adjust=adjustment,
                )
            # Rename columns to match our expected format
            df = df.rename(
                columns={
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )
            logger.info(f"Successfully fetched stock data for {normalized_ticker}")
        elif api == "em":
            if market == MarketType.US:
                logger.warning(
                    "api for em of us stocks are broken now, check new version of akshare"
                )
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
            else:  # MarketType.CHINA
                # For Chinese A-shares
                df = ak.stock_zh_a_hist(
                    symbol=normalized_ticker,
                    period="daily",
                    start_date=start_date_ak,
                    end_date=end_date_ak,
                    adjust=adjustment,
                )
            # Rename essential US columns - others will be None
            df = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_change",
                    "涨跌额": "change_amount",
                    "换手率": "turnover",
                }
            )
            logger.info(f"Successfully fetched stock data for {normalized_ticker}")

        if df is None or df.empty:
            logger.warning(
                f"No price data found for {normalized_ticker} in the specified range."
            )
            return []

        # Add missing columns that might be expected in the Price model
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

        # Convert DataFrame to Price objects, ensuring all model fields exist (as None if needed)
        for _, row in df.iterrows():
            # Use _get_value_from_row for safe extraction and conversion
            price_data = Price(
                open=_get_value_from_row(row, "open", _safe_float_convert),
                high=_get_value_from_row(row, "high", _safe_float_convert),
                low=_get_value_from_row(row, "low", _safe_float_convert),
                close=_get_value_from_row(row, "close", _safe_float_convert),
                volume=_get_value_from_row(row, "volume", _safe_int_convert),
                time=_get_value_from_row(
                    row, "date", _format_output_date
                ),  # Format date string
                # Fields potentially missing in US/HK data:
                amount=_get_value_from_row(row, "amount", _safe_float_convert),
                amplitude=_get_value_from_row(row, "amplitude", _safe_float_convert),
                pct_change=_get_value_from_row(row, "pct_change", _safe_float_convert),
                change_amount=_get_value_from_row(
                    row, "change_amount", _safe_float_convert
                ),
                turnover=_get_value_from_row(row, "turnover", _safe_float_convert),
                # Market field
                market=market,
            )
            prices_list.append(price_data)

        logger.info(
            f"Successfully fetched {len(prices_list)} price records for {normalized_ticker}"
        )

    except Exception as e:
        logger.error(
            f"Error getting price history for {normalized_ticker}: {e}", exc_info=True
        )
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
    if manual_input is not None:
        metric_data = manual_input

    else:
        normalized_ticker = normalize_ticker(ticker)
        market = detect_market(normalized_ticker)
        logger.info(
            f"Getting financial metrics for {normalized_ticker} ({market.name})..."
        )

        # Initialize all fields from the original model with None or defaults
        metric_data = {
            "ticker": normalized_ticker,
            "report_period": end_date
            if end_date
            else datetime.now().strftime("%Y-%m-%d"),
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
                    logger.warning(
                        f"Could not fetch US quote data for {normalized_ticker}"
                    )
            elif market == MarketType.HK:
                q_df = ak.stock_individual_basic_info_hk_xq(symbol=normalized_ticker)
                if not q_df.empty:
                    quote_data = q_df.iloc[0]
                else:
                    logger.warning(
                        f"Could not fetch HK quote data for {normalized_ticker}"
                    )
            else:  # MarketType.CHINA
                # Calculate Market Cap
                latest_price = None
                prices = get_prices(
                    normalized_ticker, end_date=datetime.now().strftime("%Y-%m-%d")
                )
                if prices:
                    latest_price = prices[-1].close

                total_shares = None
                try:
                    info_df = ak.stock_individual_info_em(
                        symbol=normalized_ticker, timeout=20
                    )
                    total_shares_series = info_df[info_df["item"] == "总股本"]["value"]
                    if not total_shares_series.empty:
                        value_str = total_shares_series.iloc[0]
                        num_part_list = re.findall(r"[\d\.]+", value_str)
                        if num_part_list:
                            num_part = num_part_list[0]
                            multiplier = (
                                1e8
                                if "亿" in value_str
                                else 1e4
                                if "万" in value_str
                                else 1
                            )
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
                        ind_df["日期"] = pd.to_datetime(ind_df["日期"])
                        indicator_data = ind_df.sort_values(
                            "日期", ascending=False
                        ).iloc[0]
                except Exception as e_ind:
                    logger.warning(
                        f"Failed fetching CN financial indicators for {normalized_ticker}: {e_ind}"
                    )

            # --- Populate metric_data using fetched info ---

            # Market Cap
            if market == MarketType.US:
                metric_data["market_cap"] = _safe_float_convert(
                    quote_data.get("market_capital")
                )
            elif market == MarketType.HK:
                metric_data["market_cap"] = _safe_float_convert(
                    quote_data.get("market_value")
                )
            else:  # CN
                metric_data["market_cap"] = calculated_market_cap  # Already calculated

            # PE Ratio
            if market == MarketType.US or market == MarketType.HK:
                metric_data["price_to_earnings_ratio"] = _safe_float_convert(
                    quote_data.get("pe_ratio")
                )
            elif not indicator_data.empty:  # CN from indicators
                metric_data["price_to_earnings_ratio"] = _safe_float_convert(
                    indicator_data.get("市盈率PE(TTM)")
                )

            # PB Ratio (Only available from CN indicators in this setup)
            if (
                market
                in [MarketType.CHINA_SZ, MarketType.CHINA_SH, MarketType.CHINA_BJ]
                and not indicator_data.empty
            ):
                metric_data["price_to_book_ratio"] = _safe_float_convert(
                    indicator_data.get("市净率PB(MRQ)")
                )

            # Update report_period if CN indicators provided a date
            if (
                market
                in [MarketType.CHINA_SZ, MarketType.CHINA_SH, MarketType.CHINA_BJ]
                and not indicator_data.empty
            ):
                indicator_date = _format_output_date(indicator_data.get("日期"))
                if indicator_date:
                    metric_data["report_period"] = indicator_date

            # CN Specific Metrics from Indicators
            if (
                market
                in [MarketType.CHINA_SZ, MarketType.CHINA_SH, MarketType.CHINA_BJ]
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
                    "净资产收益率ROE(加权)", is_percent=True
                )
                metric_data["net_margin"] = _parse_indicator(
                    "销售净利率(%)", is_percent=True
                )
                metric_data["operating_margin"] = _parse_indicator(
                    "营业利润率(%)", is_percent=True
                )
                metric_data["revenue_growth"] = _parse_indicator(
                    "主营业务收入增长率(%)", is_percent=True
                )
                metric_data["earnings_growth"] = _parse_indicator(
                    "净利润增长率(%)", is_percent=True
                )
                metric_data["current_ratio"] = _parse_indicator("流动比率")
                metric_data["debt_to_equity"] = _parse_indicator(
                    "资产负债率(%)", is_percent=True
                )  # Note: D/A ratio
                metric_data["earnings_per_share"] = _parse_indicator("基本每股收益(元)")
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

    metrics_list = FinancialMetrics(**metric_data)

    return metrics_list


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
    normalized_ticker = normalize_ticker(ticker)
    market = detect_market(normalized_ticker)
    logger.info(
        f"Getting financial statement data for {normalized_ticker} ({market.name})"
    )

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
                    report_dates.update(
                        fetched_data["income"][date_col_name].astype(str).tolist()
                    )
                    logger.info(
                        f"Using date column '{date_col_name}' for report periods."
                    )
                    break
            if not report_dates:
                logger.warning(
                    "Could not identify a suitable date column in the income statement."
                )

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
                                pd.to_datetime(df[date_col_name]).dt.strftime(
                                    "%Y-%m-%d"
                                )
                                == report_date_str
                            )
                        if "STD_ITEM_NAME" in df.columns:
                            tmp = df[mask][["STD_ITEM_NAME", "AMOUNT"]].transpose()
                            row_series = tmp.rename(columns=tmp.iloc[0]).drop(
                                index="STD_ITEM_NAME"
                            )
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
                "USD"
                if market == MarketType.US
                else "HKD"
                if market == MarketType.HK
                else "CNY"
            )

            # Base items (handle potential missing rows)
            ni = _get_value_from_row(
                income_row, "净利润", _safe_float_convert
            )  # Or "归属于母公司所有者的净利润"
            if ni is None:
                ni = _get_value_from_row(
                    income_row, "股东应占溢利", _safe_float_convert
                )
            rev = _get_value_from_row(income_row, "营业总收入", _safe_float_convert)
            if rev is None:
                rev = _get_value_from_row(income_row, "营运收入", _safe_float_convert)
            op = _get_value_from_row(income_row, "营业利润", _safe_float_convert)
            if op is None:
                op = _get_value_from_row(income_row, "经营溢利", _safe_float_convert)
            ca = _get_value_from_row(balance_row, "流动资产合计", _safe_float_convert)
            cl = _get_value_from_row(balance_row, "流动负债", _safe_float_convert)
            if cl is None:
                cl = _get_value_from_row(
                    balance_row, "流动负债合计", _safe_float_convert
                )
            cfo = _get_value_from_row(
                cashflow_row, "经营活动产生的现金流量净额", _safe_float_convert
            )
            if cfo is None:
                cfo = _get_value_from_row(
                    cashflow_row, "经营业务现金净额", _safe_float_convert
                )
            dep_amort = _get_value_from_row(
                cashflow_row,
                "固定资产折旧、油气资产折耗、生产性生物资产折旧",
                _safe_float_convert,
            )  # Or sum of relevant lines
            if dep_amort is None:
                dep_amort = _get_value_from_row(
                    cashflow_row, "减值及拨备", _safe_float_convert
                )
            capex_paid = _get_value_from_row(
                cashflow_row,
                "购建固定资产、无形资产和其他长期资产所支付的现金",
                _safe_float_convert,
            )
            if capex_paid is None:
                capex_paid = _get_value_from_row(
                    cashflow_row, "购建无形资产及其他资产", _safe_float_convert
                )

            item_data["net_income"] = ni
            item_data["operating_revenue"] = rev
            item_data["operating_profit"] = op
            item_data["depreciation_and_amortization"] = dep_amort

            # Calculated items
            item_data["working_capital"] = (
                (ca - cl) if ca is not None and cl is not None else None
            )
            item_data["capital_expenditure"] = (
                abs(capex_paid) if capex_paid is not None else None
            )
            item_data["free_cash_flow"] = (
                (cfo - item_data["capital_expenditure"])
                if cfo is not None and item_data["capital_expenditure"] is not None
                else None
            )

            try:
                results.append(
                    LineItem(**item_data)
                )  # Pass the dict containing potential Nones
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
        date_col = "交易日期"  # Verify name
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
                logger.warning(
                    f"Could not perform date filtering on insider trades: {date_e}"
                )
        else:
            logger.warning(f"Date column '{date_col}' not found for insider trades.")

        df = df.head(limit)

        # Convert DataFrame rows to InsiderTrade objects
        for _, row in df.iterrows():
            trade_data = InsiderTrade(
                ticker=normalized_ticker,
                # Verify column names:
                insider_name=_get_value_from_row(
                    row, "内部人名称", str, default=""
                ),  # Assuming string, provide default
                insider_relation=_get_value_from_row(row, "职务", str, default=""),
                transaction_date=_get_value_from_row(
                    row, date_col, _format_output_date
                ),
                transaction_type=_get_value_from_row(
                    row, "交易", str, default=""
                ),  # e.g., 'Sale', 'Buy'
                shares_transacted=_get_value_from_row(
                    row, "交易股数", _safe_int_convert
                ),
                shares_held_after=_get_value_from_row(
                    row, "持有股数", _safe_int_convert
                ),
                trade_price=_get_value_from_row(
                    row, "交易价格", _safe_float_convert
                ),  # Might be avg price
                market=MarketType.US,
            )
            trades.append(trade_data)

        logger.info(
            f"Successfully fetched {len(trades)} insider trade records for {normalized_ticker}"
        )

    except Exception as e:
        logger.error(
            f"Error getting insider trades for {normalized_ticker}: {e}", exc_info=True
        )
        return []

    return trades


def get_stock_news(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 10,
) -> list[CompanyNews]:
    """获取并处理个股新闻

    Args:
        symbol (str): 股票代码，如 "300059"
        start_date (str, optional): 开始日期，格式为"YYYY-MM-DD"。默认为None，表示不限制开始日期。
        end_date (str, optional): 结束日期，格式为"YYYY-MM-DD"。默认为None，表示不限制结束日期。(不包含结束日期)
        max_news (int, optional): 获取的新闻条数，默认为10条。最大支持100条

    Returns:
        list: 新闻列表，每条新闻包含标题、内容、发布时间等信息
    """
    market = detect_market(symbol)

    def limit_time(input_date, start_date, end_date):
        if start_date and input_date < start_date:
            return False
        if end_date and input_date > end_date:
            return False
        return True

    def filter_time_df(df, start_date, end_date):
        df = df[df["发布时间"].apply(limit_time, args=(start_date, end_date))]
        return df

    # 设置pandas显示选项，确保显示完整内容
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)

    # 限制最大新闻条数
    limit = min(limit, 100)

    # 获取当前日期
    today = datetime.now().strftime("%Y-%m-%d")

    # 构建新闻文件路径
    news_dir = OMNIX_PATH / "financial" / "stock_news"
    print(f"新闻保存目录: {news_dir}")

    # 确保目录存在
    try:
        news_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"成功创建或确认目录存在: {news_dir}")
    except Exception as e:
        logger.error(f"创建目录失败: {e}")
        return []

    news_file = news_dir / f"{symbol}_news.json"
    logger.info(f"新闻文件路径: {news_file}")

    # 检查是否需要更新新闻
    need_update = True
    if news_file.exists():
        try:
            with open(news_file, encoding="utf-8") as f:
                data = json.load(f)
                if data.get("date") == today:
                    cached_news = data.get("news", [])
                    if len(cached_news) >= limit:
                        logger.info(f"使用缓存的新闻数据: {news_file}")
                        return cached_news[:limit]
                    else:
                        logger.info(
                            f"缓存的新闻数量({len(cached_news)})不足，需要获取更多新闻({limit}条)"
                        )
        except Exception as e:
            print(f"读取缓存文件失败: {e}")

    logger.info(f"开始获取{symbol}的新闻数据...")

    try:
        # 获取新闻列表
        news_df = ak.stock_news_em(symbol=symbol)
        if news_df is None or len(news_df) == 0:
            logger.warning(f"未获取到{symbol}的新闻数据")
            return []

        logger.info(f"成功获取到{len(news_df)}条新闻")
        news_df = filter_time_df(news_df, start_date, end_date)

        # 实际可获取的新闻数量
        available_news_count = len(news_df)
        if available_news_count < limit:
            logger.warning(
                f"警告：实际可获取的新闻数量({available_news_count})少于请求的数量({limit})"
            )
            limit = available_news_count

        # 获取指定条数的新闻（考虑到可能有些新闻内容为空，多获取50%）
        news_list = []
        for _, row in news_df.head(int(limit * 1.5)).iterrows():
            try:
                # 获取新闻内容
                content = (
                    row["新闻内容"]
                    if "新闻内容" in row and not pd.isna(row["新闻内容"])
                    else ""
                )
                if not content:
                    content = row["新闻标题"]

                # 只去除首尾空白字符
                content = content.strip()
                if len(content) < 10:  # 内容太短的跳过
                    continue

                # 获取关键词
                keyword = (
                    row["关键词"]
                    if "关键词" in row and not pd.isna(row["关键词"])
                    else ""
                )

                # 添加新闻
                news_item = {
                    "ticker": symbol,
                    "title": row["新闻标题"].strip(),
                    "content": content,
                    "date": row["发布时间"],
                    "source": row["文章来源"].strip(),
                    "url": row["新闻链接"].strip(),
                    "keyword": keyword.strip(),
                    "market": market,
                }
                news_list.append(news_item)
                logger.debug(f"成功添加新闻: {news_item['title']}")

            except Exception as e:
                print(f"处理单条新闻时出错: {e}")
                continue

        # 按发布时间排序
        news_list.sort(key=lambda x: x["date"], reverse=True)

        # 只保留指定条数的有效新闻
        news_list = news_list[:limit]

        # 保存到文件
        try:
            save_data = {"date": today, "news": news_list}
            with open(news_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"成功保存{len(news_list)}条新闻到文件: {news_file}")
        except Exception as e:
            print(f"保存新闻数据到文件时出错: {e}")

        return [CompanyNews(**news) for news in news_list]

    except Exception as e:
        print(f"获取新闻数据时出错: {e}")
        return []


def get_market_cap(
    ticker: str,
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
    logger.info(
        f"Fetching latest market cap for {normalized_ticker} ({market.name})..."
    )

    market_cap = None
    try:
        if market == MarketType.US:
            quote_df = ak.stock_quote_us_sina(symbol=normalized_ticker)
            if not quote_df.empty:
                market_cap = _safe_float_convert(quote_df.iloc[0].get("market_capital"))
            else:
                logger.warning(
                    f"Could not fetch US quote data for market cap: {normalized_ticker}"
                )

        elif market == MarketType.HK:
            quote_df = ak.stock_quote_hk_sina(symbol=normalized_ticker)
            if not quote_df.empty:
                market_cap = _safe_float_convert(quote_df.iloc[0].get("market_value"))
            else:
                logger.warning(
                    f"Could not fetch HK quote data for market cap: {normalized_ticker}"
                )

        else:  # MarketType.CHINA
            latest_price = None
            prices_list = get_prices(
                normalized_ticker, end_date=datetime.now().strftime("%Y-%m-%d")
            )
            if prices_list:
                latest_price = prices_list[-1].close
            else:
                logger.warning(
                    f"Could not get latest price for CN {normalized_ticker} for market cap calc."
                )

            total_shares = None
            try:
                info_df = ak.stock_individual_info_em(symbol=normalized_ticker)
                total_shares_series = info_df[info_df["item"] == "总股本"]["value"]
                if not total_shares_series.empty:
                    value_str = total_shares_series.iloc[0]
                    num_part_list = re.findall(r"[\d\.]+", value_str)
                    if num_part_list:
                        num_part = num_part_list[0]
                        multiplier = (
                            1e8
                            if "亿" in value_str
                            else 1e4
                            if "万" in value_str
                            else 1
                        )
                        total_shares = float(num_part) * multiplier
                    else:
                        logger.warning(
                            f"Could not parse number from '总股本' value: {value_str}"
                        )
                else:
                    logger.warning(
                        f"Could not find '总股本' in info for CN {normalized_ticker}"
                    )
            except Exception as e_info:
                logger.warning(
                    f"Failed to get total shares for CN {normalized_ticker}: {e_info}"
                )

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
        logger.error(
            f"Error getting market cap for {normalized_ticker}: {e}", exc_info=True
        )
        return None


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
        logger.error(
            f"Date column '{date_col}' not found in Price data for DataFrame conversion."
        )
        return df  # Return df without index

    try:
        # Ensure the date column is suitable for conversion before setting index
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])  # Drop rows where date conversion failed
        if not df.empty:
            df = df.set_index(pd.DatetimeIndex(df[date_col]))
            # Optionally drop the original column if it's now the index name or redundant
            if df.index.name == date_col:
                pass  # Index name is already 'time'
            # else: # Drop if index name is different (e.g., 'Date') or keep if needed
            #     df = df.drop(columns=[date_col])
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


def get_price_data(
    ticker: str, start_date: str | None = None, end_date: str | None = None
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
    prices = get_prices(ticker, start_date, end_date)
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
        aapl_prices = get_prices(
            aapl_ticker, start_date="2024-01-01", end_date="2024-04-01"
        )
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
        tencent_prices = get_prices(
            tencent_ticker, start_date="2024-01-01", end_date="2024-04-01"
        )
        if tencent_prices:
            print(f"Tencent Prices (last 2): {tencent_prices[-2:]}")
            tencent_df = get_price_data(
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
        moutai_prices = get_prices(
            moutai_ticker, start_date="2024-01-01", end_date="2024-04-01"
        )
        if moutai_prices:
            print(f"Moutai Prices (last 2): {moutai_prices[-2:]}")
            moutai_df = get_price_data(
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
        moutai_items = search_line_items(
            moutai_ticker, limit=2
        )  # Get latest 2 periods
        print(f"Moutai Line Items (latest 2): {moutai_items}")
    except Exception as e:
        print(f"Error getting Moutai line items: {e}")

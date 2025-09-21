"""
Market Data Agent

Responsible for gathering and preprocessing market data for financial analysis.
"""

from datetime import datetime, timedelta
from typing import Any

from pyomnix.omnix_logger import get_logger

from .. import (
    get_financial_metrics,
    get_market_cap,
    get_prices,
    prices_to_df,
    search_line_items,
)
from ..utils.progress import progress
from .state import AgentState, show_agent_reasoning

# 设置日志记录
logger = get_logger("market_data_agent")


def market_data_agent(state: AgentState) -> AgentState:
    """
    Responsible for gathering and preprocessing market data.

    Args:
        state: The current state of the agent system

    Returns:
        Updated state with market data
    """
    progress.update_status("market_data_agent", None, "Starting data collection")
    show_reasoning = state["metadata"].get("show_reasoning", False)

    messages = state["messages"]
    data = state["data"]

    # Set default dates
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = data.get("end_date") or yesterday.strftime("%Y-%m-%d")

    # Ensure end_date is not in the future
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    if end_date_obj > yesterday:
        end_date = yesterday.strftime("%Y-%m-%d")
        end_date_obj = yesterday

    if not data.get("start_date"):
        # Calculate 1 year before end_date
        start_date = end_date_obj - timedelta(days=365)  # 默认获取一年的数据
        start_date = start_date.strftime("%Y-%m-%d")
    else:
        start_date = data["start_date"]

    # Get all required data for each ticker
    tickers = data.get("tickers", [])

    # Initialize data containers
    all_prices: dict[str, list[dict[str, Any]]] = {}
    all_financial_metrics: dict[str, list[Any]] = {}
    all_financial_line_items: dict[str, list[Any]] = {}
    all_market_caps: dict[str, float | None] = {}

    # Process each ticker
    for ticker in tickers:
        progress.update_status("market_data_agent", ticker, "Fetching price data")

        # 获取价格数据并验证
        try:
            prices = get_prices(ticker, start_date, end_date)
            prices_df = prices_to_df(prices)
            if prices_df.empty:
                logger.warning(f"警告：无法获取{ticker}的价格数据，将使用空数据继续")
                prices_dict = []
            else:
                prices_dict = [p.model_dump() for p in prices]
            all_prices[ticker] = prices_dict
        except Exception as e:
            logger.error(f"获取价格数据失败: {e!s}")
            all_prices[ticker] = []

        # 获取财务指标
        progress.update_status(
            "market_data_agent", ticker, "Fetching financial metrics"
        )
        try:
            financial_metrics = get_financial_metrics(ticker, end_date)
            all_financial_metrics[ticker] = [m.model_dump() for m in financial_metrics]
        except Exception as e:
            logger.error(f"获取财务指标失败: {e!s}")
            all_financial_metrics[ticker] = []

        # 获取财务报表行项目
        progress.update_status(
            "market_data_agent", ticker, "Fetching financial line items"
        )
        try:
            # Common line items to search for
            financial_line_items = search_line_items(ticker, period="ttm", limit=10)
            all_financial_line_items[ticker] = [
                item.model_dump() for item in financial_line_items
            ]
        except Exception as e:
            logger.error(f"获取财务报表失败: {e!s}")
            all_financial_line_items[ticker] = []

        # 获取市值
        progress.update_status("market_data_agent", ticker, "Fetching market cap")
        try:
            market_cap = get_market_cap(ticker, end_date)
            all_market_caps[ticker] = market_cap
        except Exception as e:
            logger.error(f"获取市值失败: {e!s}")
            all_market_caps[ticker] = None

        progress.update_status("market_data_agent", ticker, "Data collection complete")

    # 保存推理信息到metadata供API使用
    market_data_summary = {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "data_collected": {
            "price_history": {
                ticker: len(all_prices.get(ticker, [])) > 0 for ticker in tickers
            },
            "financial_metrics": {
                ticker: len(all_financial_metrics.get(ticker, [])) > 0
                for ticker in tickers
            },
            "financial_line_items": {
                ticker: len(all_financial_line_items.get(ticker, [])) > 0
                for ticker in tickers
            },
            "market_caps": {
                ticker: all_market_caps.get(ticker) is not None for ticker in tickers
            },
        },
        "summary": f"为{', '.join(tickers)}收集了从{start_date}到{end_date}的市场数据，包括价格历史、财务指标和市场信息",
    }

    if show_reasoning:
        show_agent_reasoning(market_data_summary, "Market Data Agent")
        state["metadata"]["agent_reasoning"] = market_data_summary

    progress.update_status("market_data_agent", None, "Done")

    # Update state with collected data
    return {
        "messages": messages,
        "data": {
            **data,
            "prices": all_prices,
            "start_date": start_date,
            "end_date": end_date,
            "financial_metrics": all_financial_metrics,
            "financial_line_items": all_financial_line_items,
            "market_caps": all_market_caps,
        },
        "metadata": state["metadata"],
    }

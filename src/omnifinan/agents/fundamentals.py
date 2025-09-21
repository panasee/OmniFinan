import json

from langchain_core.messages import HumanMessage

from .. import get_financial_metrics
from ..utils.progress import progress
from .state import AgentState, show_agent_reasoning


##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize fundamental analysis for each ticker
    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status(
            "fundamentals_agent", ticker, "Fetching financial metrics"
        )

        # Get the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
        )

        if not financial_metrics:
            progress.update_status(
                "fundamentals_agent", ticker, "Failed: No financial metrics found"
            )
            continue

        # Pull the most recent financial metrics
        metrics = financial_metrics[0]

        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        progress.update_status("fundamentals_agent", ticker, "Analyzing profitability")
        # 1. Profitability Analysis (采用第二版的None处理和默认值)
        return_on_equity = metrics.get("return_on_equity", 0)
        net_margin = metrics.get("net_margin", 0)
        operating_margin = metrics.get("operating_margin", 0)

        thresholds = [
            (return_on_equity, 0.15),  # Strong ROE above 15%
            (net_margin, 0.20),  # Healthy profit margins
            (operating_margin, 0.15),  # Strong operating efficiency
        ]
        profitability_score = sum(
            metric is not None and metric > threshold
            for metric, threshold in thresholds
        )

        signals.append(
            "bullish"
            if profitability_score >= 2
            else "bearish"
            if profitability_score == 0
            else "neutral"
        )
        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (
                f"ROE: {return_on_equity:.2%}"
                if return_on_equity is not None
                else "ROE: N/A"
            )
            + ", "
            + (
                f"Net Margin: {net_margin:.2%}"
                if net_margin is not None
                else "Net Margin: N/A"
            )
            + ", "
            + (
                f"Op Margin: {operating_margin:.2%}"
                if operating_margin is not None
                else "Op Margin: N/A"
            ),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing growth")
        # 2. Growth Analysis (采用第二版的简化显示)
        revenue_growth = metrics.get("revenue_growth", 0)
        earnings_growth = metrics.get("earnings_growth", 0)
        book_value_growth = metrics.get("book_value_growth", 0)

        thresholds = [
            (revenue_growth, 0.10),  # 10% revenue growth
            (earnings_growth, 0.10),  # 10% earnings growth
            (book_value_growth, 0.10),  # 10% book value growth
        ]
        growth_score = sum(
            metric is not None and metric > threshold
            for metric, threshold in thresholds
        )

        signals.append(
            "bullish"
            if growth_score >= 2
            else "bearish"
            if growth_score == 0
            else "neutral"
        )
        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (
                f"Revenue Growth: {revenue_growth:.2%}"
                if revenue_growth is not None
                else "Revenue Growth: N/A"
            )
            + ", "
            + (
                f"Earnings Growth: {earnings_growth:.2%}"
                if earnings_growth is not None
                else "Earnings Growth: N/A"
            ),
        }

        progress.update_status(
            "fundamentals_agent", ticker, "Analyzing financial health"
        )
        # 3. Financial Health (采用第二版的简化判断逻辑)
        current_ratio = metrics.get("current_ratio", 0)
        debt_to_equity = metrics.get("debt_to_equity", 0)
        free_cash_flow_per_share = metrics.get("free_cash_flow_per_share", 0)
        earnings_per_share = metrics.get("earnings_per_share", 0)

        health_score = 0
        if current_ratio and current_ratio > 1.5:  # Strong liquidity
            health_score += 1
        if debt_to_equity and debt_to_equity < 0.5:  # Conservative debt levels
            health_score += 1
        if (
            free_cash_flow_per_share
            and earnings_per_share
            and free_cash_flow_per_share > earnings_per_share * 0.8
        ):  # Strong FCF conversion
            health_score += 1

        signals.append(
            "bullish"
            if health_score >= 2
            else "bearish"
            if health_score == 0
            else "neutral"
        )
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (
                f"Current Ratio: {current_ratio:.2f}"
                if current_ratio is not None
                else "Current Ratio: N/A"
            )
            + ", "
            + (
                f"D/E: {debt_to_equity:.2f}"
                if debt_to_equity is not None
                else "D/E: N/A"
            ),
        }

        progress.update_status(
            "fundamentals_agent", ticker, "Analyzing valuation ratios"
        )
        # 4. Price to X ratios (采用第二版更合理的比较方向)
        pe_ratio = metrics.get("price_to_earnings_ratio", 0)
        pb_ratio = metrics.get("price_to_book_ratio", 0)
        ps_ratio = metrics.get("price_to_sales_ratio", 0)

        thresholds = [
            (pe_ratio, 25),  # Lower P/E is better
            (pb_ratio, 3),  # Lower P/B is better
            (ps_ratio, 5),  # Lower P/S is better
        ]
        price_ratio_score = sum(
            metric is not None and metric < threshold  # 改为<比较
            for metric, threshold in thresholds
        )

        signals.append(
            "bullish"  # 低估值是bullish，与第二版一致
            if price_ratio_score >= 2
            else "bearish"
            if price_ratio_score == 0
            else "neutral"
        )
        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (f"P/E: {pe_ratio:.2f}" if pe_ratio is not None else "P/E: N/A")
            + ", "
            + (f"P/B: {pb_ratio:.2f}" if pb_ratio is not None else "P/B: N/A")
            + ", "
            + (f"P/S: {ps_ratio:.2f}" if ps_ratio is not None else "P/S: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Calculating final signal")
        # Determine overall signal (保持第一版逻辑)
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level (保持第一版格式)
        total_signals = len(signals)
        confidence = (
            round(max(bullish_signals, bearish_signals) / total_signals, 2) * 100
        )

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("fundamentals_agent", ticker, "Done")

    # Create the fundamental analysis message (保持第一版结构)
    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name="fundamentals_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["fundamentals_agent"] = fundamental_analysis

    return state | {"messages": [message], "data": data}

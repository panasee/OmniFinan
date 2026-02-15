import json

import pandas as pd
from langchain_core.messages import HumanMessage

from ..data.unified_service import UnifiedDataService
from ..utils.progress import progress
from .state import (
    AgentState,
    default_risk_debate_state,
    show_agent_reasoning,
    show_workflow_status,
)


##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    show_workflow_status("Risk Manager")
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]
    data_service = state["metadata"].get("data_service")
    if not isinstance(data_service, UnifiedDataService):
        raise RuntimeError("risk_management_agent requires metadata.data_service")

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    debate_state = state["data"].get("debate_state", {})
    risk_debate_state = state["data"].get("risk_debate_state") or default_risk_debate_state(
        max_rounds=int(state["metadata"].get("max_risk_discuss_rounds", 1))
    )
    current_prices = {}  # Store prices here to avoid redundant API calls
    price_frames: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        prices = data_service.get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )
        if not prices:
            continue
        prices_df = pd.DataFrame(prices)
        if prices_df.empty:
            continue
        prices_df["time"] = pd.to_datetime(prices_df["time"], errors="coerce")
        prices_df = prices_df.dropna(subset=["time"]).sort_values("time")
        if prices_df.empty:
            continue
        price_frames[ticker] = prices_df
        current_prices[ticker] = float(prices_df["close"].iloc[-1])

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")
        prices_df = price_frames.get(ticker)
        if prices_df is None:
            progress.update_status(
                "risk_management_agent", ticker, "Failed: No price data found"
            )
            continue

        progress.update_status(
            "risk_management_agent", ticker, "Calculating risk metrics"
        )

        # 1. Calculate Risk Metrics
        returns = prices_df["close"].pct_change().dropna()
        daily_vol = returns.std()
        # Annualized volatility approximation
        volatility = daily_vol * (252**0.5)

        # 计算波动率的历史分布
        rolling_std = returns.rolling(window=120, min_periods=20).std() * (252**0.5)
        volatility_mean = rolling_std.mean()
        volatility_std = rolling_std.std()
        volatility_percentile = (
            (volatility - volatility_mean) / volatility_std
            if volatility_std != 0
            else 0
        )

        # Simple historical VaR at 95% confidence
        var_95 = returns.quantile(0.05)
        # 使用60天窗口计算最大回撤
        max_drawdown = (
            prices_df["close"]
            / prices_df["close"].rolling(window=60, min_periods=10).max()
            - 1
        ).min()

        # 2. Market Risk Assessment
        market_risk_score = 0

        # Volatility scoring based on percentile
        if volatility_percentile > 1.5:  # 高于1.5个标准差
            market_risk_score += 2
        elif volatility_percentile > 1.0:  # 高于1个标准差
            market_risk_score += 1

        # VaR scoring
        # Note: var_95 is typically negative. The more negative, the worse.
        if var_95 < -0.03:
            market_risk_score += 2
        elif var_95 < -0.02:
            market_risk_score += 1

        # Max Drawdown scoring
        if max_drawdown < -0.20:  # Severe drawdown
            market_risk_score += 2
        elif max_drawdown < -0.10:
            market_risk_score += 1

        progress.update_status(
            "risk_management_agent", ticker, "Calculating position limits"
        )

        # 3. Position Size Limits
        # Calculate portfolio value
        current_price = current_prices[ticker]

        # Calculate current position value for this ticker
        position = portfolio.get("positions", {}).get(ticker, {})
        long_shares = float(position.get("long", 0) or 0)
        short_shares = float(position.get("short", 0) or 0)
        current_position_value = (long_shares - short_shares) * current_price
        if not position and "cost_basis" in portfolio:
            current_position_value = float(portfolio.get("cost_basis", {}).get(ticker, 0) or 0)

        # Calculate total portfolio value using stored prices
        total_portfolio_value = float(portfolio.get("cash", 0) or 0)
        for t in tickers:
            t_pos = portfolio.get("positions", {}).get(t, {})
            t_long = float(t_pos.get("long", 0) or 0)
            t_short = float(t_pos.get("short", 0) or 0)
            t_px = current_prices.get(t)
            if t_px is None:
                continue
            total_portfolio_value += (t_long - t_short) * t_px
        if not portfolio.get("positions") and "cost_basis" in portfolio:
            total_portfolio_value = float(portfolio.get("cash", 0) or 0) + sum(
                float(v or 0) for v in portfolio.get("cost_basis", {}).values()
            )

        # Base limit is 20% of portfolio for any single position
        base_position_limit = max(total_portfolio_value, 0.0) * 0.20

        # Adjust position limit based on risk score
        if market_risk_score >= 4:
            # Reduce position for high risk
            position_limit = base_position_limit * 0.5
        elif market_risk_score >= 2:
            # Slightly reduce for moderate risk
            position_limit = base_position_limit * 0.75
        else:
            # Keep base size for low risk
            position_limit = base_position_limit

        # For existing positions, subtract current position value from limit
        remaining_position_limit = position_limit - current_position_value

        # Ensure we don't exceed available cash
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))

        # 4. Stress Testing
        stress_test_scenarios = {
            "market_crash": -0.20,
            "moderate_decline": -0.10,
            "slight_decline": -0.05,
        }

        stress_test_results = {}
        for scenario, decline in stress_test_scenarios.items():
            potential_loss = current_position_value * decline
            portfolio_impact = (
                potential_loss / total_portfolio_value
                if total_portfolio_value != 0
                else float("nan")
            )
            stress_test_results[scenario] = {
                "potential_loss": float(potential_loss),
                "portfolio_impact": float(portfolio_impact),
            }

        # 5. Generate Trading Action
        if market_risk_score >= 9:
            trading_action = "hold"
        elif market_risk_score >= 7:
            trading_action = "reduce"
        elif market_risk_score >= 5:
            trading_action = "caution"
        else:
            trading_action = "normal"

        # Cap risk score at 10
        risk_score = min(round(market_risk_score), 10)

        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "risk_score": risk_score,
            "trading_action": trading_action,
            "signal": "bearish"
            if risk_score >= 7
            else "neutral"
            if risk_score >= 4
            else "bullish",
            "confidence": float(min(risk_score / 10, 1.0)),
            "risk_metrics": {
                "volatility": float(volatility),
                "value_at_risk_95": float(var_95),
                "max_drawdown": float(max_drawdown),
                "market_risk_score": market_risk_score,
                "stress_test_results": stress_test_results,
            },
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
                "risk_assessment": f"Risk Score {risk_score}/10: Volatility={volatility:.2%}, VaR={var_95:.2%}, Max Drawdown={max_drawdown:.2%}",
            },
        }

        progress.update_status("risk_management_agent", ticker, "Done")

        risk_debate_state["risky_history"].append(
            {ticker: {"signal": "bearish", "reason": "volatility/var/drawdown stress"}}
        )
        risk_debate_state["safe_history"].append(
            {ticker: {"signal": "neutral", "reason": "position limit and cash constraints"}}
        )
        risk_debate_state["neutral_history"].append(
            {ticker: {"signal": risk_analysis[ticker]["signal"], "reason": "risk score synthesis"}}
        )
        debate_state.setdefault("risk_arguments", []).append(
            {
                "ticker": ticker,
                "signal": risk_analysis[ticker]["signal"],
                "risk_score": risk_score,
                "reason": risk_analysis[ticker]["reasoning"]["risk_assessment"],
            }
        )

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if show_reasoning:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")
        # 保存推理信息到metadata供API使用
        state["metadata"]["agent_reasoning"] = risk_analysis

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis
    risk_debate_state["judge_decision"] = {
        ticker: {
            "signal": details["signal"],
            "confidence": details["confidence"],
            "trading_action": details["trading_action"],
        }
        for ticker, details in risk_analysis.items()
    }
    risk_debate_state["count"] = int(risk_debate_state.get("count", 0)) + 1
    state["data"]["risk_debate_state"] = risk_debate_state
    state["data"]["debate_state"] = debate_state

    show_workflow_status("Risk Manager", "completed")
    return state | {"messages": state["messages"] + [message], "data": data}

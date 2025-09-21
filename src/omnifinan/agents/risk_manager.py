import json

from langchain_core.messages import HumanMessage

from .. import get_prices, prices_to_df
from ..utils.progress import progress
from .state import AgentState, show_agent_reasoning, show_workflow_status


##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    show_workflow_status("Risk Manager")
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )

        if not prices:
            progress.update_status(
                "risk_management_agent", ticker, "Failed: No price data found"
            )
            continue

        prices_df = prices_to_df(prices)

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
        current_price = prices_df["close"].iloc[-1]
        current_prices[ticker] = current_price  # Store the current price

        # Calculate current position value for this ticker
        current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0)

        # Calculate total portfolio value using stored prices
        total_portfolio_value = portfolio.get("cash", 0) + sum(
            portfolio.get("cost_basis", {}).get(t, 0)
            for t in portfolio.get("cost_basis", {})
        )

        # Base limit is 20% of portfolio for any single position
        base_position_limit = total_portfolio_value * 0.20

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

    show_workflow_status("Risk Manager", "completed")
    return state | {"messages": state["messages"] + [message], "data": data}

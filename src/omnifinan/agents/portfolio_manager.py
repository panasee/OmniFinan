import json
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..utils.llm import call_llm
from ..utils.progress import progress
from .state import AgentState, show_agent_reasoning


class AgentSignal(TypedDict):
    agent_name: str
    signal: str
    confidence: float


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(
        description="Confidence in the decision, between 0.0 and 100.0"
    )
    reasoning: str = Field(description="Reasoning for the decision")
    agent_signals: list[AgentSignal] | None = Field(
        description="Signals from different agents that influenced this decision",
        default=None,
    )


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(
        description="Dictionary of ticker to trading decisions"
    )


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders for multiple tickers"""

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    progress.update_status("portfolio_management_agent", None, "Analyzing signals")

    # Get position limits, current prices, and signals for every ticker
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status(
            "portfolio_management_agent", ticker, "Processing analyst signals"
        )

        # Get position limits and current prices for the ticker
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # Get signals for the ticker
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                ticker_signals[agent] = {
                    "signal": signals[ticker]["signal"],
                    "confidence": signals[ticker]["confidence"],
                }
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status(
        "portfolio_management_agent", None, "Making trading decisions"
    )

    # Generate the trading decision
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        model_name=state["metadata"]["model_name"],
        provider_api=state["metadata"]["provider_api"],
        language=state["metadata"].get("language", "Chinese"),
    )

    # Format detailed analysis reports for each ticker
    detailed_reports = {}
    for ticker, decision in result.decisions.items():
        detailed_reports[ticker] = format_decision(
            ticker=ticker,
            action=decision.action,
            quantity=decision.quantity,
            confidence=decision.confidence,
            agent_signals=decision.agent_signals or [],
            reasoning=decision.reasoning,
            signals_by_ticker=signals_by_ticker.get(ticker, {}),
            current_price=current_prices.get(ticker, 0),
        )

    # Create the portfolio management message
    message = HumanMessage(
        content=json.dumps(
            {
                ticker: decision.model_dump()
                for ticker, decision in result.decisions.items()
            }
        ),
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(detailed_reports, "Portfolio Management Agent")

    progress.update_status("portfolio_management_agent", None, "Done")

    return state | {"messages": state["messages"] + [message], "data": state["data"]}


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float | dict] | None,
    model_name: str,
    provider_api: str,
    language: str = "Chinese",
) -> PortfolioManagerOutput:
    """Attempts to get a decision from the LLM with retry logic"""
    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""Always respond in {language}. You are a portfolio manager making final trading decisions based on multiple tickers.

              When weighing the different signals for direction and timing:
              1. Valuation Analysis (35% weight)
                 - Primary driver of fair value assessment
                 - Determines if price offers good entry/exit point
              
              2. Fundamental Analysis (30% weight)
                 - Business quality and growth assessment
                 - Determines conviction in long-term potential
              
              3. Technical Analysis (25% weight)
                 - Secondary confirmation
                 - Helps with entry/exit timing
              
              4. Sentiment Analysis (10% weight)
                 - Final consideration
                 - Can influence sizing within risk limits

              Trading Rules:
              - For long positions:
                * Only buy if you have available cash
                * Only sell if you currently hold long shares of that ticker
                * Sell quantity must be ≤ current long position shares
                * Buy quantity must be ≤ max_shares for that ticker
              
              - For short positions:
                * Only short if you have available margin (position value × margin requirement)
                * Only cover if you currently have short shares of that ticker
                * Cover quantity must be ≤ current short position shares
                * Short quantity must respect margin requirements
              
              - The max_shares values are pre-calculated to respect position limits
              - Consider both long and short opportunities based on signals
              - Maintain appropriate risk management with both long and short exposure

              Available Actions:
              - "buy": Open or add to long position
              - "sell": Close or reduce long position
              - "short": Open or add to short position
              - "cover": Close or reduce short position
              - "hold": No action

              Inputs:
              - signals_by_ticker: dictionary of ticker → signals
              - max_shares: maximum shares allowed per ticker
              - portfolio_cash: current cash in portfolio
              - portfolio_positions: current positions (both long and short)
              - current_prices: current prices for each ticker
              - margin_requirement: current margin requirement for short positions (e.g., 0.5 means 50%)
              - total_margin_used: total margin currently in use
              """,
            ),
            (
                "human",
                """Based on the team's analysis, make your trading decisions for each ticker.

              Here are the signals by ticker:
              {signals_by_ticker}

              Current Prices:
              {current_prices}

              Maximum Shares Allowed For Purchases:
              {max_shares}

              Portfolio Cash: {portfolio_cash}
              Current Positions: {portfolio_positions}
              Current Margin Requirement: {margin_requirement}
              Total Margin Used: {total_margin_used}

              For each ticker, provide:
              - "action": "buy" | "sell" | "short" | "cover" | "hold"
              - "quantity": <positive integer>
              - "confidence": <float between 0 and 100>
              - "agent_signals": <list of agent signals including agent name, signal (bullish | bearish | neutral), and their confidence>
              - "reasoning": <concise explanation of the decision including how you weighted the signals>

              Output strictly in JSON with the following structure:
              {
                "decisions": {
                  "TICKER1": {
                    "action": "buy/sell/short/cover/hold",
                    "quantity": integer,
                    "confidence": float between 0 and 100,
                    "agent_signals": [
                      {"agent_name": "agent1", "signal": "bullish/bearish/neutral", "confidence": float},
                      ...
                    ],
                    "reasoning": "string"
                  },
                  "TICKER2": {
                    ...
                  },
                  ...
                }
              }
              """,
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            "max_shares": json.dumps(max_shares, indent=2),
            "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
            "portfolio_positions": json.dumps(portfolio.get("positions", {}), indent=2),
            "margin_requirement": f"{portfolio.get('margin_requirement', 0.5):.2f}",
            "total_margin_used": f"{portfolio.get('margin_used', 0):.2f}",
        }
    )

    # Create default factory for PortfolioManagerOutput
    def create_default_portfolio_output():
        return PortfolioManagerOutput(
            decisions={
                ticker: PortfolioDecision(
                    action="hold",
                    quantity=0,
                    confidence=0.0,
                    reasoning="Error in portfolio management, defaulting to hold",
                    agent_signals=[],
                )
                for ticker in tickers
            }
        )

    if not portfolio:
        return create_default_portfolio_output()

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        provider_api=provider_api,
        pydantic_model=PortfolioManagerOutput,
        agent_name="portfolio_management_agent",
        default_factory=create_default_portfolio_output,
    )


def format_decision(
    ticker: str,
    action: str,
    quantity: int,
    confidence: float,
    agent_signals: list[AgentSignal],
    reasoning: str,
    signals_by_ticker: dict,
    current_price: float,
) -> dict:
    """Format the trading decision into a standardized output format with detailed analysis."""

    # 获取各个agent的信号
    def get_signal(agent_name):
        return next(
            (signal for signal in agent_signals if signal["agent_name"] == agent_name),
            None,
        )

    fundamental_signal = get_signal("fundamental_analysis")
    valuation_signal = get_signal("valuation_analysis")
    technical_signal = get_signal("technical_analysis")
    sentiment_signal = get_signal("sentiment_analysis")
    risk_signal = get_signal("risk_management")

    # 转换信号为中文
    def signal_to_chinese(signal):
        if not signal:
            return "无数据"
        if signal["signal"] == "bullish":
            return "看多"
        elif signal["signal"] == "bearish":
            return "看空"
        return "中性"

    # 创建详细分析报告
    detailed_analysis = f"""
====================================
          投资分析报告 - {ticker}
====================================

一、策略分析

1. 基本面分析 (权重30%):
   信号: {signal_to_chinese(fundamental_signal)}
   置信度: {fundamental_signal["confidence"] * 100:.0f}% if fundamental_signal else "无数据"

2. 估值分析 (权重35%):
   信号: {signal_to_chinese(valuation_signal)}
   置信度: {valuation_signal["confidence"] * 100:.0f}% if valuation_signal else "无数据"

3. 技术分析 (权重25%):
   信号: {signal_to_chinese(technical_signal)}
   置信度: {technical_signal["confidence"] * 100:.0f}% if technical_signal else "无数据"

4. 情绪分析 (权重10%):
   信号: {signal_to_chinese(sentiment_signal)}
   置信度: {sentiment_signal["confidence"] * 100:.0f}% if sentiment_signal else "无数据"

二、风险评估
风险信号: {signal_to_chinese(risk_signal)}
置信度: {risk_signal["confidence"] * 100:.0f}% if risk_signal else "无数据"

三、投资建议
操作建议: {"买入" if action == "buy" else "卖出" if action == "sell" else "做空" if action == "short" else "平仓" if action == "cover" else "持有"}
交易数量: {quantity}股
当前价格: {current_price:.2f}
决策置信度: {confidence:.0f}%

四、决策依据
{reasoning}

===================================="""

    return {
        "action": action,
        "quantity": quantity,
        "confidence": confidence,
        "agent_signals": agent_signals,
        "current_price": current_price,
        "分析报告": detailed_analysis,
    }

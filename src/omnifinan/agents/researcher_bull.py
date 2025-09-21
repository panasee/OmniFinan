"""
Bullish Researcher Agent

Analyzes signals from a bullish perspective and generates optimistic investment thesis.
"""

import json
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from pyomnix.omnix_logger import get_logger

from ..utils.progress import progress
from .state import AgentState, show_agent_reasoning

# 设置日志记录
logger = get_logger("researcher_bull_agent")


class BullishThesis(BaseModel):
    """Model for the bullish thesis output."""

    perspective: Literal["bullish"] = "bullish"
    confidence: float = Field(description="Confidence level between 0 and 1")
    thesis_points: list[str] = Field(description="List of bullish thesis points")
    reasoning: str = Field(description="Reasoning behind the bullish thesis")


def researcher_bull_agent(state: AgentState) -> AgentState:
    """
    Analyzes signals from a bullish perspective and generates optimistic investment thesis.

    Args:
        state: The current state of the agent system

    Returns:
        Updated state with bullish thesis
    """
    progress.update_status(
        "researcher_bull_agent", None, "Analyzing from bullish perspective"
    )
    show_reasoning = state["metadata"].get("show_reasoning", False)

    # Get the tickers
    tickers = state["data"].get("tickers", [])

    # Initialize results container
    bullish_analyses: dict[str, BullishThesis] = {}

    for ticker in tickers:
        progress.update_status(
            "researcher_bull_agent", ticker, "Collecting analyst signals"
        )

        # Get analyst signals for this ticker
        analyst_signals = state["data"].get("analyst_signals", {})

        # Fetch signals from different analysts
        technical_signals = analyst_signals.get("technical_analyst_agent", {}).get(
            ticker, {}
        )
        fundamental_signals = analyst_signals.get("fundamentals_agent", {}).get(
            ticker, {}
        )
        sentiment_signals = analyst_signals.get("sentiment_agent", {}).get(ticker, {})
        valuation_signals = analyst_signals.get("valuation_agent", {}).get(ticker, {})

        # Analyze from bullish perspective
        bullish_points = []
        confidence_scores = []

        progress.update_status(
            "researcher_bull_agent", ticker, "Analyzing technical signals"
        )
        # Technical Analysis
        if technical_signals.get("signal") == "bullish":
            bullish_points.append(
                f"Technical indicators show bullish momentum with {technical_signals.get('confidence')}% confidence"
            )
            confidence_scores.append(
                float(technical_signals.get("confidence", 0)) / 100
            )
        else:
            bullish_points.append(
                "Technical indicators may be conservative, presenting buying opportunities"
            )
            confidence_scores.append(0.3)

        progress.update_status(
            "researcher_bull_agent", ticker, "Analyzing fundamental signals"
        )
        # Fundamental Analysis
        if fundamental_signals.get("signal") == "bullish":
            bullish_points.append(
                f"Strong fundamentals with {fundamental_signals.get('confidence')}% confidence"
            )
            confidence_scores.append(
                float(fundamental_signals.get("confidence", 0)) / 100
            )
        else:
            bullish_points.append("Company fundamentals show potential for improvement")
            confidence_scores.append(0.3)

        progress.update_status(
            "researcher_bull_agent", ticker, "Analyzing sentiment signals"
        )
        # Sentiment Analysis
        if sentiment_signals.get("signal") == "bullish":
            bullish_points.append(
                f"Positive market sentiment with {sentiment_signals.get('confidence')}% confidence"
            )
            confidence_scores.append(
                float(sentiment_signals.get("confidence", 0)) / 100
            )
        else:
            bullish_points.append(
                "Market sentiment may be overly pessimistic, creating value opportunities"
            )
            confidence_scores.append(0.3)

        progress.update_status(
            "researcher_bull_agent", ticker, "Analyzing valuation signals"
        )
        # Valuation Analysis
        if valuation_signals.get("signal") == "bullish":
            bullish_points.append(
                f"Stock appears undervalued with {valuation_signals.get('confidence')}% confidence"
            )
            confidence_scores.append(
                float(valuation_signals.get("confidence", 0)) / 100
            )
        else:
            bullish_points.append(
                "Current valuation may not fully reflect growth potential"
            )
            confidence_scores.append(0.3)

        # Calculate overall bullish confidence
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.5
        )

        # Create the bullish thesis
        bullish_thesis = BullishThesis(
            perspective="bullish",
            confidence=avg_confidence,
            thesis_points=bullish_points,
            reasoning="Bullish thesis based on comprehensive analysis of technical, fundamental, sentiment, and valuation factors",
        )

        bullish_analyses[ticker] = bullish_thesis
        progress.update_status(
            "researcher_bull_agent", ticker, "Bullish analysis complete"
        )

    # Create messages for each ticker
    messages = state["messages"].copy()
    for ticker, thesis in bullish_analyses.items():
        message = HumanMessage(
            content=json.dumps(thesis.model_dump()),
            name="researcher_bull_agent",
        )
        messages.append(message)

        if show_reasoning:
            show_agent_reasoning(thesis.model_dump(), f"Bullish Researcher - {ticker}")

    # Update state metadata
    if bullish_analyses and show_reasoning:
        state["metadata"]["agent_reasoning"] = next(
            iter(bullish_analyses.values())
        ).model_dump()

    progress.update_status("researcher_bull_agent", None, "Done")

    # Update state with bullish analyses
    return {
        "messages": messages,
        "data": {
            **state["data"],
            "bullish_analyses": {
                ticker: analysis.model_dump()
                for ticker, analysis in bullish_analyses.items()
            },
        },
        "metadata": state["metadata"],
    }

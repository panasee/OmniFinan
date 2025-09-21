"""
Bearish Researcher Agent

Analyzes signals from a bearish perspective and generates cautionary investment thesis.
"""

import json
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from pyomnix.omnix_logger import get_logger

from ..utils.progress import progress
from .state import AgentState, show_agent_reasoning

# 设置日志记录
logger = get_logger("researcher_bear_agent")


class BearishThesis(BaseModel):
    """Model for the bearish thesis output."""

    perspective: Literal["bearish"] = "bearish"
    confidence: float = Field(description="Confidence level between 0 and 1")
    thesis_points: list[str] = Field(description="List of bearish thesis points")
    reasoning: str = Field(description="Reasoning behind the bearish thesis")


def researcher_bear_agent(state: AgentState) -> AgentState:
    """
    Analyzes signals from a bearish perspective and generates cautionary investment thesis.

    Args:
        state: The current state of the agent system

    Returns:
        Updated state with bearish thesis
    """
    progress.update_status(
        "researcher_bear_agent", None, "Analyzing from bearish perspective"
    )
    show_reasoning = state["metadata"].get("show_reasoning", False)

    # Get the tickers
    tickers = state["data"].get("tickers", [])

    # Initialize results container
    bearish_analyses: dict[str, BearishThesis] = {}

    for ticker in tickers:
        progress.update_status(
            "researcher_bear_agent", ticker, "Collecting analyst signals"
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

        # Analyze from bearish perspective
        bearish_points = []
        confidence_scores = []

        progress.update_status(
            "researcher_bear_agent", ticker, "Analyzing technical signals"
        )
        # Technical Analysis
        if technical_signals.get("signal") == "bearish":
            bearish_points.append(
                f"Technical indicators show bearish momentum with {technical_signals.get('confidence')}% confidence"
            )
            confidence_scores.append(
                float(technical_signals.get("confidence", 0)) / 100
            )
        else:
            bearish_points.append(
                "Technical rally may be temporary, suggesting potential reversal"
            )
            confidence_scores.append(0.3)

        progress.update_status(
            "researcher_bear_agent", ticker, "Analyzing fundamental signals"
        )
        # Fundamental Analysis
        if fundamental_signals.get("signal") == "bearish":
            bearish_points.append(
                f"Concerning fundamentals with {fundamental_signals.get('confidence')}% confidence"
            )
            confidence_scores.append(
                float(fundamental_signals.get("confidence", 0)) / 100
            )
        else:
            bearish_points.append("Current fundamental strength may not be sustainable")
            confidence_scores.append(0.3)

        progress.update_status(
            "researcher_bear_agent", ticker, "Analyzing sentiment signals"
        )
        # Sentiment Analysis
        if sentiment_signals.get("signal") == "bearish":
            bearish_points.append(
                f"Negative market sentiment with {sentiment_signals.get('confidence')}% confidence"
            )
            confidence_scores.append(
                float(sentiment_signals.get("confidence", 0)) / 100
            )
        else:
            bearish_points.append(
                "Market sentiment may be overly optimistic, indicating potential risks"
            )
            confidence_scores.append(0.3)

        progress.update_status(
            "researcher_bear_agent", ticker, "Analyzing valuation signals"
        )
        # Valuation Analysis
        if valuation_signals.get("signal") == "bearish":
            bearish_points.append(
                f"Stock appears overvalued with {valuation_signals.get('confidence')}% confidence"
            )
            confidence_scores.append(
                float(valuation_signals.get("confidence", 0)) / 100
            )
        else:
            bearish_points.append(
                "Current valuation may not fully reflect downside risks"
            )
            confidence_scores.append(0.3)

        # Calculate overall bearish confidence
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.5
        )

        # Create the bearish thesis
        bearish_thesis = BearishThesis(
            perspective="bearish",
            confidence=avg_confidence,
            thesis_points=bearish_points,
            reasoning="Bearish thesis based on comprehensive analysis of technical, fundamental, sentiment, and valuation factors",
        )

        bearish_analyses[ticker] = bearish_thesis
        progress.update_status(
            "researcher_bear_agent", ticker, "Bearish analysis complete"
        )

    # Create messages for each ticker
    messages = state["messages"].copy()
    for ticker, thesis in bearish_analyses.items():
        message = HumanMessage(
            content=json.dumps(thesis.model_dump()),
            name="researcher_bear_agent",
        )
        messages.append(message)

        if show_reasoning:
            show_agent_reasoning(thesis.model_dump(), f"Bearish Researcher - {ticker}")

    # Update state metadata
    if bearish_analyses and show_reasoning:
        state["metadata"]["agent_reasoning"] = next(
            iter(bearish_analyses.values())
        ).model_dump()

    progress.update_status("researcher_bear_agent", None, "Done")

    # Update state with bearish analyses
    return {
        "messages": messages,
        "data": {
            **state["data"],
            "bearish_analyses": {
                ticker: analysis.model_dump()
                for ticker, analysis in bearish_analyses.items()
            },
        },
        "metadata": state["metadata"],
    }

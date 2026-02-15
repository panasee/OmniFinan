"""Regression tests for OmniFinan graph edge routing."""

from omnifinan.agents.edges import (
    should_continue_investment_debate,
    should_continue_risk_review,
)


def test_investment_debate_routes_bull_then_bear_then_judge() -> None:
    state = {
        "messages": [],
        "data": {
            "investment_debate_state": {"count": 0, "max_rounds": 1},
        },
        "metadata": {},
    }
    assert should_continue_investment_debate(state) == "bull"

    state["data"]["investment_debate_state"]["count"] = 1
    assert should_continue_investment_debate(state) == "bear"

    state["data"]["investment_debate_state"]["count"] = 2
    assert should_continue_investment_debate(state) == "judge"


def test_risk_review_routes_until_max_rounds_then_portfolio() -> None:
    state = {
        "messages": [],
        "data": {
            "risk_debate_state": {"count": 0, "max_rounds": 2},
        },
        "metadata": {},
    }
    assert should_continue_risk_review(state) == "risk"

    state["data"]["risk_debate_state"]["count"] = 1
    assert should_continue_risk_review(state) == "risk"

    state["data"]["risk_debate_state"]["count"] = 2
    assert should_continue_risk_review(state) == "portfolio"

"""Conditional edge functions for OmniFinan trading graph."""

from __future__ import annotations

from typing import Literal

from .state import AgentState


def should_continue_investment_debate(state: AgentState) -> Literal["bull", "bear", "judge"]:
    """Alternate bull/bear rounds and route to judge when round budget is exhausted."""
    debate_state = state["data"].get("investment_debate_state") or {}
    count = int(debate_state.get("count", 0))
    max_rounds = int(debate_state.get("max_rounds", 1))
    if count >= 2 * max_rounds:
        return "judge"
    return "bear" if count % 2 == 1 else "bull"


def should_continue_risk_review(state: AgentState) -> Literal["risk", "portfolio"]:
    """Loop risk review until max rounds, then route to execution."""
    risk_state = state["data"].get("risk_debate_state") or {}
    count = int(risk_state.get("count", 0))
    max_rounds = int(risk_state.get("max_rounds", 1))
    if count >= max_rounds:
        return "portfolio"
    return "risk"


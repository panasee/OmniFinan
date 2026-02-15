"""Macro analyst agent based on central bank and rates data."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from ..utils.progress import progress
from .state import AgentState, show_agent_reasoning


def _latest_value(macro: dict[str, Any], key: str) -> float | None:
    series = macro.get("series", {}).get(key, {})
    latest = series.get("latest")
    if not isinstance(latest, dict):
        return None
    value = latest.get("value")
    return float(value) if isinstance(value, int | float) else None


def _macro_signal_from_rates(macro: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
    fed_rate = _latest_value(macro, "fed_policy_rate")
    sofr_rate = _latest_value(macro, "sofr")
    us_cpi = _latest_value(macro, "us_cpi_yoy")
    us_unemp = _latest_value(macro, "us_unemployment_rate")
    china_lpr_1y = _latest_value(macro, "china_lpr_1y")
    shibor_3m = _latest_value(macro, "china_shibor_3m")

    bullish = 0
    bearish = 0
    notes: list[str] = []

    # US liquidity and policy stance.
    if fed_rate is not None:
        if fed_rate >= 4.5:
            bearish += 2
            notes.append(f"Fed policy rate remains restrictive ({fed_rate:.2f}%)")
        elif fed_rate <= 3.0:
            bullish += 2
            notes.append(f"Fed policy rate is accommodative ({fed_rate:.2f}%)")
    if sofr_rate is not None:
        if sofr_rate >= 4.5:
            bearish += 1
            notes.append(f"SOFR is elevated ({sofr_rate:.2f}%), tightening funding conditions")
        elif sofr_rate <= 3.0:
            bullish += 1
            notes.append(f"SOFR is easing ({sofr_rate:.2f}%), improving funding conditions")

    # US macro cycle.
    if us_cpi is not None:
        if us_cpi > 4.0:
            bearish += 1
            notes.append(f"US inflation remains high ({us_cpi:.2f}%)")
        elif us_cpi <= 3.0:
            bullish += 1
            notes.append(f"US inflation is relatively contained ({us_cpi:.2f}%)")
    if us_unemp is not None:
        if us_unemp >= 4.6:
            bearish += 1
            notes.append(f"US unemployment is softening growth ({us_unemp:.2f}%)")
        elif us_unemp <= 4.0:
            bullish += 1
            notes.append(f"US labor market remains resilient ({us_unemp:.2f}%)")

    # China policy support.
    if china_lpr_1y is not None:
        if china_lpr_1y <= 3.5:
            bullish += 1
            notes.append(f"China 1Y LPR is supportive ({china_lpr_1y:.2f}%)")
        elif china_lpr_1y >= 4.0:
            bearish += 1
            notes.append(f"China 1Y LPR is relatively tight ({china_lpr_1y:.2f}%)")
    if shibor_3m is not None and shibor_3m >= 2.5:
        bearish += 1
        notes.append(f"SHIBOR 3M is elevated ({shibor_3m:.2f}%)")

    score = bullish - bearish
    if score >= 2:
        signal = "bullish"
    elif score <= -2:
        signal = "bearish"
    else:
        signal = "neutral"

    confidence = min(0.9, 0.5 + abs(score) * 0.12)
    reasoning = {
        "score": score,
        "bullish_factors": bullish,
        "bearish_factors": bearish,
        "notes": notes or ["Macro datapoints are mixed or unavailable."],
        "macro_snapshot": {
            "fed_policy_rate": fed_rate,
            "sofr": sofr_rate,
            "us_cpi_yoy": us_cpi,
            "us_unemployment_rate": us_unemp,
            "china_lpr_1y": china_lpr_1y,
            "china_shibor_3m": shibor_3m,
        },
    }
    return signal, confidence, reasoning


def macro_analyst_agent(state: AgentState) -> AgentState:
    """Generate macro-economy signal shared across tickers."""
    data = state["data"]
    tickers = data.get("tickers", [])
    show_reasoning = state["metadata"].get("show_reasoning", False)
    macro_indicators = data.get("macro_indicators") or {"series": {}, "latest": {}}

    progress.update_status("macro_analyst_agent", None, "Analyzing macro indicators")
    signal, confidence, reasoning = _macro_signal_from_rates(macro_indicators)

    output = {
        ticker: {"signal": signal, "confidence": confidence, "reasoning": reasoning}
        for ticker in tickers
    }
    message = HumanMessage(content=json.dumps(output), name="macro_analyst_agent")

    if show_reasoning:
        show_agent_reasoning(output, "Macro Analyst Agent")

    state["data"]["analyst_signals"]["macro_analyst_agent"] = output
    progress.update_status("macro_analyst_agent", None, "Done")
    return state | {"messages": [message], "data": data}

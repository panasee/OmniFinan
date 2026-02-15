"""Tests for macro analyst agent behavior."""

from omnifinan.agents.macro import macro_analyst_agent


def test_macro_agent_emits_signal_for_each_ticker() -> None:
    state = {
        "messages": [],
        "data": {
            "tickers": ["MSFT", "00700"],
            "analyst_signals": {},
            "macro_indicators": {
                "series": {
                    "fed_policy_rate": {"latest": {"value": 5.0}},
                    "sofr": {"latest": {"value": 5.1}},
                    "us_cpi_yoy": {"latest": {"value": 3.4}},
                    "us_unemployment_rate": {"latest": {"value": 4.6}},
                    "china_lpr_1y": {"latest": {"value": 3.0}},
                    "china_shibor_3m": {"latest": {"value": 1.6}},
                }
            },
        },
        "metadata": {"show_reasoning": False},
    }

    updated = macro_analyst_agent(state)
    signals = updated["data"]["analyst_signals"]["macro_analyst_agent"]
    assert set(signals.keys()) == {"MSFT", "00700"}
    assert signals["MSFT"]["signal"] in {"bullish", "neutral", "bearish"}
    assert 0 <= signals["MSFT"]["confidence"] <= 1

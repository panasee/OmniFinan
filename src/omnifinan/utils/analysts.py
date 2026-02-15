"""Constants and utilities related to analysts configuration."""

from ..agents.fundamentals import fundamentals_agent
from ..agents.macro import macro_analyst_agent
from ..agents.sentiment import sentiment_agent
from ..agents.technicals import technical_analyst_agent
from ..agents.valuation import valuation_agent

# TradingAgents-style: only functional/domain analysts feed the debate pipeline.
# Persona agents are intentionally excluded from the default orchestration.
ANALYST_CONFIG = {
    "technical_analyst": {
        "display_name": "Technical Analyst",
        "agent_func": technical_analyst_agent,
        "order": 0,
    },
    "fundamentals_analyst": {
        "display_name": "Fundamentals Analyst",
        "agent_func": fundamentals_agent,
        "order": 1,
    },
    "macro_analyst": {
        "display_name": "Macro Analyst",
        "agent_func": macro_analyst_agent,
        "order": 2,
    },
    "sentiment_analyst": {
        "display_name": "Sentiment Analyst",
        "agent_func": sentiment_agent,
        "order": 3,
    },
    "valuation_analyst": {
        "display_name": "Valuation Analyst",
        "agent_func": valuation_agent,
        "order": 4,
    },
}

# Derive ANALYST_ORDER from ANALYST_CONFIG for backwards compatibility
ANALYST_ORDER = [
    (config["display_name"], key)
    for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
]


def get_analyst_nodes():
    """Get the mapping of analyst keys to their (node_name, agent_func) tuples."""
    return {
        key: (f"{key}_agent", config["agent_func"])
        for key, config in ANALYST_CONFIG.items()
    }


def get_default_analyst_keys() -> list[str]:
    """Return default domain analysts for TradingAgents-style pipeline."""
    return list(ANALYST_CONFIG.keys())

"""LCEL runnable factories for OmniFinan agents."""

from __future__ import annotations

from typing import Any

from pyomnix.agents.runnables import create_structured_output_chain


def create_finance_structured_chain(llm: Any, schema: type[Any], system_prompt: str):
    """Create a structured-output chain for financial agent outputs."""
    return create_structured_output_chain(
        llm=llm,
        schema=schema,
        system_prompt=system_prompt,
    )

